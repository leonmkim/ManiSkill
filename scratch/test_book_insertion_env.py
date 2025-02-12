#%%
import gymnasium as gym
import mani_skill.envs
import time
from mani_skill.utils.wrappers import CPUGymWrapper
import matplotlib.pyplot as plt
import torch
import tqdm
import numpy as np
from IPython.display import Video

from mani_skill.trajectory.dataset import ManiSkillTrajectoryDataset
from mani_skill.utils.io_utils import load_json
from mani_skill.trajectory.utils import index_dict, dict_to_list_of_dicts
from mani_skill.utils.visualization.misc import images_to_video
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.utils.wrappers.record_zarr import RecordEpisodeZarr

import multiprocessing

from pathlib import Path

import cv2

import pyspacemouse

import time
#%%
class SpacemouseInput:
    def __init__(
            self,
            mode:str='delta_pose',
            start_gripper_closed=True,
    ):
        success = pyspacemouse.open()
        if not success:
            print("No SpaceMouse Found")
            exit()

        supported_modes = ['delta_pose', 'target_pose']
        assert mode in ['delta_pose', 'target_pose'], f"mode should be in {supported_modes}, got {mode}"
        self.spacemouse_input_function = self.spacemouse_input_to_delta_pose if mode == 'delta_pose' else self.apply_spacemouse_input_to_target_pose

        self.mode = mode
        self.start_gripper_closed = start_gripper_closed
        self.gripper_action = -1 if start_gripper_closed else 1
        # self.translation_factor = 0.02
        # self.rotation_factor = 0.02

        self.translation_factor = 0.1
        self.rotation_factor = 0.15

        self.button_timeout = 0.4
        self.last_button_press_time = time.perf_counter()
        
        self.manager = multiprocessing.Manager()
        self.current_spacemouse_state = self.manager.dict()

        self.read_spacemouse_process = multiprocessing.Process(target=self.read_spacemouse)
        self.read_spacemouse_process.start()
    
    def read_spacemouse(self):
        while True:
            state = pyspacemouse.read()
            self.current_spacemouse_state.update({
                'x': state.x,
                'y': state.y,
                'z': state.z,
                'roll': state.roll,
                'pitch': state.pitch,
                'yaw': state.yaw,
                'buttons': state.buttons,
            })
            # print(self.current_spacemouse_state)
            time.sleep(.001)
                
    def get_action(self):
        spacemouse_input = self.current_spacemouse_state
        return self.spacemouse_input_function(spacemouse_input)

    def spacemouse_input_to_delta_pose(self, spacemouse_event):
        delta_pose = np.zeros(7)
        # delta_pose[0] = -spacemouse_event.y * self.translation_factor
        delta_pose[1] = spacemouse_event['x'] * self.translation_factor
        delta_pose[2] = spacemouse_event['z'] * self.translation_factor
        delta_pose[3] = spacemouse_event['roll'] * self.rotation_factor
        # delta_pose[4] = spacemouse_event.pitch * self.rotation_factor
        # delta_pose[5] = spacemouse_event.yaw * self.rotation_factor
        if spacemouse_event['buttons'][1] and time.perf_counter() - self.last_button_press_time > self.button_timeout:
            self.gripper_action = -self.gripper_action
            self.last_button_press_time = time.perf_counter()
        delta_pose[6] = self.gripper_action

        return delta_pose

    def apply_spacemouse_input_to_target_pose(target_pose, spacemouse_event):
        target_pose[:3] += spacemouse_event[:3]*0.01
        target_pose[3:] += spacemouse_event[3:]*0.01
        return target_pose
    
    def close(self):
        pyspacemouse.close()
        self.read_spacemouse_process.terminate()
        self.read_spacemouse_process.join()
    
spacemouse_input = SpacemouseInput()
desired_viewing_size = (256, 256)
output_dir = Path("/mnt/crucialSSD/datasetsSSD/fish_datasets/simulated/teleop")
record_demonstrations = True

#%%
## testing book insertion task
env = gym.make(
    # "LiftPegUpright-v1", 
    "BookInsertion-v0", 
    cam_resize_factor=0.5,
    reward_mode="none", 
    sim_backend='physx_cpu', 
    # render_mode="rgb_array", 
    # render_mode="sensors", 
    obs_mode="rgb+depth+segmentation",
    # obs_mode="none",
    control_mode="pd_ee_target_delta_pose",
    # control_mode="pd_ee_delta_pose",
    sim_config=dict(
        sim_freq=100, # default 100
        control_freq=20, # default 20
        scene_config=dict(
            solver_position_iterations=15, # 15 is the default
            contact_offset=0.02, # 0.02 is the default
            # contact_offset=0.02, # 0.02 is the default
            cpu_workers=0, # 0 is the default
        )
    ),
    viewer_camera_configs=dict(
        shader_pack="minimal"
    ),
    human_render_camera_configs=dict(
        shader_pack="minimal"
    )
)
if record_demonstrations:
    # env = RecordEpisode(
    env = RecordEpisodeZarr(
        env,
        output_dir=output_dir,
        save_video=False,
        info_on_video=False,
        record_reward=False,
        source_type="teleoperation",
        source_desc="teleoperation via spacemouse",
    )
seed = 0
num_trajs = 0
#%%
sim_dt = 1.0 / env.sim_config.sim_freq
sim_dt_bw_step = sim_dt * (env.sim_config.sim_freq / env.sim_config.control_freq)

human_render_cam_params = env.scene.human_render_cameras['render_camera'].get_params()
human_render_cam_intrisic = human_render_cam_params['intrinsic_cv'][0]
human_render_cam_cam2world_gl = human_render_cam_params['cam2world_gl'][0][:3, :4]
human_render_cam_extrinsic_cv = human_render_cam_params['extrinsic_cv'][0]
#%%
base_camera_cam2world_gl = env.scene.sensors['base_camera'].get_params()['cam2world_gl'][0]
base_camera_extrinsic_cv = env.scene.sensors['base_camera'].get_params()['extrinsic_cv'][0]
#%% 
segmentation_id_exclusion_list = list()
for i in range(9):
    segmentation_id_exclusion_list.append(f"panda_link{i}")
segmentation_id_exclusion_list.extend(["panda_hand", "panda_hand_tcp", "panda_leftfinger", "panda_rightfinger", "panda_leftfinger_pad", "panda_rightfinger_pad"])
segmentation_id_exclusion_list.extend(["target_EE_pose", "camera_pose"])
segmentation_map_ids = dict()
for key, value in env.segmentation_id_map.items():
    entity_name = value.name
    if entity_name not in segmentation_id_exclusion_list:
        segmentation_map_ids[entity_name] = key

# %%
obs, info = env.reset(seed=seed)
#%%
# frame = obs['sensor_data']['base_camera']['segmentation'][0].cpu().numpy()
# frame = env.render_rgb_array()[0].cpu().numpy()
# plt.imshow(frame)
# plt.imshow(obs['sensor_data']['base_camera']['Color'][0][:,:,3].cpu().numpy())
#%%
# cv2.namedWindow("frame", cv2.WINDOW_AUTOSIZE)

# frame = obs['sensor_data']['base_camera']['rgb'][0].cpu().numpy()
# frame = (frame*0.5 + obs['extra']['extrinsic_contact_map'][0].cpu().numpy()[..., None]*255*0.5).astype(np.uint8)
# frame = cv2.cvtColor(env.render_rgb_array()[0].cpu().numpy(), cv2.COLOR_RGB2BGR)
# frame = cv2.cvtColor(env.render()[0].cpu().numpy(), cv2.COLOR_RGB2BGR)

# frame = cv2.resize(frame, desired_viewing_size, interpolation=cv2.INTER_NEAREST)
# cv2.imshow("frame", frame)
# plt.imshow(frame)

viewer = env.render_human()
# viewer.paused = True
#%%
# frames = [env.render_rgb_array()[0].cpu().numpy()]
# for i in tqdm.tqdm(range(500)):
while True:
    start_time = time.perf_counter()
    while True:
        # action = env.action_space.sample()
        action = spacemouse_input.get_action()
        obs, reward, terminated, truncated, info = env.step(action)

        env.render_human()

        # current_frame = cv2.cvtColor(env.render_rgb_array()[0].cpu().numpy(), cv2.COLOR_RGB2BGR)
        # current_frame = obs['sensor_data']['base_camera']['rgb'][0].cpu().numpy()
        # current_frame = obs['sensor_data']['base_camera']['Color'][0][:,:,:3].cpu().numpy()

        # current_frame = (current_frame*0.5 + obs['extra']['extrinsic_contact_map'][0].cpu().numpy()[..., None]*255*0.5).astype(np.uint8)
        # current_frame = cv2.resize(current_frame, desired_viewing_size, interpolation=cv2.INTER_NEAREST)

        # cv2.imshow("frame", current_frame)
        # key = cv2.waitKey(1) & 0xFF
        # if key == ord('q') or key == ord('c') or key == ord('r'):
        if viewer.window.key_press('q'):
            # q: quit the script and stop collecting data. Save trajectories and optionally videos.
            # c: stop this episode and record the trajectory and move on to a new episode
            # r: restart
            key = ord('q')
            break
        elif viewer.window.key_press('c'): 
            key = ord('c')
            break
        elif viewer.window.key_press('r'):
            key = ord('r')
            break

        # frames.append(current_frame)
        elapsed_timesteps = info["elapsed_steps"].item()
        elapsed_simtime = elapsed_timesteps * sim_dt_bw_step
        elapsed_realtime = time.perf_counter() - start_time
        # time_to_sleep = sim_dt_bw_step - elapsed_time
        time_to_sleep = elapsed_simtime - elapsed_realtime
        # if time_to_sleep > 0:
        #     time.sleep(time_to_sleep)
        if elapsed_timesteps % 100 == 0:
            print(f"realtime_factor: {elapsed_simtime/elapsed_realtime} | elapsed steps: {elapsed_timesteps} | elapsed rt {elapsed_realtime} | elapsed simt {elapsed_simtime}")
    
    if record_demonstrations:
        if key == ord('q'):
            num_trajs += 1
            break
        elif key == ord('c'):
            seed += 1
            num_trajs += 1
            env.reset(seed=seed)
            viewer = env.render_human()
            continue
        elif key == ord('r'):
            env.reset(seed=seed, options=dict(save_trajectory=False))
            viewer = env.render_human()
            continue
    else:
        break

cv2.destroyAllWindows()
#%%
# if record_demonstrations:
#     h5_file_path = env._h5_file.copy
#     json_file_path = env._json_path

env.close()
del env

spacemouse_input.close()
