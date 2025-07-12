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
from mani_skill.envs.tasks.tabletop.book_insertion import GraspedBookConfig, BookEndsConfig, EnvBooksConfig, SlotConfig

from pathlib import Path

import cv2

import time

from mani_skill.utils.teleoperation import SpacemouseInput

import logging
record_logger = logging.getLogger("record_logger")
#%%
path_to_demo = Path("/mnt/crucialSSD/datasetsSSD/fish_datasets/simulated/teleop/20250606_154009.zarr")
import zarr
demo = zarr.open(path_to_demo, mode='r')

#%%
spacemouse_input = SpacemouseInput(sixd_mask=[0,1,1,1,0,0])
# spacemouse_input = SpacemouseInput(sixd_mask=[0,0,0,1,1,0])
desired_viewing_size = (256, 256)
output_dir = Path("/mnt/crucialSSD/datasetsSSD/fish_datasets/simulated/teleop")
record_demonstrations = True
record_video = False

#%%
## testing book insertion task
joint_stiffness = 100.0
joint_damping = 2*np.sqrt(joint_stiffness)
env = gym.make(
    # "LiftPegUpright-v1", 
    "BookInsertion-v0", 
    cam_resize_factor=0.5,
    reward_mode="none", 
    sim_backend='physx_cpu', 
    render_mode="rgb_array", 
    # render_mode="sensors", 
    render_backend="gpu",
    obs_mode="rgb+depth+segmentation",
    render_contact_map=True,
    render_dtc_maps=False,
    render_normals_maps=False,
    suppress_evaluation=True,
    book_ends_config=BookEndsConfig(
        mode='spring',
        height=0.25,
        wall_height=0.25,
        mass=1.0,
        friction=0.0,
        color="#808080", # default color
        joint_stiffness=joint_stiffness, 
        joint_damping=joint_damping,
        travel_limit=0.125,
    ),
    grasped_book_config=GraspedBookConfig(
        randomize_color=True,
        randomize_density=False,
        randomize_length=False,
        randomize_height=True,
        randomize_width=True,
    ),
    env_books_config=EnvBooksConfig(
        randomize_color=False,
        randomize_density=False,
        randomize_height=False,
        randomize_length=False,
        randomize_width=False,
    ),
    slot_config=SlotConfig(
        y_randomization_bounds=[-0.05, 0.05],
        # y_randomization_bounds=0.0,
    ),
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
        save_video=record_video,
        save_trajectory=True,
        info_on_video=False,
        record_reward=False,
        video_fps=20,
        source_type="teleoperation",
        source_desc="teleoperation via spacemouse",
        save_grasped_book_info=True,
        save_env_book_info=True,
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
# segmentation_id_exclusion_list = list()
# for i in range(9):
#     segmentation_id_exclusion_list.append(f"panda_link{i}")
# segmentation_id_exclusion_list.extend(["panda_hand", "panda_hand_tcp", "panda_leftfinger", "panda_rightfinger", "panda_leftfinger_pad", "panda_rightfinger_pad"])
# segmentation_id_exclusion_list.extend(["target_EE_pose", "camera_pose"])
# segmentation_map_ids = dict()
# for key, value in env.segmentation_id_map.items():
#     entity_name = value.name
#     if entity_name not in segmentation_id_exclusion_list:
#         segmentation_map_ids[entity_name] = key

# %%
obs, info = env.reset(seed=seed)
#%%
# frame = obs['sensor_data']['base_camera']['segmentation'][0].cpu().numpy()
# frame = env.render_rgb_array()[0].cpu().numpy()
# plt.imshow(frame)
# plt.imshow(obs['sensor_data']['base_camera']['Color'][0][:,:,3].cpu().numpy())
#%%
cv2.namedWindow("frame", cv2.WINDOW_AUTOSIZE)
frame = cv2.cvtColor(env.render_rgb_array()[0].cpu().numpy(), cv2.COLOR_RGB2BGR)
cv2.imshow("frame", frame)

# frame = obs['sensor_data']['base_camera']['rgb'][0].cpu().numpy()
# frame = (frame*0.5 + obs['extra']['extrinsic_contact_map'][0].cpu().numpy()*255*0.5).astype(np.uint8)
# frame = cv2.cvtColor(env.render()[0].cpu().numpy(), cv2.COLOR_RGB2BGR)

# frame = cv2.resize(frame, desired_viewing_size, interpolation=cv2.INTER_NEAREST)
# plt.imshow(frame)

# viewer = env.render_human()
# viewer.paused = True
#%%
# frames = [env.render_rgb_array()[0].cpu().numpy()]
# for i in tqdm.tqdm(range(500)):
while True:
# for i in range(1):
    start_time = time.perf_counter()
    while True:
    # for j in range(25):
        # action = env.action_space.sample()
        # start_signal = None
        action, start_signal = spacemouse_input.get_action()
        obs, reward, terminated, truncated, info = env.step(action[np.newaxis, :], start_signal=start_signal)

        # env.render_human()

        current_frame = cv2.cvtColor(env.render_rgb_array()[0].cpu().numpy(), cv2.COLOR_RGB2BGR)
        # current_frame = obs['sensor_data']['base_camera']['rgb'][0].cpu().numpy()
        # current_frame = obs['sensor_data']['base_camera']['Color'][0][:,:,:3].cpu().numpy()

        # current_frame = (current_frame*0.5 + obs['extra']['extrinsic_contact_map'][0].cpu().numpy()*255*0.5).astype(np.uint8)
        # current_frame = cv2.resize(current_frame, desired_viewing_size, interpolation=cv2.INTER_NEAREST)

        cv2.imshow("frame", current_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('c') or key == ord('r'):
            break
        
        # if viewer.window.key_press('q'):
        #     # q: quit the script and stop collecting data. Save trajectories and optionally videos.
        #     # c: stop this episode and record the trajectory and move on to a new episode
        #     # r: restart
        #     key = ord('q')
        #     break
        # elif viewer.window.key_press('c'): 
        #     key = ord('c')
        #     break
        # elif viewer.window.key_press('r'):
        #     key = ord('r')
        #     break

        # frames.append(current_frame)
        elapsed_timesteps = info["elapsed_steps"].item()
        elapsed_simtime = elapsed_timesteps * sim_dt_bw_step
        elapsed_realtime = time.perf_counter() - start_time
        time_to_sleep = elapsed_simtime - elapsed_realtime
        if time_to_sleep > 0:
            time.sleep(time_to_sleep)
        if elapsed_timesteps % 50 == 0:
            record_logger.info(f"realtime_factor: {elapsed_simtime/elapsed_realtime} | elapsed steps: {elapsed_timesteps} | elapsed rt {elapsed_realtime} | elapsed simt {elapsed_simtime}")
    
    if record_demonstrations:
        # pass
        if key == ord('q'):
            num_trajs += 1
            break
        elif key == ord('c'):
            seed += 1
            num_trajs += 1
            env.reset(seed=seed)
            record_logger.info(f"starting new episode with seed {seed}")
            # viewer = env.render_human()
            spacemouse_input.reset()
            continue
        elif key == ord('r'):
            env.reset(seed=seed, options=dict(save_trajectory=False))
            record_logger.info(f"restarting episode with seed {seed}")
            # viewer = env.render_human()
            spacemouse_input.reset()
            continue
    else:
        break

cv2.destroyAllWindows()
if key == ord('q'):
    # dont save the trajectory
    env.reset(seed=seed, options=dict(save_trajectory=False))
#%%
# if record_demonstrations:
#     h5_file_path = env._h5_file.copy
#     json_file_path = env._json_path

env.close()
del env

# spacemouse_input.close()

# TODO: try adding contact map to extra states and the end effector pose (to get back observation.state)