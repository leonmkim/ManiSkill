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
import h5py

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
        self.translation_factor = 0.02
        self.rotation_factor = 0.02
        
    def get_action(self):
        spacemouse_input = pyspacemouse.read()
        return self.spacemouse_input_function(spacemouse_input)

    def spacemouse_input_to_delta_pose(self, spacemouse_event):
        delta_pose = np.zeros(7)
        # delta_pose[0] = -spacemouse_event.y * self.translation_factor
        delta_pose[1] = spacemouse_event.x * self.translation_factor
        delta_pose[2] = spacemouse_event.z * self.translation_factor
        delta_pose[3] = spacemouse_event.roll * self.rotation_factor
        # delta_pose[4] = spacemouse_event.pitch * self.rotation_factor
        # delta_pose[5] = spacemouse_event.yaw * self.rotation_factor
        if spacemouse_event.buttons[1]:
            self.gripper_action = -self.gripper_action
        delta_pose[6] = self.gripper_action

        return delta_pose

    def apply_spacemouse_input_to_target_pose(target_pose, spacemouse_event):
        target_pose[:3] += spacemouse_event[:3]*0.01
        target_pose[3:] += spacemouse_event[3:]*0.01
        return target_pose
spacemouse_input = SpacemouseInput()
#%%
## testing book insertion task
env = gym.make(
                # "LiftPegUpright-v1", 
                "BookInsertion-v0", 
               reward_mode="none", 
               sim_backend='physx_cpu', 
               render_mode="rgb_array", 
                control_mode="pd_ee_target_delta_pose",
                # control_mode="pd_ee_delta_pose",
                sim_config=dict(
                    sim_freq=100,
                ),
                viewer_camera_configs=dict(
                    shader_pack="minimal"
                ),
                human_render_camera_configs=dict(
                    shader_pack="minimal"
               )
)
            #    control_mode="pd_ee_pose"
#%%
sim_dt = 1 / env.sim_config.sim_freq
sim_dt_bw_step = sim_dt * (env.sim_config.sim_freq / env.sim_config.control_freq)
# %%
env.reset()
#%%
target_pose = env.agent.tcp.pose
#%%

#%%
# cv2.namedWindow("frame", cv2.WINDOW_AUTOSIZE)
# frame = cv2.cvtColor(env.render_rgb_array()[0].cpu().numpy(), cv2.COLOR_RGB2BGR)
# cv2.imshow("frame", frame)
# plt.imshow(frame)
viewer = env.render_human()
# viewer.paused = True
#%%
# frames = [env.render_rgb_array()[0].cpu().numpy()]
# for i in tqdm.tqdm(range(500)):
start_time = time.perf_counter()
while True:
    # action = env.action_space.sample()
    action = spacemouse_input.get_action()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render_human()
    # current_frame = cv2.cvtColor(env.render_rgb_array()[0].cpu().numpy(), cv2.COLOR_RGB2BGR)
    # cv2.imshow("frame", current_frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    
    # frames.append(current_frame)
    elapsed_timesteps = info["elapsed_steps"].item()
    elapsed_simtime = elapsed_timesteps * sim_dt
    elapsed_realtime = time.perf_counter() - start_time
    # time_to_sleep = sim_dt_bw_step - elapsed_time
    time_to_sleep = elapsed_simtime - elapsed_realtime
    if time_to_sleep > 0:
        time.sleep(time_to_sleep)
    if elapsed_timesteps % 100 == 0:
        print(f"realtime_factor: {elapsed_simtime/elapsed_realtime}")
    
#%%
env.close()
# cv2.destroyAllWindows()
# %%
# images_to_video(frames, output_dir=".", video_name="book_insertion", fps=30, )

# %%
