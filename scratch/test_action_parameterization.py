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

import zarr
ZARR_VERSION=int(zarr.__version__.split('.')[0])

from pathlib import Path

import cv2

import time
import json

import sapien

import trimesh as tm
from PIL import Image
import io
# np.set_printoptions(linewidth=np.inf)

import tqdm
import torch
import pytorch3d as p3d
from pytorch3d import transforms

def construct_env_state_dict(zarr_data, index):
    env_state_dict = dict()
    env_state_dict['actors'] = dict()
    env_state_dict['articulations'] = dict()

    for key in zarr_data['actors']:
        if key == 'camera_pose':
            pass
        else:
            env_state_dict['actors'][key] = zarr_data['actors'][key][index]

    for key in zarr_data['articulations']:
        env_state_dict['articulations'][key] = zarr_data['articulations'][key][index]

    return env_state_dict
#%%
demo_path = Path('/mnt/crucialSSD/datasetsSSD/fish_datasets/simulated/teleop/FISH/expert_demos/frankagym/FrankaInsertion-v1/206_sim_demos_leftof4thbook_springbookends_nograspedrand_noenvrand_slotrand_20hz_act')
demo = zarr.open(demo_path / 'demos.zarr', mode='r')
json_data = load_json(demo_path / 'demos.json')
#%%
joint_stiffness = 100.0
joint_damping = 2*np.sqrt(joint_stiffness)
## testing book insertion task
# env = gym.make(
#     # "LiftPegUpright-v1", 
#     "BookInsertion-v0", 
#     cam_resize_factor=0.5,
#     reward_mode="none", 
#     sim_backend='physx_cpu', 
#     render_mode="rgb_array",
#     render_contact_map=False,
#     render_dtc_maps=False,
#     render_normals_maps=False,
#     suppress_evaluation=True, 
#     book_ends_config=BookEndsConfig(
#         mode='spring',
#         height=0.25,
#         wall_height=0.25,
#         mass=1.0,
#         friction=0.0,
#         color="#808080", # default color
#         joint_stiffness=joint_stiffness, 
#         joint_damping=joint_damping,
#         travel_limit=0.125,
#     ),
#     grasped_book_config=GraspedBookConfig(
#         randomize_color=False,
#         randomize_density=False,
#         randomize_length=False,
#         randomize_height=False,
#         randomize_width=False,
#     ),
#     env_books_config=EnvBooksConfig(
#         randomize_color=False,
#         randomize_density=False,
#         randomize_height=False,
#         randomize_length=False,
#         randomize_width=False,
#     ),
#     slot_config=SlotConfig(
#         y_randomization_bounds=[-0.05, 0.05],
#     ),
#     # render_mode="sensors", 
#     render_backend="gpu",
#     obs_mode="rgb+depth+segmentation",
#     # obs_mode="none",
#     # control_mode="pd_ee_target_delta_pose",
#     control_mode="pd_ee_target_delta_pose_unnormalized",
#     # control_mode="pd_ee_delta_pose",
#     sim_config=dict(
#         sim_freq=100, # default 100
#         control_freq=20, # default 20
#         scene_config=dict(
#             solver_position_iterations=15, # 15 is the default
#             contact_offset=0.02, # 0.02 is the default
#             # contact_offset=0.02, # 0.02 is the default
#             cpu_workers=0, # 0 is the default
#         )
#     ),
#     viewer_camera_configs=dict(
#         shader_pack="minimal"
#     ),
#     human_render_camera_configs=dict(
#         shader_pack="minimal"
#     )
# )

# sim_dt = 1.0 / env.sim_config.sim_freq
# sim_dt_bw_step = sim_dt * (env.sim_config.sim_freq / env.sim_config.control_freq)

# human_render_cam_params = env.scene.human_render_cameras['render_camera'].get_params()
# human_render_cam_intrisic = human_render_cam_params['intrinsic_cv'][0]
# human_render_cam_cam2world_gl = human_render_cam_params['cam2world_gl'][0][:3, :4]
# human_render_cam_extrinsic_cv = human_render_cam_params['extrinsic_cv'][0]
#%%
snap_to_env_state = False

# for episode_idx, episode_dict in enumerate(json_data['episodes']):
# use tqdm to show progress
# for episode_idx, episode_dict in tqdm.tqdm(enumerate(json_data['episodes']), total=len(json_data['episodes'])):
episode_idx = 0
episode_dict = json_data['episodes'][episode_idx]
assert episode_dict['episode_id'] == episode_idx
# episode_idx = episode_dict['episode_id']
seed = episode_dict['episode_seed']
num_steps = episode_dict['elapsed_steps']
episode_start_idx = 0 if episode_idx == 0 else demo['meta']['episode_ends'][episode_idx - 1]
env_state_episode_start_idx = episode_start_idx + episode_idx
episode_end_idx = demo['meta']['episode_ends'][episode_idx]
env_state_episode_end_idx = episode_end_idx + episode_idx + 1
assert episode_end_idx - episode_start_idx == num_steps, f"mismatch in episode steps: {episode_end_idx - episode_start_idx} vs {num_steps}"

#%%
from mani_skill.utils.common import get_plan_target_poses_in_current_pose
action_plan_length = 28
assert num_steps >= action_plan_length, f"num_steps {num_steps} < action_plan_length {action_plan_length}"
idx_in_episode = 80
idx = episode_start_idx + idx_in_episode
env_state_idx = idx + episode_idx
#%%
pos_lower_limit = -0.1
pos_upper_limit = 0.1
rot_lower_limit = -0.1
rot_upper_limit = 0.1

action_high = np.array([pos_upper_limit, pos_upper_limit, pos_upper_limit, rot_upper_limit, rot_upper_limit, rot_upper_limit], dtype=np.float32)
action_low = np.array([pos_lower_limit, pos_lower_limit, pos_lower_limit, rot_lower_limit, rot_lower_limit, rot_lower_limit], dtype=np.float32)
# maniskill quaternions have the real part first
gt_delta_actions_normalized = torch.from_numpy(demo['data']['action'][idx:idx+action_plan_length])
gt_delta_actions = gt_delta_actions_normalized.clone()
gt_delta_actions[:, :6] = .01* (gt_delta_actions_normalized[:, :6] - 0.5 * (action_high + action_low)) / (0.5 * (action_high - action_low))
gt_delta_actions[:, 3:6] *= -1.0 # for some reason, the rotation flips sign when they normalize the actions
current_target_pose = torch.from_numpy(demo['data']['actors']['target_EE_pose'][env_state_idx:env_state_idx+1])
current_pose = torch.from_numpy(demo['data']['observation.state'][idx:idx+1, :7])
current_target_pose_in_current_pose = get_plan_target_poses_in_current_pose(current_pose, current_target_pose, rotation_representation='quaternion')

def unroll_delta_actions(delta_actions, init_pose):
    gt_target_poses_in_current_pose = torch.zeros(action_plan_length, 7, dtype=delta_actions.dtype, device=delta_actions.device)
    gt_target_poses_in_current_pose[:,:3] = init_pose[:, :3] + torch.cumsum(delta_actions[:, :3], dim=0)
    gt_delta_quaternions = transforms.axis_angle_to_quaternion(delta_actions[:, 3:6])
    for i in range(len(gt_delta_quaternions)):
        if i == 0:
            gt_target_poses_in_current_pose[i, 3:7] = transforms.quaternion_multiply(gt_delta_quaternions[i], init_pose[0, 3:7])
        else:
            gt_target_poses_in_current_pose[i, 3:7] = transforms.quaternion_multiply(gt_delta_quaternions[i], gt_target_poses_in_current_pose[i-1, 3:7])
    return gt_target_poses_in_current_pose

gt_target_poses_in_current_pose = unroll_delta_actions(gt_delta_actions, current_target_pose_in_current_pose)

plan_target_poses = torch.from_numpy(demo['data']['actors']['target_EE_pose'][env_state_idx+1:env_state_idx+1+action_plan_length])
# plan_target_poses_in_current_pose = get_plan_target_poses_in_current_pose(current_pose, plan_target_poses, rotation_representation='axis_angle')
plan_target_poses_in_current_pose = get_plan_target_poses_in_current_pose(current_pose, plan_target_poses, rotation_representation='quaternion')
#%%
from mani_skill.utils.common import get_plan_target_poses_in_current_pose_scipy
plan_target_poses_in_current_pose_scipy = get_plan_target_poses_in_current_pose_scipy(current_pose, plan_target_poses, rotation_representation='axis_angle')
#%%
import timeit
number = 5000
args = (current_pose, plan_target_poses, 'axis_angle')
time_torch3d = timeit.timeit('get_plan_target_poses_in_current_pose(*args)', globals=globals(), number=5000)
average_time_torch3d = time_torch3d / number
time_scipy = timeit.timeit('get_plan_target_poses_in_current_pose_scipy(*args)', globals=globals(), number=5000)
average_time_scipy = time_scipy / number
# put current pose and plan target poses on torch gpu
current_pose_gpu = current_pose.to('cuda')
plan_target_poses_gpu = plan_target_poses.to('cuda')
# plan_target_poses_in_current_pose_gpu = get_plan_target_poses_in_current_pose(current_pose_gpu, plan_target_poses_gpu, rotation_representation='axis_angle')
args_gpu = (current_pose_gpu, plan_target_poses_gpu, 'axis_angle')
time_torch3d_gpu = timeit.timeit('get_plan_target_poses_in_current_pose(*args_gpu)', globals=globals(), number=5000)
average_time_torch3d_gpu = time_torch3d_gpu / number
print(f'torch3d: {average_time_torch3d:.4f}s, torch3d (gpu): {average_time_torch3d_gpu:.4f}s, scipy: {average_time_scipy:.4f}s')
#%%
# def sleep_for_a_hundredth_of_a_second():
#     time.sleep(0.01)

# time_sleep = timeit.timeit('sleep_for_a_hundredth_of_a_second()', globals=globals(), number=10)
#%%
#%%
# current_target_pose = torch.from_numpy(demo['data']['actors']['target_EE_pose'][idx:idx+1])
# current_target_pose_in_current_pose = get_plan_target_poses_in_current_pose(current_pose, current_target_pose, rotation_representation='axis_angle')

# all_target_poses_in_current_pose = torch.concat((current_target_pose_in_current_pose, plan_target_poses_in_current_pose), dim=0)
# delta_actions_from_current_pose = get_delta_actions_from_plan_target_poses(all_target_poses_in_current_pose, gt_delta_actions[:, 6:], input_rotation_representation='axis_angle', output_rotation_representation='axis_angle')
#%%
# frames_with_gt = []
# obs, info = env.reset(seed=seed)

# # if snap_to_env_state:
# current_env_state_dict = construct_env_state_dict(demo['data'], env_state_episode_start_idx)
# env.set_state_dict(current_env_state_dict)

# start_time = time.perf_counter()
# # while True:
# for i in tqdm.tqdm(range(action_plan_length)):
#     # action = demo['data']['action'][episode_start_idx + i]
#     action = gt_delta_actions[i]
#     obs, reward, terminated, truncated, info = env.step(action)
#     if snap_to_env_state:
#         current_env_state_dict = construct_env_state_dict(demo['data'], env_state_episode_start_idx + i + 1)
#         env.set_state_dict(current_env_state_dict)

#     frames_with_gt.append(obs['sensor_data']['base_camera']['rgb'][0].cpu().numpy())
#     # frames_with_gt.append(env.render_rgb_array()[0].cpu().numpy())
#     elapsed_timesteps = info["elapsed_steps"].item()
#     elapsed_simtime = elapsed_timesteps * sim_dt_bw_step
#     elapsed_realtime = time.perf_counter() - start_time
#     # time_to_sleep = sim_dt_bw_step - elapsed_time
#     time_to_sleep = elapsed_simtime - elapsed_realtime
#     # if time_to_sleep > 0:
#     #     time.sleep(time_to_sleep)
#     if elapsed_timesteps % 50 == 0:
#         print(f"realtime_factor: {elapsed_simtime/elapsed_realtime} | elapsed steps: {elapsed_timesteps} | elapsed rt {elapsed_realtime} | elapsed simt {elapsed_simtime}")

# #%%
# frames_with_reconstruct_in_current_pose = []
# obs, info = env.reset(seed=seed)

# # if snap_to_env_state:
# current_env_state_dict = construct_env_state_dict(demo['data'], env_state_episode_start_idx)
# env.set_state_dict(current_env_state_dict)

# start_time = time.perf_counter()
# # while True:
# for i in tqdm.tqdm(range(action_plan_length)):
#     # action = demo['data']['action'][episode_start_idx + i]
#     action = delta_actions_from_current_pose[i]
#     obs, reward, terminated, truncated, info = env.step(action)
#     if snap_to_env_state:
#         current_env_state_dict = construct_env_state_dict(demo['data'], env_state_episode_start_idx + i + 1)
#         env.set_state_dict(current_env_state_dict)

#     frames_with_reconstruct_in_current_pose.append(obs['sensor_data']['base_camera']['rgb'][0].cpu().numpy())
#     elapsed_timesteps = info["elapsed_steps"].item()
#     elapsed_simtime = elapsed_timesteps * sim_dt_bw_step
#     elapsed_realtime = time.perf_counter() - start_time
#     # time_to_sleep = sim_dt_bw_step - elapsed_time
#     time_to_sleep = elapsed_simtime - elapsed_realtime
#     # if time_to_sleep > 0:
#     #     time.sleep(time_to_sleep)
#     if elapsed_timesteps % 50 == 0:
#         print(f"realtime_factor: {elapsed_simtime/elapsed_realtime} | elapsed steps: {elapsed_timesteps} | elapsed rt {elapsed_realtime} | elapsed simt {elapsed_simtime}")

# images_to_video(frames_with_reconstruct_in_current_pose, './', 'frames_with_reconstruct_in_current_pose.mp4', fps=20)

# #%%
# frames_with_reconstruct_in_world = []
# obs, info = env.reset(seed=seed)

# # if snap_to_env_state:
# current_env_state_dict = construct_env_state_dict(demo['data'], env_state_episode_start_idx)
# env.set_state_dict(current_env_state_dict)

# start_time = time.perf_counter()
# # while True:
# for i in tqdm.tqdm(range(action_plan_length)):
#     # action = demo['data']['action'][episode_start_idx + i]
#     action = delta_actions_from_world[i]
#     obs, reward, terminated, truncated, info = env.step(action)
#     if snap_to_env_state:
#         current_env_state_dict = construct_env_state_dict(demo['data'], env_state_episode_start_idx + i + 1)
#         env.set_state_dict(current_env_state_dict)

#     frames_with_reconstruct_in_world.append(obs['sensor_data']['base_camera']['rgb'][0].cpu().numpy())
#     elapsed_timesteps = info["elapsed_steps"].item()
#     elapsed_simtime = elapsed_timesteps * sim_dt_bw_step
#     elapsed_realtime = time.perf_counter() - start_time
#     # time_to_sleep = sim_dt_bw_step - elapsed_time
#     time_to_sleep = elapsed_simtime - elapsed_realtime
#     # if time_to_sleep > 0:
#     #     time.sleep(time_to_sleep)
#     if elapsed_timesteps % 50 == 0:
#         print(f"realtime_factor: {elapsed_simtime/elapsed_realtime} | elapsed steps: {elapsed_timesteps} | elapsed rt {elapsed_realtime} | elapsed simt {elapsed_simtime}")
# #%%
# images_to_video(frames_with_reconstruct_in_world, './', 'frames_with_reconstruct_in_world.mp4', fps=20)
# #%%

# images_to_video(frames_with_gt, './', 'frames_with_gt.mp4', fps=20)
# #%%

# #%%
# gt_images = demo['data']['observation.rgb'][idx:idx+action_plan_length]
# images_to_video(gt_images, './', 'frames_with_gt_from_demo.mp4', fps=20)
# #%%

# env.close()
# del env