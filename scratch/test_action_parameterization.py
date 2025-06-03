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
from mani_skill.utils.common import get_delta_actions_from_plan_target_poses

import zarr
ZARR_VERSION=int(zarr.__version__.split('.')[0])

from pathlib import Path

from scipy.spatial.transform import Rotation as R
from pytorch3d import transforms

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

demo_path = Path('/mnt/crucialSSD/maniskill_evals/frankagym_pixels/FrankaInsertion-v1/106788_0/150000/seed_start_1000000_seed_end_1000003/2025-05-31_18-49-08/20250531_184911.zarr')
demo = zarr.open(demo_path, mode='r')
#%%
# demo_path = Path('/mnt/crucialSSD/datasetsSSD/fish_datasets/simulated/teleop/FISH/expert_demos/frankagym/FrankaInsertion-v1/206_sim_demos_leftof4thbook_springbookends_nograspedrand_noenvrand_slotrand_20hz_act')
# demo_path = Path('/mnt/crucialSSD/datasetsSSD/fish_datasets/simulated/teleop/1_sim_demos_w_recovery_leftof4thbook_springbookends_nograspedrand_noenvrand_slotrand_20hz_act')
demo_path = Path('/mnt/crucialSSD/datasetsSSD/fish_datasets/simulated/teleop/test_action_reparameterization_just_pitch_then_roll_1_demo')
# demo_path = Path('/mnt/crucialSSD/datasetsSSD/fish_datasets/simulated/teleop/test_action_reparameterization_just_roll_1_demo')

vid_output_dir = Path('.') / demo_path.name
vid_output_dir.mkdir(exist_ok=True, parents=True)
demo = zarr.open(demo_path / 'demos.zarr', mode='r')
json_data = load_json(demo_path / 'demos.json')
#%%
rgb_images = demo['data']['observation.rgb']

images_to_video(
    rgb_images, 
    vid_output_dir, 
    'original_demo_frames.mp4', 
    fps=20, 
)
#%%
joint_stiffness = 100.0
joint_damping = 2*np.sqrt(joint_stiffness)
## testing book insertion task
env = gym.make(
    # "LiftPegUpright-v1", 
    "BookInsertion-v0", 
    cam_resize_factor=0.5,
    reward_mode="none", 
    sim_backend='physx_cpu', 
    render_mode="rgb_array",
    render_contact_map=False,
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
        # y_randomization_bounds=[-0.05, 0.05],
        y_randomization_bounds=0.0,
    ),
    # render_mode="sensors", 
    render_backend="gpu",
    obs_mode="rgb+depth+segmentation",
    # obs_mode="none",
    # control_mode="pd_ee_target_pose",
    # control_mode="pd_ee_target_delta_pose",
    control_mode="pd_ee_target_delta_pose_unnormalized",
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

sim_dt = 1.0 / env.sim_config.sim_freq
sim_dt_bw_step = sim_dt * (env.sim_config.sim_freq / env.sim_config.control_freq)

human_render_cam_params = env.scene.human_render_cameras['render_camera'].get_params()
human_render_cam_intrisic = human_render_cam_params['intrinsic_cv'][0]
human_render_cam_cam2world_gl = human_render_cam_params['cam2world_gl'][0][:3, :4]
human_render_cam_extrinsic_cv = human_render_cam_params['extrinsic_cv'][0]
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
from mani_skill.utils.common import apply_transform_to_poses
action_plan_length = len(demo['data']['action'][episode_start_idx:episode_start_idx+num_steps])-1
assert num_steps >= action_plan_length, f"num_steps {num_steps} < action_plan_length {action_plan_length}"
idx_in_episode = 0
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
#%%
# current_target_pose = torch.from_numpy(demo['data']['actors']['target_EE_pose'][env_state_idx:env_state_idx+1, :7])
# plan_target_poses = torch.from_numpy(demo['data']['actors']['target_EE_pose'][env_state_idx+1:env_state_idx+1+action_plan_length, :7])
current_target_pose = torch.from_numpy(demo['data']['observation.target_pose'][env_state_idx:env_state_idx+1, :7])
plan_target_poses = torch.from_numpy(demo['data']['observation.target_pose'][env_state_idx+1:env_state_idx+1+action_plan_length, :7])
all_target_poses = torch.concat((current_target_pose, plan_target_poses), dim=0)
delta_actions_from_plan_target_poses_in_axis_angle = get_delta_actions_from_plan_target_poses(all_target_poses, gripper_actions=gt_delta_actions[:, -1], input_rotation_representation='quaternion', output_rotation_representation='axis_angle')
delta_actions_from_plan_target_poses_in_euler_angles = get_delta_actions_from_plan_target_poses(all_target_poses, gripper_actions=gt_delta_actions[:, -1], input_rotation_representation='quaternion', output_rotation_representation='euler_angles')
#%%
trans_actions = gt_delta_actions[:, :3]
# plot the time series of the rotation actions
plt.figure(figsize=(10, 5))
plt.plot(trans_actions[:, 0], label='x')
plt.plot(trans_actions[:, 1], label='y')
plt.plot(trans_actions[:, 2], label='z')
plt.xlabel('Time step')
plt.ylabel('Translation action')
plt.title('Translation Actions Over Time')
plt.legend()
plt.grid()
#%%
trans_actions = delta_actions_from_plan_target_poses_in_axis_angle[:, :3]
# plot the time series of the rotation actions
plt.figure(figsize=(10, 5))
plt.plot(trans_actions[:, 0], label='x')
plt.plot(trans_actions[:, 1], label='y')
plt.plot(trans_actions[:, 2], label='z')
plt.xlabel('Time step')
plt.ylabel('Translation action')
plt.title('Translation Actions Over Time')
plt.legend()
plt.grid()
#%%
trans_actions = delta_actions_from_plan_target_poses_in_euler_angles[:, :3]
# plot the time series of the rotation actions
plt.figure(figsize=(10, 5))
plt.plot(trans_actions[:, 0], label='x')
plt.plot(trans_actions[:, 1], label='y')
plt.plot(trans_actions[:, 2], label='z')
plt.xlabel('Time step')
plt.ylabel('Rotation action')
plt.title('Rotation Actions Over Time')
plt.legend()
plt.grid()
#%%
rot_actions = gt_delta_actions[:, 3:6]
# plot the time series of the rotation actions
plt.figure(figsize=(10, 5))
plt.plot(rot_actions[:, 0], label='x')
plt.plot(rot_actions[:, 1], label='y')
plt.plot(rot_actions[:, 2], label='z')
plt.xlabel('Time step')
plt.ylabel('Rotation action')
plt.title('Rotation Actions Over Time')
plt.legend()
plt.grid()
#%%
# rot_actions = delta_actions_from_plan_target_poses_in_axis_angle[:, 3:6]
rot_actions = delta_actions_from_plan_target_poses_in_axis_angle[:, 3:6] - gt_delta_actions[:, 3:6]
# plot the time series of the rotation actions
plt.figure(figsize=(10, 5))
plt.plot(rot_actions[:, 0], label='x')
plt.plot(rot_actions[:, 1], label='y')
plt.plot(rot_actions[:, 2], label='z')
plt.xlabel('Time step')
plt.ylabel('Rotation action')
plt.title('Rotation Actions Over Time')
plt.legend()
plt.grid()
#%%
# rot_actions = delta_actions_from_plan_target_poses_in_euler_angles[:, 3:6]
rot_actions = delta_actions_from_plan_target_poses_in_euler_angles[:, 3:6] - gt_delta_actions[:, 3:6]
# plot the time series of the rotation actions
plt.figure(figsize=(10, 5))
plt.plot(rot_actions[:, 0], label='x')
plt.plot(rot_actions[:, 1], label='y')
plt.plot(rot_actions[:, 2], label='z')
plt.xlabel('Time step')
plt.ylabel('Rotation action')
plt.title('Rotation Actions Over Time')
plt.legend()
plt.grid()
#%%
gt_delta_actions_in_quaternion = transforms.axis_angle_to_quaternion(gt_delta_actions[:, 3:6])
gt_delta_rotations_in_matrix = transforms.axis_angle_to_matrix(gt_delta_actions[:, 3:6])
gt_delta_rotations_in_euler_angles = transforms.matrix_to_euler_angles(gt_delta_rotations_in_matrix, convention='XYZ')
gt_delta_rotations_from_euler_angles_to_matrix = transforms.euler_angles_to_matrix(gt_delta_rotations_in_euler_angles, convention='XYZ')
gt_delta_rotations_from_matrix_to_axis_angle = transforms.matrix_to_axis_angle(gt_delta_rotations_in_matrix)
gt_delta_rotations_from_matrix_to_quaternion = transforms.matrix_to_quaternion(gt_delta_rotations_in_matrix)
print(f"max error in axis angle: {torch.max(torch.abs(gt_delta_rotations_from_matrix_to_axis_angle - gt_delta_actions[:, 3:6]))}")
print(f"max error in quaternion: {torch.max(torch.abs(gt_delta_rotations_from_matrix_to_quaternion - gt_delta_actions_in_quaternion))}")
#%%
# plot the time series of the rotation actions
plt.figure(figsize=(10, 5))
plt.plot(gt_delta_rotations_from_matrix_to_axis_angle[:, 0], label='x')
plt.plot(gt_delta_rotations_from_matrix_to_axis_angle[:, 1], label='y')
plt.plot(gt_delta_rotations_from_matrix_to_axis_angle[:, 2], label='z')
plt.xlabel('Time step')
plt.ylabel('Rotation action')
plt.title('Rotation Actions Over Time')
plt.legend()
plt.grid()
#%%
# now test recovering in scipy
gt_delta_rotations_from_euler_angles_to_axis_angle = torch.from_numpy(R.from_euler('XYZ', gt_delta_rotations_in_euler_angles.cpu().numpy(), degrees=False).as_rotvec())
gt_delta_rotations_from_euler_angles_to_quaternion = torch.from_numpy(R.from_euler('XYZ', gt_delta_rotations_in_euler_angles.cpu().numpy(), degrees=False).as_quat(scalar_first=True))
print(f"max error in axis angle: {torch.max(torch.abs(gt_delta_rotations_from_euler_angles_to_axis_angle - gt_delta_actions[:, 3:6]))}")
print(f"max error in quaternion: {torch.max(torch.abs(gt_delta_rotations_from_euler_angles_to_quaternion - gt_delta_actions_in_quaternion))}")
#%%
# plot the time series of the rotation actions
plt.figure(figsize=(10, 5))
plt.plot(gt_delta_rotations_from_euler_angles_to_axis_angle[:, 0], label='x')
plt.plot(gt_delta_rotations_from_euler_angles_to_axis_angle[:, 1], label='y')
plt.plot(gt_delta_rotations_from_euler_angles_to_axis_angle[:, 2], label='z')
plt.xlabel('Time step')
plt.ylabel('Rotation action')
plt.title('Rotation Actions Over Time')
plt.legend()
plt.grid()

#%%
EE_orientations_in_quaternion = torch.from_numpy(demo['data']['observation.state'][idx:idx+action_plan_length, 3:7]) # maniskill uses scalar/real first convention
EE_orientations_in_axis_angle = transforms.quaternion_to_axis_angle(EE_orientations_in_quaternion)
EE_orientations_in_matrix = transforms.quaternion_to_matrix(EE_orientations_in_quaternion)
EE_orientations_in_euler_angles = transforms.matrix_to_euler_angles(EE_orientations_in_matrix, convention='XYZ')
EE_orientations_from_euler_angles_to_matrix = transforms.euler_angles_to_matrix(EE_orientations_in_euler_angles, convention='XYZ')
EE_orientations_from_matrix_to_quaternion = transforms.matrix_to_quaternion(EE_orientations_in_matrix)
# find where the max error in quaternion is
quaternion_error = torch.linalg.norm(torch.abs(EE_orientations_from_matrix_to_quaternion - EE_orientations_in_quaternion), dim=1)
index_with_max_error = torch.argmax(quaternion_error)

print(f"max error in matrix: {torch.max(torch.abs(EE_orientations_from_euler_angles_to_matrix - EE_orientations_in_matrix))}")
print(f"max error in quaternion: {torch.max(torch.abs(EE_orientations_from_matrix_to_quaternion - EE_orientations_in_quaternion))}")
#%%
EE_orientations_from_euler_angles_to_quaternion = torch.from_numpy(R.from_euler('XYZ', EE_orientations_in_euler_angles.cpu().numpy(), degrees=False).as_quat(scalar_first=True))
EE_orientations_from_euler_angles_to_axis_angle = torch.from_numpy(R.from_euler('XYZ', EE_orientations_in_euler_angles.cpu().numpy(), degrees=False).as_rotvec())
quaternion_error = torch.linalg.norm(torch.abs(EE_orientations_from_euler_angles_to_quaternion - EE_orientations_in_quaternion), dim=1)
axis_angle_error = torch.linalg.norm(torch.abs(EE_orientations_from_euler_angles_to_axis_angle - EE_orientations_in_axis_angle), dim=1)
print(f"max error in quaternion: {torch.max(torch.abs(EE_orientations_from_euler_angles_to_quaternion - EE_orientations_in_quaternion))}")
print(f"max error in axis angle: {torch.max(torch.abs(EE_orientations_from_euler_angles_to_axis_angle - EE_orientations_in_axis_angle))}")
#%%
# plot the quaternion error
plt.figure(figsize=(10, 5))
plt.plot(axis_angle_error, label='quaternion error')
plt.xlabel('Time step')
plt.ylabel('Quaternion error')
plt.title('Quaternion Error Over Time')
plt.legend()
plt.grid()
#%%
frames_with_gt = []
obs, info = env.reset(seed=seed)

# if snap_to_env_state:
current_env_state_dict = construct_env_state_dict(demo['data'], env_state_episode_start_idx)
env.set_state_dict(current_env_state_dict)

start_time = time.perf_counter()
# while True:
for i in tqdm.tqdm(range(action_plan_length)):
    # action = demo['data']['action'][episode_start_idx + i]
    action = gt_delta_actions[i]
    obs, reward, terminated, truncated, info = env.step(action)
    if snap_to_env_state:
        current_env_state_dict = construct_env_state_dict(demo['data'], env_state_episode_start_idx + i + 1)
        env.set_state_dict(current_env_state_dict)

    frames_with_gt.append(obs['sensor_data']['base_camera']['rgb'][0].cpu().numpy())
    # frames_with_gt.append(env.render_rgb_array()[0].cpu().numpy())
    elapsed_timesteps = info["elapsed_steps"].item()
    elapsed_simtime = elapsed_timesteps * sim_dt_bw_step
    elapsed_realtime = time.perf_counter() - start_time
    # time_to_sleep = sim_dt_bw_step - elapsed_time
    time_to_sleep = elapsed_simtime - elapsed_realtime
    # if time_to_sleep > 0:
    #     time.sleep(time_to_sleep)
    if elapsed_timesteps % 50 == 0:
        print(f"realtime_factor: {elapsed_simtime/elapsed_realtime} | elapsed steps: {elapsed_timesteps} | elapsed rt {elapsed_realtime} | elapsed simt {elapsed_simtime}")
images_to_video(frames_with_gt, vid_output_dir, 'frames_with_gt.mp4', fps=20)
#%%

joint_stiffness = 100.0
joint_damping = 2*np.sqrt(joint_stiffness)
## testing book insertion task
env = gym.make(
    # "LiftPegUpright-v1", 
    "BookInsertion-v0", 
    cam_resize_factor=0.5,
    reward_mode="none", 
    sim_backend='physx_cpu', 
    render_mode="rgb_array",
    render_contact_map=False,
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
        # y_randomization_bounds=[-0.05, 0.05],
        y_randomization_bounds=0.0,
    ),
    # render_mode="sensors", 
    render_backend="gpu",
    obs_mode="rgb+depth+segmentation",
    # obs_mode="none",
    # control_mode="pd_ee_pose",
    control_mode="pd_ee_target_pose",
    # control_mode="pd_ee_target_delta_pose",
    # control_mode="pd_ee_target_delta_pose_unnormalized",
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
obs, info = env.reset(seed=seed)

#%%
current_target_pose = torch.from_numpy(demo['data']['actors']['target_EE_pose'][env_state_idx:env_state_idx+1, :7])
current_pose = torch.from_numpy(demo['data']['observation.state'][idx:idx+1, :7])
current_target_pose_in_current_pose = apply_transform_to_poses(current_pose, current_target_pose, rotation_representation='quaternion', mode='subtract')
#%%
from mani_skill.utils.common import unroll_delta_actions

gt_actions_in_world = gt_delta_actions.clone()
gt_actions_in_world[:, :-1] = unroll_delta_actions(gt_delta_actions[:, :6].unsqueeze(0), current_target_pose, output_rotation_representation='euler_angles')[0]
#%%
frames_with_gt = []
obs, info = env.reset(seed=seed)

# if snap_to_env_state:
current_env_state_dict = construct_env_state_dict(demo['data'], env_state_episode_start_idx)
env.set_state_dict(current_env_state_dict)

start_time = time.perf_counter()
# while True:
for i in tqdm.tqdm(range(action_plan_length)):
    # action = demo['data']['action'][episode_start_idx + i]
    action = gt_actions_in_world[i]
    obs, reward, terminated, truncated, info = env.step(action)
    if snap_to_env_state:
        current_env_state_dict = construct_env_state_dict(demo['data'], env_state_episode_start_idx + i + 1)
        env.set_state_dict(current_env_state_dict)

    frames_with_gt.append(obs['sensor_data']['base_camera']['rgb'][0].cpu().numpy())
    # frames_with_gt.append(env.render_rgb_array()[0].cpu().numpy())
    elapsed_timesteps = info["elapsed_steps"].item()
    elapsed_simtime = elapsed_timesteps * sim_dt_bw_step
    elapsed_realtime = time.perf_counter() - start_time
    # time_to_sleep = sim_dt_bw_step - elapsed_time
    time_to_sleep = elapsed_simtime - elapsed_realtime
    # if time_to_sleep > 0:
    #     time.sleep(time_to_sleep)
    if elapsed_timesteps % 50 == 0:
        print(f"realtime_factor: {elapsed_simtime/elapsed_realtime} | elapsed steps: {elapsed_timesteps} | elapsed rt {elapsed_realtime} | elapsed simt {elapsed_simtime}")
images_to_video(frames_with_gt, vid_output_dir, 'frames_with_gt_in_world.mp4', fps=20)
# #%%
plan_target_poses = torch.from_numpy(demo['data']['actors']['target_EE_pose'][env_state_idx+1:env_state_idx+1+action_plan_length])
# plan_target_poses_in_current_pose = get_plan_target_poses_in_current_pose(current_pose, plan_target_poses, rotation_representation='axis_angle')
plan_target_poses_in_current_pose = apply_transform_to_poses(current_pose, plan_target_poses, rotation_representation='quaternion')
from mani_skill.utils.common import get_plan_target_poses_in_current_pose_scipy
plan_target_poses_in_current_pose_scipy = get_plan_target_poses_in_current_pose_scipy(current_pose, plan_target_poses, rotation_representation='axis_angle')
#%%

#%%
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
# %%
