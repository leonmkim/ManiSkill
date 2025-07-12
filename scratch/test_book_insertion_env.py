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

from mani_skill.utils.wrappers.record_rerun import RecordEpisodeRerun
import multiprocessing

from pathlib import Path

import cv2

import time

from mani_skill.utils.teleoperation import SpacemouseInput
spacemouse_input = SpacemouseInput(sixd_mask=[0,0,0,0,0,0])
desired_viewing_size = (256, 256)

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
    render_contact_map=False,
    render_dtc_maps=False,
    render_normals_maps=False,
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
        randomize_color=False,
        randomize_density=False,
        randomize_length=False,
        randomize_height=False,
        randomize_width=False,
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
    ),
    # obs_mode="none",
    control_mode="pd_ee_target_delta_pose",
    # control_mode="pd_ee_pose",
    # control_mode="pd_ee_target_pose",
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
#%%
obs, info = env.reset(seed=0)
#%%
import numpy as np
from scipy.spatial.transform import Rotation as R

gripper_dims = np.array([
    [0.032, 0.1, .033], # x, y, z (width, depth, height) main body
    [.018/2, .027/2, .054/2], # x, y, z (width, depth, height) finger
    [.018/2, .027/2, .054/2], # x, y, z (width, depth, height) finger
])

z_bottom_of_finger_to_center = 0.009
z_top_of_finger_to_bottom_of_body = 0.007

gripper_centers = np.array([
    [0, 0, -(gripper_dims[0, 2] + (gripper_dims[1,2]*2 - z_bottom_of_finger_to_center) - z_top_of_finger_to_bottom_of_body)], # main body
    [0, gripper_dims[1,1], -(gripper_dims[1,2]-z_bottom_of_finger_to_center)], # finger 1
    [0, -gripper_dims[1,1], -(gripper_dims[1,2]-z_bottom_of_finger_to_center)], # finger 2
])

gripper_orientations = R.from_quat([
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
])
#%%
world_tf_gripper_posquat = env.get_state_dict()['actors']['target_EE_pose'][0].cpu().numpy()[:7]
world_tf_gripper = np.eye(4)
world_tf_gripper[:3, :3] = R.from_quat(world_tf_gripper_posquat[3:], scalar_first=True).as_matrix()
world_tf_gripper[:3, 3] = world_tf_gripper_posquat[:3]
world_tf_gripper_rot = R.from_quat(world_tf_gripper_posquat[3:], scalar_first=True)
#%%
# transform box centers and orientations to world frame
new_gripper_centers = (world_tf_gripper[:3, :3] @ gripper_centers.T).T + world_tf_gripper[:3, 3]

new_gripper_orientations = R.from_matrix(world_tf_gripper[:3, :3] @ gripper_orientations.as_matrix())
#%%
new_gripper_centers_alt = world_tf_gripper_rot.apply(gripper_centers) + world_tf_gripper[:3, 3]
new_gripper_orientations_alt = world_tf_gripper_rot*gripper_orientations
#%%
rerun_output_dir = Path("/mnt/crucialSSD/maniskill_evals/rerun_test")
import datetime
rerun_output_dir = rerun_output_dir / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# env = RecordEpisodeRerun(
#     env,
#     output_dir=rerun_output_dir,
# )
# env = gym.make(
#     # "LiftPegUpright-v1", 
#     "SpringArticulationEnv-v0", 
#     reward_mode="none", 
#     sim_backend='physx_cpu', 
#     render_mode="rgb_array", 
#     # render_mode="sensors", 
#     render_backend="gpu",
#     obs_mode="rgb",
#     control_mode="pd_ee_target_delta_pose",
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
# seed = 0
#%%
seed = 1_000_000
num_trajs = 0
#%%
sim_dt = 1.0 / env.sim_config.sim_freq
sim_dt_bw_step = sim_dt * (env.sim_config.sim_freq / env.sim_config.control_freq)

# human_render_cam_params = env.scene.human_render_cameras['render_camera'].get_params()
# human_render_cam_intrisic = human_render_cam_params['intrinsic_cv'][0]
# human_render_cam_cam2world_gl = human_render_cam_params['cam2world_gl'][0][:3, :4]
# human_render_cam_extrinsic_cv = human_render_cam_params['extrinsic_cv'][0]
# #%%
# base_camera_cam2world_gl = env.scene.sensors['base_camera'].get_params()['cam2world_gl'][0]

# base_camera_extrinsic_cv = env.scene.sensors['base_camera'].get_params()['extrinsic_cv'][0]
#%% 
obs, info = env.reset(seed=seed)
#%%
# frame = obs['sensor_data']['base_camera']['segmentation'][0].cpu().numpy()
# frame = env.render_rgb_array()[0].cpu().numpy()
# plt.imshow(frame)
# plt.imshow(obs['sensor_data']['base_camera']['Color'][0][:,:,3].cpu().numpy())
#%%
# cv2.namedWindow("frame", cv2.WINDOW_AUTOSIZE)

# frame = obs['sensor_data']['base_camera']['rgb'][0].cpu().numpy()
# frame = (frame*0.5 + obs['extra']['extrinsic_contact_map'][0].cpu().numpy()*255*0.5).astype(np.uint8)
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
        action, _ = spacemouse_input.get_action()
        obs, reward, terminated, truncated, info = env.step(action)

        env.render_human()

        # current_frame = cv2.cvtColor(env.render_rgb_array()[0].cpu().numpy(), cv2.COLOR_RGB2BGR)
        # current_frame = obs['sensor_data']['base_camera']['rgb'][0].cpu().numpy()
        # current_frame = obs['sensor_data']['base_camera']['Color'][0][:,:,:3].cpu().numpy()

        # current_frame = (current_frame*0.5 + obs['extra']['extrinsic_contact_map'][0].cpu().numpy()*255*0.5).astype(np.uint8)
        # current_frame = cv2.resize(current_frame, desired_viewing_size, interpolation=cv2.INTER_NEAREST)

        # cv2.imshow("frame", current_frame)
        # key = cv2.waitKey(1) & 0xFF
        # if key == ord('q') or key == ord('c') or key == ord('r'):
        #     break
        
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
        if time_to_sleep > 0:
            time.sleep(time_to_sleep)
        if elapsed_timesteps % 50 == 0:
            print(f"realtime_factor: {elapsed_simtime/elapsed_realtime} | elapsed steps: {elapsed_timesteps} | elapsed rt {elapsed_realtime} | elapsed simt {elapsed_simtime}")
            # print(f"success: {info['success']} | success duration: {info['elapsed_success_duration']} | t. success: {info['transient_success']} | z_distance: {info['z_distance_bw_top_of_grasped_book_and_top_of_slot']}")
    
    if key == ord('q'):
        num_trajs += 1
        break
    elif key == ord('c'):
        seed += 1
        num_trajs += 1
        env.reset(seed=seed)
        viewer = env.render_human()
        spacemouse_input.reset()
        continue
    elif key == ord('r'):
        env.reset(seed=seed, options=dict(save_trajectory=False))
        viewer = env.render_human()
        spacemouse_input.reset()
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

# TODO: try adding contact map to extra states and the end effector pose (to get back observation.state)