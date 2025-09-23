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
from pytorch3d import transforms

from mani_skill.trajectory.dataset import ManiSkillTrajectoryDataset
from mani_skill.utils.io_utils import load_json
from mani_skill.trajectory.utils import index_dict, dict_to_list_of_dicts
from mani_skill.utils.visualization.misc import images_to_video
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.utils.wrappers.record_zarr import RecordEpisodeZarr


from mani_skill.envs.tasks.tabletop.peg_insertion_side_custom import BoxConfig, PegConfig, RobotConfig

from mani_skill.utils.wrappers.record_rerun import RecordEpisodeRerun
import multiprocessing

from pathlib import Path

import cv2

import time

# from mani_skill.utils.teleoperation import SpacemouseInput
# spacemouse_input = SpacemouseInput(sixd_mask=[1,1,1,0,0,1])
desired_viewing_size = (256, 256)

## testing book insertion task
joint_stiffness = 100.0
joint_damping = 2*np.sqrt(joint_stiffness)
env = gym.make(
    # "LiftPegUpright-v1", 
    "PegInsertionSideCustom-v1", 
    cam_resize_factor=1.0,
    reward_mode="none", 
    sim_backend='physx_cpu', 
    render_mode="rgb_array", 
    # render_mode="sensors", 
    render_backend="gpu",
    obs_mode="rgb",
    # obs_mode="none",
    render_contact_map=False,
    render_dtc_maps=False,
    render_normals_maps=False,
    render_contact_forces_map=False,
    control_mode="pd_ee_target_delta_pose",
    # control_mode="pd_ee_pose",
    # control_mode="pd_ee_target_pose",
    # control_mode="pd_ee_target_delta_pose_unnormalized",
    # control_mode="pd_ee_delta_pose",
    # urdf_config=urdf_config,
    box_config=BoxConfig(
        randomize_color=True,
        randomize_tolerance=True,
        nominal_tolerance=0.003,
        tolerance_randomization_bounds=[0.003, 0.015],
        nominal_x_position=0.45,
        randomize_x_position=True,
        x_position_delta_randomization_bounds=[-0.05, 0.05],
        nominal_y_position=0.25,
        randomize_y_position=True,
        y_position_delta_randomization_bounds=[-0.05,0.05],
        nominal_yaw=np.pi*(10/16),
        randomize_yaw=False,
        yaw_delta_randomization_bounds=[-np.pi/8, np.pi/8],
        randomize_hole_center_location=False,
        hole_center_randomization_bounds=[-1.0,1.0],
    ),
    robot_config=RobotConfig(
        init_qpos=[-0.45725486, 0.18291518, 0.16500726, -2.2905693, -0.0728711, 2.4728112, -1.0869355, 0.02300941, 0.02296073],
        gripper_friction=4.0,
        gripper_patch_radius=0.1,
    ),
    peg_config=PegConfig(
        randomize_color=False,
        randomize_length=False,
        nominal_length=0.105,
        length_randomization_bounds=[0.085,0.125],
        nominal_radius=0.02,
        randomize_radius=True,
        radius_randomization_bounds=[0.015,0.03],
    ),
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
# # ##########################
# # Render and save a video of the contact forces map overlaid on the RGB image
# # ##########################
# obs, info = env.reset(seed=0)
# force_map_overlaid = []
# blend_alpha = 0.2
# for i in tqdm.tqdm(range(15)):
# # while True:
#     action = env.action_space.sample()
#     # action, _ = spacemouse_input.get_action()
#     obs, reward, terminated, truncated, info = env.step(action)

#     # convert forces to normals coloring
#     max_force_magnitude = 0.005
#     contact_forces_map = obs['extra']['extrinsic_contact_forces_map'][0].cpu().numpy()
#     contact_forces_map = contact_forces_map / max_force_magnitude
#     contact_forces_map = np.clip(contact_forces_map, -1, 1)
#     contact_forces_map = (contact_forces_map + 1) / 2.0 * 255
#     contact_forces_map = contact_forces_map.astype(np.uint8)

#     rgb_image = obs['sensor_data']['base_camera']['rgb'][0].cpu().numpy()

#     force_map_overlay = (rgb_image* blend_alpha + contact_forces_map * (1 - blend_alpha)).astype(np.uint8)
#     force_map_overlaid.append(force_map_overlay)
# #%%
# video_path = Path("./")
# images_to_video(
#     force_map_overlaid,
#     output_dir=video_path,
#     video_name="force_map_overlaid",
#     fps=20,
#     quality=10,

# )
# #%%
# plt.imshow(rgb_image)
# plt.imshow(contact_forces_map, alpha=0.7)
# #%%
# # save plots to directory
# output_dir = Path("./force_map_plots")
# output_dir.mkdir(parents=True, exist_ok=True)
# cam_tf_world = env.base_camera_extrinsic_cv #bx4x4
# cam_rot_world = cam_tf_world[:, :3, :3] #bx3x3
# import einops
# for i in tqdm.tqdm(range(50)):
#     action = env.action_space.sample()
#     # action, _ = spacemouse_input.get_action()
#     obs, reward, terminated, truncated, info = env.step(action)
#     rgb_image = obs['sensor_data']['base_camera']['rgb'][0].cpu().numpy()
#     contact_forces_map = obs['extra']['extrinsic_contact_forces_map'] #bxHxWx3
#     # transform the forces into camera frame
#     # contact_forces_map_cam = torch.cat([
#     #     contact_forces_map,
#     #     torch.ones_like(contact_forces_map[..., :1], device=contact_forces_map.device, dtype=contact_forces_map.dtype)
#     # ], dim=-1) # bxHxWx4
#     b, h, w, c = contact_forces_map.shape
#     contact_forces_map = einops.rearrange(contact_forces_map, 'b h w c -> (b h w) c')
#     contact_forces_map = einops.rearrange(torch.matmul(cam_rot_world, contact_forces_map.unsqueeze(-1)).squeeze(-1) , '(b h w) c -> b h w c', b=b, h=h, w=w, c=c) # bxHxWx3
#     # ignore the last dimension which points into the optical axis
#     contact_forces_map_cam = contact_forces_map[..., :2]
#     # scale newtons into meters
#     contact_forces_map_cam = contact_forces_map_cam * (1.0 / 0.1)
#     # plot using quiver
#     fig, ax = plt.subplots(figsize=(20, 20))
#     ax.imshow(rgb_image)
#     # plot the contact forces using quiver
#     y, x = np.mgrid[0:h, 0:w]
#     # scale the forces to fit into the image
#     scale = 100.0
#     ax.quiver(x, y, contact_forces_map_cam[0, ..., 0].cpu().numpy() * scale, 
#             contact_forces_map_cam[0, ..., 1].cpu().numpy() * scale, 
#             angles='xy', scale_units='xy', scale=1, color='b')
#     # save the figure
#     fig.savefig(output_dir / f"force_map_{i:03d}.png")
#     plt.close(fig)
# #%%
# # read the saved images and create a video
# images_for_video = []
# for i in tqdm.tqdm(range(50)):
#     img_path = output_dir / f"force_map_{i:03d}.png"
#     img = plt.imread(img_path)
#     images_for_video.append(img)

# images_to_video(
#     images_for_video,
#     output_dir=output_dir,
#     video_name="force_map_quiver_video",
#     fps=20,
#     quality=10,
# )
# # ##########################
# # Render and save a video of the contact forces map overlaid on the RGB image
# # ##########################
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

# end_effector_target_pose = obs['agent']['controller']['arm']['target_pose']
# b = end_effector_pose.shape[0]
# delta_pose = torch.zeros((b, 6))
# delta_pose[:, :3] = end_effector_target_pose[:, :3] - end_effector_pose[:, :3]
# delta_pose[:, 3:] = transforms.quaternion_to_axis_angle(transforms.quaternion_multiply(transforms.quaternion_invert(end_effector_pose[:, 3:]), end_effector_target_pose[:, 3:]))
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

# viewer = env.render_human()
# viewer.paused = True
#%%
#%%
# plt.ion()

# force_torque_queue = np.zeros((100, 6), dtype=np.float32)

# fig, (ax_force, ax_torque) = plt.subplots(2, 1, figsize=(20, 10))
# fig.suptitle('Force and Torque on Panda Hand')

# line_fx, = ax_force.plot(force_torque_queue[:, 0], label='x')
# line_fy, = ax_force.plot(force_torque_queue[:, 1], label='y')
# line_fz, = ax_force.plot(force_torque_queue[:, 2], label='z')

# ax_force.grid()
# ax_force.set_title('Force on Panda Hand')
# ax_force.set_xlabel('Time step')
# ax_force.set_ylabel('Force (N)')
# ax_force.set_ylim(-15, 15)
# # ax_force.set_ylim(-1, 1)
# ax_force.legend()

# line_tx, = ax_torque.plot(force_torque_queue[:, 3], label='roll')
# line_ty, = ax_torque.plot(force_torque_queue[:, 4], label='pitch')
# line_tz, = ax_torque.plot(force_torque_queue[:, 5], label='yaw')
# ax_torque.grid()
# ax_torque.set_title('Torque on Panda Hand')
# ax_torque.set_xlabel('Time step')
# ax_torque.set_ylabel('Torque (Nm)')
# ax_torque.set_ylim(-10, 10)
# # ax_torque.set_ylim(-1, 1)
# ax_torque.legend()

# plt.tight_layout()
#%%
num_seeds = 50
start_seed = 0
import imageio
fps = 15
quality = 8
path_to_video = Path("./paper_peg_insertion_vids")
path_to_video.mkdir(parents=True, exist_ok=True)
with imageio.get_writer(path_to_video / 'peg_insertion_reset_dist.mp4', fps=fps, quality=quality) as video_writer:
    for seed in range(start_seed, start_seed + num_seeds):
        obs, info = env.reset(seed=seed)
        current_frame = obs['sensor_data']['base_camera']['rgb'][0].cpu().numpy()
        video_writer.append_data(current_frame)
#%%
# if record_demonstrations:
#     h5_file_path = env._h5_file.copy
#     json_file_path = env._json_path

env.close()
del env

spacemouse_input.close()

# TODO: try adding contact map to extra states and the end effector pose (to get back observation.state)