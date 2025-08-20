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
from mani_skill.envs.tasks.tabletop.peg_insertion_side_custom import BoxConfig, PegConfig, RobotConfig
from mani_skill.utils.common import batched_position_to_pixel_coordinates, get_extrinsic_contact_map_data, get_extra_contact_features, convert_sapien_pose_to_transform_matrix, get_cuboid_dict, cuboid_intersection_test
import zarr
ZARR_VERSION=int(zarr.__version__.split('.')[0])

from pathlib import Path

import cv2

import time
import json

import click

import sapien

import trimesh as tm
from PIL import Image
import io
# np.set_printoptions(linewidth=np.inf)

import tqdm

from mani_skill.envs.tasks.tabletop.book_insertion import get_book_primitive_mesh_list, convert_sapien_pose_to_transform_matrix, get_table_primitive_mesh_list, get_env_object_meshes_list
#%%
import sys, os
# add contact_estimation to the path
path_to_this_file = Path(os.path.abspath(__file__))
path_to_contact_estimation = path_to_this_file.parents[2] / "contact_estimation"
sys.path.append(str(path_to_contact_estimation))
from src.dataset.gazebo_to_trimesh import create_trimesh_camera, generate_rays_from_camera, generate_min_distances_image, normals_to_xyz_map, get_min_grasped_obj_sdf_at_env_hits_data, get_min_env_sdf_at_grasped_obj_hits_data, camera_marker_transformed

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

# add the meshes to a trimesh scene
# scene = tm.Scene(base_frame='world', camera=tm_camera, camera_transform=tm_camera.transform)
# # scene.add_geometry(table_mesh, parent_node_name='world')
# # scene.add_geometry(camera_marker, parent_node_name='world')
# scene.add_geometry(env_mesh, parent_node_name='world')
# scene.add_geometry(EE_object_mesh, parent_node_name='world')

# # scene_img = scene.save_image(resolution=(320, 240))
# scene_img = scene.save_image()
# scene_img = np.array(Image.open(io.BytesIO(scene_img)))[:240, :320]
# # plt.imshow(obs['sensor_data']['base_camera']['rgb'][0].cpu().numpy(), alpha=0.5)
# plt.imshow(env.render_rgb_array()[0].cpu().numpy(), alpha=0.5)
# # plt.imshow(obs['sensor_data']['base_camera']['segmentation'][0].cpu().numpy(), alpha=0.5)
# plt.imshow(scene_img, alpha=0.5)
# # plt.imshow(scene_img)

# # scene.show()
#%%
@click.command()
@click.argument('episode-idx', type=int)
@click.argument('path-to-demo-root-dir', type=click.Path(exists=True, path_type=Path))
@click.argument('demo-name', type=str)
@click.argument('task-gym-name', type=str)
@click.argument("record-contact-features", type=bool)
@click.argument("record-extrinsic-contact-forces-maps", type=bool)
@click.argument("record-external-end_effector-wrench", type=bool)
def main(episode_idx, path_to_demo_root_dir, demo_name, task_gym_name, record_contact_features, record_extrinsic_contact_forces_maps, record_external_end_effector_wrench):
    # path_to_demo_root_dir = Path("/mnt/crucialSSD/datasetsSSD/fish_datasets/simulated/teleop/424_sim_w_recovery_demos_leftof4thbook_springbookends_graspedrand_noenvrand_slotrand_20hz_act")
    # path_to_demo_root_dir = Path("/mnt/crucialSSD/datasetsSSD/fish_datasets/simulated/teleop/1_demo_test")
    # path_to_demo_root_dir = Path("/mnt/crucialSSD/datasetsSSD/fish_datasets/simulated/teleop/FISH/expert_demos/frankagym/FrankaInsertion-v1/2_demo_test")
    assert task_gym_name in [
        "BookInsertion-v0",
        "PegInsertionSideCustom-v1",
    ], "Invalid task gym name"

    path_to_demo_dir = path_to_demo_root_dir / demo_name

    path_to_zarr = path_to_demo_dir / "demos.zarr"
    path_to_json = path_to_demo_dir / "demos.json"

    snap_to_env_state = True
    # record_contact_features = False
    # record_extrinsic_contact_forces_maps = True
    # record_external_end_effector_wrench = True
    max_num_contact = 50
    #%%
    zarr_store = zarr.open(str(path_to_zarr), mode='r+')
    with open(path_to_json, 'r') as f:
        json_data = json.load(f)
    #%%
    # forces = np.linalg.norm(zarr_store['data']['gt_contact']['observation.contact_forces'][:], axis=-1)
    # # filter out nans
    # forces = forces[~np.isnan(forces)]
    # print(f"max contact force: {forces.max()}")
    # #%%
    # contact_forces = zarr_store['data']['gt_contact']['observation.contact_forces'][:]
    # contact_forces = contact_forces[~np.isnan(contact_forces).any(axis=-1)]
    # print(f"max contact force: {contact_forces.max()}")
    # print(f"min contact force: {contact_forces.min()}")
    # #%%
    # forces = np.linalg.norm(zarr_store['data']['observation.end_effector_external_wrench_in_world'][:, :3], axis=-1)
    # print(f"max external force at end effector: {forces.max()}")
    # torques = np.linalg.norm(zarr_store['data']['observation.end_effector_external_wrench_in_world'][:, 3:], axis=-1)
    # print(f"max external torque at end effector: {torques.max()}")
    #%%
    # episode_idx = 0
    # # save plots to directory
    # output_dir = Path("./force_map_plots")
    # output_dir.mkdir(parents=True, exist_ok=True)
    # cam_tf_world = torch.tensor(zarr_store['meta']['episode_cam_tf_world'][episode_idx])
    # cam_rot_world = cam_tf_world[:3, :3] #bx3x3
    # import einops
    # episode_start = zarr_store['meta']['episode_ends'][episode_idx - 1] if episode_idx > 0 else 0
    # episode_end = zarr_store['meta']['episode_ends'][episode_idx]
    # for i in range(episode_start, episode_end):
    #     rgb_image = zarr_store['data']['observation.rgb'][i]
    #     contact_forces_map = torch.tensor(zarr_store['data']['gt_contact']['observation.contact_forces_map'][i]) #bxHxWx3
    #     # transform the forces into camera frame
    #     # contact_forces_map_cam = torch.cat([
    #     #     contact_forces_map,
    #     #     torch.ones_like(contact_forces_map[..., :1], device=contact_forces_map.device, dtype=contact_forces_map.dtype)
    #     # ], dim=-1) # bxHxWx4
    #     h, w, c = contact_forces_map.shape
    #     contact_forces_map = einops.rearrange(contact_forces_map, 'h w c -> (h w) c')
    #     contact_forces_map = einops.rearrange(torch.matmul(cam_rot_world, contact_forces_map.unsqueeze(-1)).squeeze(-1) , '(h w) c -> h w c', h=h, w=w, c=c) # bxHxWx3
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
    #     ax.quiver(x, y, contact_forces_map_cam[..., 0].cpu().numpy() * scale, 
    #             contact_forces_map_cam[..., 1].cpu().numpy() * scale, 
    #             angles='xy', scale_units='xy', scale=1, color='b')
    #     # save the figure
    #     fig.savefig(output_dir / f"force_map_{i:03d}.png")
    #     plt.close(fig)
    # #%%
    # # read the saved images and create a video
    # images_for_video = []
    # for i in range(episode_start, episode_end):
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
    # #%%
    # force_torques = zarr_data['observation.end_effector_external_wrench_in_world'][episode_start:episode_end]
    # plt.figure(figsize=(10, 10))
    # plt.subplot(2, 1, 1)
    # plt.xlim(63,72)
    # plt.plot(force_torques[:, :3])
    # plt.title("External Force at End Effector in World Frame")
    # plt.legend(['Fx', 'Fy', 'Fz'])
    # plt.subplot(2, 1, 2)
    # plt.xlim(63,72)
    # plt.plot(force_torques[:, 3:])
    # plt.title("External Torque at End Effector in World Frame")
    # plt.legend(['Tx', 'Ty', 'Tz'])
    #%%
    keys_to_check = list()
    compressors = zarr_store['data']['observation.rgb'].compressors[0]
    # create the datasets for contact features if they don't exist
    image_shape = zarr_store['data']['observation.rgb'].shape[1:3]
    zarr_data = zarr_store['data']
    zarr_gt_contact = zarr_store['data']['gt_contact']
    if 'episode_data' not in zarr_store:
        zarr_store.create_group('episode_data')
    if f'episode_{episode_idx}' not in zarr_store['episode_data']:
        zarr_store['episode_data'].create_group(f'episode_{episode_idx}')
    zarr_episode_data = zarr_store['episode_data'][f'episode_{episode_idx}']
    if 'gt_contact' not in zarr_episode_data:
        zarr_episode_data.create_group('gt_contact')
    zarr_episode_data_gt_contact = zarr_episode_data['gt_contact']

    if record_contact_features:
        # if 'observation.EE_dtc_map' not in zarr_gt_contact:
        zarr_episode_data_gt_contact.create_array('observation.EE_dtc_map', shape=(0,) + image_shape + (1,), chunks=(1,) + image_shape + (1,), dtype=np.float32, compressor=compressors, overwrite=True)
        # if 'observation.EE_normals_map' not in zarr_gt_contact:
        zarr_episode_data_gt_contact.create_array('observation.EE_normals_map', shape=(0,) + image_shape + (3,), chunks=(1,) + image_shape + (3,), dtype=np.float32, compressor=compressors, overwrite=True)
        # if 'observation.env_dtc_map' not in zarr_gt_contact:
        zarr_episode_data_gt_contact.create_array('observation.env_dtc_map', shape=(0,) + image_shape + (1,), chunks=(1,) + image_shape + (1,), dtype=np.float32, compressor=compressors, overwrite=True)
        # if 'observation.env_normals_map' not in zarr_gt_contact:
        zarr_episode_data_gt_contact.create_array('observation.env_normals_map', shape=(0,) + image_shape + (3,), chunks=(1,) + image_shape + (3,), dtype=np.float32, compressor=compressors, overwrite=True)
        keys_to_check.append('gt_contact/observation.EE_dtc_map')
        keys_to_check.append('gt_contact/observation.EE_normals_map')
        keys_to_check.append('gt_contact/observation.env_dtc_map')
        keys_to_check.append('gt_contact/observation.env_normals_map')

    if record_extrinsic_contact_forces_maps:
        zarr_episode_data_gt_contact.create_array('observation.contact_forces_map', shape=(0,) + image_shape + (3,), chunks=(1,) + image_shape + (3,), dtype=np.float32, compressor=compressors, overwrite=True)
        zarr_episode_data_gt_contact.create_array('observation.contact_forces', shape=(0,) + (max_num_contact, 3), chunks=(1,) + (max_num_contact, 3), dtype=np.float32, compressor=compressors, overwrite=True)
        keys_to_check.append('gt_contact/observation.contact_forces_map')
        keys_to_check.append('gt_contact/observation.contact_forces')

    if record_external_end_effector_wrench:
        zarr_episode_data.create_array('observation.end_effector_external_wrench_in_world', shape=(0, 6), chunks=(1, 6), dtype=np.float32, compressor=compressors, overwrite=True)
        keys_to_check.append('observation.end_effector_external_wrench_in_world')
    #%%
    # for episode_idx, episode_dict in tqdm.tqdm(enumerate(json_data['episodes']), total=len(json_data['episodes'])):
    # # episode_idx = 100
    #     episode_dict = json_data['episodes'][episode_idx]
    #     assert episode_dict['episode_id'] == episode_idx
    #     # episode_idx = episode_dict['episode_id']
    #     seed = episode_dict['episode_seed']
    #     num_steps = episode_dict['elapsed_steps']
    #     episode_start_idx = 0 if episode_idx == 0 else zarr_store['meta']['episode_ends'][episode_idx - 1]
    #     env_state_episode_start_idx = episode_start_idx + episode_idx
    #     episode_end_idx = zarr_store['meta']['episode_ends'][episode_idx]
    #     env_state_episode_end_idx = episode_end_idx + episode_idx + 1
    #     # set the elapsed_steps in json to the number of steps in the zarr file
    #     json_data['episodes'][episode_idx]['elapsed_steps'] = int(episode_end_idx - episode_start_idx)
    #     # assert episode_end_idx - episode_start_idx == num_steps, f"mismatch in episode steps: {episode_end_idx - episode_start_idx} vs {num_steps}"

    # # update the json file with the new number of steps
    # with open(path_to_json, 'w') as f:
    #     json.dump(json_data, f, indent=4)
    #%%
    joint_stiffness = 100.0
    joint_damping = 2*np.sqrt(joint_stiffness)
    ## testing book insertion task
    if task_gym_name == "BookInsertion-v0":
        env = gym.make(
            "BookInsertion-v0", 
            # task_gym_name,
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
                y_randomization_bounds=[-0.05, 0.05],
                # y_randomization_bounds=0.0,
            ),
            # render_mode="sensors", 
            render_backend="gpu",
            obs_mode="rgb",
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
    elif task_gym_name == "PegInsertionSideCustom-v1":
        joint_stiffness = 100.0
        joint_damping = 2*np.sqrt(joint_stiffness)
        env = gym.make(
            # "LiftPegUpright-v1", 
            "PegInsertionSideCustom-v1", 
            cam_resize_factor=0.5,
            reward_mode="none", 
            sim_backend='physx_cpu', 
            render_mode="rgb_array", 
            # render_mode="sensors", 
            render_backend="gpu",
            obs_mode="rgb",
            render_contact_map=False,
            render_contact_forces_map=False,
            render_dtc_maps=False,
            render_normals_maps=False,
            suppress_evaluation=True,
            # obs_mode="none",
            control_mode="pd_ee_target_delta_pose",
            # control_mode="pd_ee_delta_pose",
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
    else:
        raise ValueError(f"Unknown task gym name: {task_gym_name}")
    #%%
    sim_dt = 1.0 / env.sim_config.sim_freq
    sim_dt_bw_step = sim_dt * (env.sim_config.sim_freq / env.sim_config.control_freq)

    human_render_cam_params = env.scene.human_render_cameras['render_camera'].get_params()
    human_render_cam_intrisic = human_render_cam_params['intrinsic_cv'][0]
    human_render_cam_cam2world_gl = human_render_cam_params['cam2world_gl'][0][:3, :4]
    human_render_cam_extrinsic_cv = human_render_cam_params['extrinsic_cv'][0]

    #%%
    # for episode_idx, episode_dict in enumerate(json_data['episodes']):
    # use tqdm to show progress
    # for episode_idx, episode_dict in tqdm.tqdm(enumerate(json_data['episodes']), total=len(json_data['episodes'])):
    # episode_idx = 152
    episode_dict = json_data['episodes'][episode_idx]
    assert episode_dict['episode_id'] == episode_idx
    # episode_idx = episode_dict['episode_id']
    seed = episode_dict['episode_seed']
    num_steps = episode_dict['elapsed_steps']
    episode_start_idx = 0 if episode_idx == 0 else zarr_store['meta']['episode_ends'][episode_idx - 1]
    env_state_episode_start_idx = episode_start_idx + episode_idx
    episode_end_idx = zarr_store['meta']['episode_ends'][episode_idx]
    env_state_episode_end_idx = episode_end_idx + episode_idx + 1
    assert episode_end_idx - episode_start_idx == num_steps, f"mismatch in episode steps: {episode_end_idx - episode_start_idx} vs {num_steps}"
    #%%
    obs, info = env.reset(seed=seed)

    if snap_to_env_state:
        current_env_state_dict = construct_env_state_dict(zarr_store['data'], env_state_episode_start_idx)
        env.set_state_dict(current_env_state_dict)
    #%%
    if record_contact_features:
        env_mesh_list, env_mesh = env.get_env_object_meshes()
        EE_object_mesh_list, EE_object_mesh = env.get_grasped_object_mesh()
        contact_features_dict = get_extra_contact_features(env_mesh_list, env_mesh, EE_object_mesh_list, EE_object_mesh, env.tm_camera, render_dtc_maps=True, render_normals_maps=True)
        zarr_episode_data_gt_contact['observation.env_dtc_map'].append(contact_features_dict['env_dtc_map'].cpu().numpy())
        zarr_episode_data_gt_contact['observation.env_normals_map'].append(contact_features_dict['env_normals_map'].cpu().numpy())
        zarr_episode_data_gt_contact['observation.EE_dtc_map'].append(contact_features_dict['EE_dtc_map'].cpu().numpy())
        zarr_episode_data_gt_contact['observation.EE_normals_map'].append(contact_features_dict['EE_normals_map'].cpu().numpy())
    
    if record_extrinsic_contact_forces_maps:
        if task_gym_name == 'BookInsertion-v0':
            grasped_object_name = 'grasped_book'
        elif task_gym_name == 'PegInsertionSideCustom-v1':
            grasped_object_name = 'peg'
        else:
            raise ValueError(f"Unknown task_gym_name: {task_gym_name}")
        contact_map_data = get_extrinsic_contact_map_data(env.num_envs, env.device, env.scene, env.max_extrinsic_contacts, env.camera_height, env.camera_width, env.base_camera_intrinsic, env.base_camera_extrinsic_cv, grasped_object_name, 'panda', return_contact_positions_map=False, return_contact_forces_map=True)
        assert contact_map_data['extrinsic_contact_forces_map'].ndim == 4, f"expected extrinsic_contact_forces_map to have shape (B, H, W, 3), got {contact_map_data['extrinsic_contact_forces_map'].shape}"
        assert contact_map_data['extrinsic_contact_forces_map'].shape[0] == 1, f"expected extrinsic_contact_forces_map to have shape (1, H, W, 3), got {contact_map_data['extrinsic_contact_forces_map'].shape}"
        assert contact_map_data['extrinsic_contact_forces_map'].shape[1:3] == image_shape, f"expected extrinsic_contact_forces_map to have shape (1, {image_shape[0]}, {image_shape[1]}, 3), got {contact_map_data['extrinsic_contact_forces_map'].shape}"
        assert contact_map_data['extrinsic_contact_forces_map'].shape[3] == 3, f"expected extrinsic_contact_forces_map to have shape (1, H, W, 3), got {contact_map_data['extrinsic_contact_forces_map'].shape}"
        zarr_episode_data_gt_contact['observation.contact_forces_map'].append(contact_map_data['extrinsic_contact_forces_map'].cpu().numpy())

        assert contact_map_data['extrinsic_contact_forces'].ndim == 3, f"expected extrinsic_contact_forces to have shape (B, N, 3), got {contact_map_data['extrinsic_contact_forces'].shape}"
        assert contact_map_data['extrinsic_contact_forces'].shape[0] == 1, f"expected extrinsic_contact_forces to have shape (1, N, 3), got {contact_map_data['extrinsic_contact_forces'].shape}"
        assert contact_map_data['extrinsic_contact_forces'].shape[1] <= max_num_contact, f"expected extrinsic_contact_forces to have shape (1, N, 3) with N <= {max_num_contact}, got {contact_map_data['extrinsic_contact_forces'].shape}"
        zarr_episode_data_gt_contact['observation.contact_forces'].append(contact_map_data['extrinsic_contact_forces'].cpu().numpy())

    if record_external_end_effector_wrench:
        W_FT_EE = env.agent.get_external_wrench_at_end_effector(in_world_frame=True)
        assert W_FT_EE.ndim == 2, f"expected W_FT_EE to have shape (B, 6), got {W_FT_EE.shape}"
        assert W_FT_EE.shape[0] == 1, f"expected W_FT_EE to have shape (1, 6), got {W_FT_EE.shape}"
        assert W_FT_EE.shape[1] == 6, f"expected W_FT_EE to have shape (1, 6), got {W_FT_EE.shape}"
        zarr_episode_data['observation.end_effector_external_wrench_in_world'].append(W_FT_EE.cpu().numpy())

    #%%
    start_time = time.perf_counter()
    # while True:
    for i in tqdm.tqdm(range(num_steps)):
        # action = env.action_space.sample()
        action = zarr_store['data']['action'][episode_start_idx + i]
        obs, reward, terminated, truncated, info = env.step(action)
        if snap_to_env_state:
            current_env_state_dict = construct_env_state_dict(zarr_store['data'], env_state_episode_start_idx + i + 1)
            env.set_state_dict(current_env_state_dict)

        # don't save if its the last step
        if i < num_steps - 1:
            #%%
            if record_contact_features:
                # the obs from the env already has a dimension at the beginning for num_envs
                env_mesh_list, env_mesh = env.get_env_object_meshes()
                EE_object_mesh_list, EE_object_mesh = env.get_grasped_object_mesh()
                contact_features_dict = get_extra_contact_features(env_mesh_list, env_mesh, EE_object_mesh_list, EE_object_mesh, env.tm_camera, render_dtc_maps=True, render_normals_maps=True)
                zarr_episode_data_gt_contact['observation.env_dtc_map'].append(contact_features_dict['env_dtc_map'].cpu().numpy())
                zarr_episode_data_gt_contact['observation.env_normals_map'].append(contact_features_dict['env_normals_map'].cpu().numpy())
                zarr_episode_data_gt_contact['observation.EE_dtc_map'].append(contact_features_dict['EE_dtc_map'].cpu().numpy())
                zarr_episode_data_gt_contact['observation.EE_normals_map'].append(contact_features_dict['EE_normals_map'].cpu().numpy())

            if record_extrinsic_contact_forces_maps:
                contact_map_data = get_extrinsic_contact_map_data(env.num_envs, env.device, env.scene, env.max_extrinsic_contacts, env.camera_height, env.camera_width, env.base_camera_intrinsic, env.base_camera_extrinsic_cv, grasped_object_name, 'panda', return_contact_positions_map=False, return_contact_forces_map=True)
                assert contact_map_data['extrinsic_contact_forces_map'].ndim == 4, f"expected extrinsic_contact_forces_map to have shape (B, H, W, 3), got {contact_map_data['extrinsic_contact_forces_map'].shape}"
                assert contact_map_data['extrinsic_contact_forces_map'].shape[0] == 1, f"expected extrinsic_contact_forces_map to have shape (1, H, W, 3), got {contact_map_data['extrinsic_contact_forces_map'].shape}"
                assert contact_map_data['extrinsic_contact_forces_map'].shape[1:3] == image_shape, f"expected extrinsic_contact_forces_map to have shape (1, {image_shape[0]}, {image_shape[1]}, 3), got {contact_map_data['extrinsic_contact_forces_map'].shape}"
                assert contact_map_data['extrinsic_contact_forces_map'].shape[3] == 3, f"expected extrinsic_contact_forces_map to have shape (1, H, W, 3), got {contact_map_data['extrinsic_contact_forces_map'].shape}"
                zarr_episode_data_gt_contact['observation.contact_forces_map'].append(contact_map_data['extrinsic_contact_forces_map'].cpu().numpy())

                assert contact_map_data['extrinsic_contact_forces'].ndim == 3, f"expected extrinsic_contact_forces to have shape (B, N, 3), got {contact_map_data['extrinsic_contact_forces'].shape}"
                assert contact_map_data['extrinsic_contact_forces'].shape[0] == 1, f"expected extrinsic_contact_forces to have shape (1, N, 3), got {contact_map_data['extrinsic_contact_forces'].shape}"
                assert contact_map_data['extrinsic_contact_forces'].shape[1] <= max_num_contact, f"expected extrinsic_contact_forces to have shape (1, N, 3) with N <= {max_num_contact}, got {contact_map_data['extrinsic_contact_forces'].shape}"
                zarr_episode_data_gt_contact['observation.contact_forces'].append(contact_map_data['extrinsic_contact_forces'].cpu().numpy())

            if record_external_end_effector_wrench:
                W_FT_EE = env.agent.get_external_wrench_at_end_effector(in_world_frame=True)
                assert W_FT_EE.ndim == 2, f"expected W_FT_EE to have shape (B, 6), got {W_FT_EE.shape}"
                assert W_FT_EE.shape[0] == 1, f"expected W_FT_EE to have shape (1, 6), got {W_FT_EE.shape}"
                assert W_FT_EE.shape[1] == 6, f"expected W_FT_EE to have shape (1, 6), got {W_FT_EE.shape}"
                zarr_data['observation.end_effector_external_wrench_in_world'].append(W_FT_EE.cpu().numpy())

        # # frames.append(current_frame)
        # elapsed_timesteps = info["elapsed_steps"].item()
        # elapsed_simtime = elapsed_timesteps * sim_dt_bw_step
        # elapsed_realtime = time.perf_counter() - start_time
        # # time_to_sleep = sim_dt_bw_step - elapsed_time
        # time_to_sleep = elapsed_simtime - elapsed_realtime
        # # if time_to_sleep > 0:
        # #     time.sleep(time_to_sleep)
        # if elapsed_timesteps % 50 == 0:
        #     print(f"realtime_factor: {elapsed_simtime/elapsed_realtime} | elapsed steps: {elapsed_timesteps} | elapsed rt {elapsed_realtime} | elapsed simt {elapsed_simtime}")
    
    # assert zarr_gt_contact['observation.env_dtc_map'].shape[0] == episode_end_idx, f"mismatch in number of contact features: {zarr_gt_contact['observation.env_dtc_map'].shape[0]} vs {episode_end_idx}"

    for key in keys_to_check:
        assert key in zarr_episode_data, f"key {key} not found in zarr_episode_data"
        assert zarr_episode_data[key].shape[0] == num_steps, f"mismatch in number of contact features for {key}: {zarr_episode_data[key].shape[0]} vs {num_steps}"
    #%%
    # if key == ord('q'):
    #     break
    # elif key == ord('c'):
    #     # seed += 1
    #     # num_trajs += 1
    #     # env.reset(seed=seed)
    #     # # viewer = env.render_human()
    #     # spacemouse_input.reset()
    #     continue
    # # elif key == ord('r'):
    #     # env.reset(seed=seed, options=dict(save_trajectory=False))
    #     # # viewer = env.render_human()
    #     # spacemouse_input.reset()
    #     # continue
    # # else:
    # #     break
    # assert 

    # cv2.destroyAllWindows()
    #%%
    # if record_demonstrations:
    #     h5_file_path = env._h5_file.copy
    #     json_file_path = env._json_path

    env.close()
    del env

if __name__ == "__main__":
    main()
    print(f"Saved masks to {path_to_zarr}.")
    print("Done.")