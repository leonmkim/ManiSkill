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

import zarr
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
path_to_demo_dataset = Path('/mnt/bighdd/fish_contact_backup/expert_demos/192_240x320_multiobj_twodim_recovery_5hz_zstd7_EE_pxl_coords_expert_demos_imp_act/demos.zarr')
demo_dataset_zarr = zarr.open(str(path_to_demo_dataset), mode='r')
#%%
desired_viewing_size = (256, 256)
path_to_demo_root_dir = Path("/mnt/crucialSSD/datasetsSSD/fish_datasets/simulated/teleop/FISH/expert_demos/frankagym/FrankaInsertion-v1/413_sim_demos_left_of_4th_book_20hz_act")
path_to_zarr = path_to_demo_root_dir / "demos.zarr"
path_to_json = path_to_demo_root_dir / "demos.json"

snap_to_env_state = True
record_contact_features = True
#%%
zarr_store = zarr.open(str(path_to_zarr), mode='r+')
with open(path_to_json, 'r') as f:
    json_data = json.load(f)
#%%
compressor = zarr_store.data['observation.rgb'].compressor
# create the datasets for contact features if they don't exist
image_shape = zarr_store.data['observation.rgb'].shape[1:3]
zarr_gt_contact = zarr_store.data.gt_contact
if record_contact_features:
    if 'observation.EE_dtc_map' not in zarr_gt_contact:
        zarr_gt_contact.create_dataset('observation.EE_dtc_map', shape=(0,) + image_shape + (1,), chunks=(1,) + image_shape + (1,), dtype=np.float32, compressor=compressor, overwrite=True)
    if 'observation.EE_normals_map' not in zarr_gt_contact:
        zarr_gt_contact.create_dataset('observation.EE_normals_map', shape=(0,) + image_shape + (3,), chunks=(1,) + image_shape + (3,), dtype=np.float32, compressor=compressor, overwrite=True)
    if 'observation.env_dtc_map' not in zarr_gt_contact:
        zarr_gt_contact.create_dataset('observation.env_dtc_map', shape=(0,) + image_shape + (1,), chunks=(1,) + image_shape + (1,), dtype=np.float32, compressor=compressor, overwrite=True)
    if 'observation.env_normals_map' not in zarr_gt_contact:
        zarr_gt_contact.create_dataset('observation.env_normals_map', shape=(0,) + image_shape + (3,), chunks=(1,) + image_shape + (3,), dtype=np.float32, compressor=compressor, overwrite=True)

#%%

## testing book insertion task
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
for episode_idx, episode_dict in tqdm.tqdm(enumerate(json_data['episodes']), total=len(json_data['episodes'])):
# episode_idx = 0
    episode_dict = json_data['episodes'][episode_idx]
    assert episode_dict['episode_id'] == episode_idx
    # episode_idx = episode_dict['episode_id']
    seed = episode_dict['episode_seed']
    num_steps = episode_dict['elapsed_steps']
    episode_start_idx = 0 if episode_idx == 0 else zarr_store.meta.episode_ends[episode_idx - 1]
    env_state_episode_start_idx = episode_start_idx + episode_idx
    episode_end_idx = zarr_store.meta.episode_ends[episode_idx]
    env_state_episode_end_idx = episode_end_idx + episode_idx + 1
    assert episode_end_idx - episode_start_idx == num_steps, f"mismatch in episode steps: {episode_end_idx - episode_start_idx} vs {num_steps}"
    #%%
    obs, info = env.reset(seed=seed)

    if snap_to_env_state:
        current_env_state_dict = construct_env_state_dict(zarr_store.data, env_state_episode_start_idx)
        env.set_state_dict(current_env_state_dict)
    # plt.imshow(obs['sensor_data']['base_camera']['rgb'][0].cpu().numpy())

    base_camera_intrinsic_cv = env.scene.sensors['base_camera'].get_params()['intrinsic_cv'][0].clone()

    base_camera_cam2world_gl = env.scene.sensors['base_camera'].get_params()['cam2world_gl'][0].clone() # this is world to cam
    tm_camera = create_trimesh_camera(base_camera_intrinsic_cv, base_camera_cam2world_gl.cpu().numpy())

    #%%
    # # frame = obs['sensor_data']['base_camera']['rgb'][0].cpu().numpy()
    # # frame = (frame*0.5 + obs['extra']['extrinsic_contact_map'][0].cpu().numpy()*255*0.5).astype(np.uint8)
    # # frame = cv2.cvtColor(env.render_rgb_array()[0].cpu().numpy(), cv2.COLOR_RGB2BGR)
    # # frame = cv2.cvtColor(env.render()[0].cpu().numpy(), cv2.COLOR_RGB2BGR)
    # frame = cv2.cvtColor(env.render_sensors()[0,:, :320].cpu().numpy(), cv2.COLOR_RGB2BGR)
    # recorded_frame = zarr_store.data['observation.rgb'][episode_start_idx]
    # # assert (recorded_frame == frame).all(), "mismatch in recorded frame"
    # # print(np.ptp(recorded_frame - frame))
    # # recorded_frame = zarr_store.data.gt_segmentation['observation.EE_obj_mask'][episode_start_idx]
    # # repeat to match 3 channels
    # # recorded_frame = np.repeat(recorded_frame[:, :, np.newaxis], 3, axis=2)
    # # max_segmentation_id = recorded_frame.max()
    # # recorded_frame = (recorded_frame / max_segmentation_id * 255).astype(np.uint8)

    # frame = (frame*0.5 + recorded_frame*0.5).astype(np.uint8)
    # # frame = (frame*recorded_frame).astype(np.uint8)

    # # frame = cv2.resize(frame, desired_viewing_size, interpolation=cv2.INTER_NEAREST)
    # # cv2.imshow("frame", frame)
    # # plt.imshow(frame)

    # # viewer = env.render_human()
    # # viewer.paused = True
    # # #%%

    # # camera_marker = camera_marker_transformed(tm_camera)
    #%%
    # make a book using the sizes and poses from the env

    # EE_object_mesh = env.non_merged_grasped_books_list[0].get_collision_meshes()[0]
    # EE_object_transform = env.non_merged_grasped_books_list[0].pose

    length, width, height = env.grasped_book_sizes[0].tolist()
    binding_thickness = env.binding_thickness
    cover_thickness = env.cover_thickness
    cover_overhang = env.cover_overhang
    EE_object_transform = convert_sapien_pose_to_transform_matrix(env.non_merged_grasped_books_list[0].pose)

    EE_object_mesh_list = get_book_primitive_mesh_list(length, width, height, binding_thickness, cover_thickness, cover_overhang, global_transform=EE_object_transform)
    EE_object_mesh = tm.util.concatenate(EE_object_mesh_list)
    #%%
    env_book_sizes_list = [env.env_book_sizes[0,i].tolist() for i in range(env.non_merged_env_books_list)]
    env_book_poses_list = [convert_sapien_pose_to_transform_matrix(env_book_over_envs[0].pose) for env_book_over_envs in env.non_merged_env_books_list]
    env_object_meshes_list = get_env_object_meshes_list(env_book_sizes_list, env_book_poses_list, binding_thickness, cover_thickness, cover_overhang)

    # table_mesh = env.table_scene.table.get_collision_meshes()

    table_length, table_width, table_height = env.table_scene.table_length, env.table_scene.table_width, env.table_scene.table_height
    table_pose = convert_sapien_pose_to_transform_matrix(env.table_scene.table.pose)
    table_mesh = get_table_primitive_mesh_list(table_length, table_width, table_height, global_transform=table_pose)
    #%%
    env_mesh = tm.util.concatenate(env_object_meshes_list + table_mesh)
    env_mesh_list = env_object_meshes_list + table_mesh
    #%%

    ray_origins, ray_directions, pixels_uv = generate_rays_from_camera(tm_camera)
    env_hit_min_locations, env_hit_min_pixels_uv, env_hit_min_distances, env_hit_min_index_tri, env_hit_min_ray_directions = get_min_grasped_obj_sdf_at_env_hits_data(ray_origins, ray_directions, pixels_uv, env_mesh, EE_object_mesh_list)
    EE_obj_sdf_on_env_image, EE_obj_sdf_on_env_mask = generate_min_distances_image(env_hit_min_pixels_uv, env_hit_min_distances, tm_camera.resolution[::-1])
    EE_obj_sdf_on_env_image = EE_obj_sdf_on_env_image.astype(np.float32)[:240, :320, np.newaxis]
    EE_obj_sdf_on_env_mask = EE_obj_sdf_on_env_mask.astype(bool)[:240, :320]
    assert EE_obj_sdf_on_env_image.shape == image_shape + (1,)
    # plt.imshow(obs['sensor_data']['base_camera']['rgb'][0].cpu().numpy(), alpha=0.5)
    # plt.imshow(EE_obj_sdf_on_env_image)
    # plt.imshow(scene_img, alpha=0.3)
    #%%
    min_env_surface_normals = env_mesh.face_normals[env_hit_min_index_tri]
    env_xyz_normals_image, env_xyz_normals_image_mask = normals_to_xyz_map(min_env_surface_normals, tm_camera.resolution[::-1], env_hit_min_pixels_uv)#, fill_value=1.0/np.sqrt(3.0))
    env_xyz_normals_image = env_xyz_normals_image.astype(np.float32)[:240, :320]
    env_xyz_normals_image_mask = env_xyz_normals_image_mask.astype(bool)[:240, :320]
    # plt.imshow(env_xyz_normals_image, alpha=0.5)
    # plt.imshow(obs['sensor_data']['base_camera']['rgb'][0].cpu().numpy(), alpha=0.5)
    assert env_xyz_normals_image.shape == image_shape + (3,)
    #%%
    EE_obj_hit_min_locations, EE_obj_hit_min_pixels_uv, EE_obj_hit_min_distances, EE_obj_hit_min_index_tri, EE_obj_hit_min_ray_directions = get_min_env_sdf_at_grasped_obj_hits_data(ray_origins, ray_directions, pixels_uv, env_mesh_list, EE_object_mesh)
    env_sdf_on_EE_obj_image, env_sdf_on_EE_obj_mask = generate_min_distances_image(EE_obj_hit_min_pixels_uv, EE_obj_hit_min_distances, tm_camera.resolution[::-1])
    env_sdf_on_EE_obj_image = env_sdf_on_EE_obj_image.astype(np.float32)[:240, :320, np.newaxis]
    env_sdf_on_EE_obj_mask = env_sdf_on_EE_obj_mask.astype(bool)[:240, :320]
    # plt.imshow(obs['sensor_data']['base_camera']['rgb'][0].cpu().numpy(), alpha=0.5)
    # plt.imshow(env_sdf_on_EE_obj_image)
    assert env_sdf_on_EE_obj_image.shape == image_shape + (1,)
    #%%
    min_EE_object_surface_normals = EE_object_mesh.face_normals[EE_obj_hit_min_index_tri] # these are normalized already
    EE_object_xyz_normals_image, EE_object_xyz_normals_image_mask = normals_to_xyz_map(min_EE_object_surface_normals, tm_camera.resolution[::-1], EE_obj_hit_min_pixels_uv)#, fill_value=1.0/np.sqrt(3.0))
    EE_object_xyz_normals_image = EE_object_xyz_normals_image.astype(np.float32)[:240, :320]
    EE_object_xyz_normals_image_mask = EE_object_xyz_normals_image_mask.astype(bool)[:240, :320]

    # plt.imshow(EE_object_xyz_normals_image, alpha=0.5)
    # plt.imshow(obs['sensor_data']['base_camera']['rgb'][0].cpu().numpy(), alpha=0.5)
    assert EE_object_xyz_normals_image.shape == image_shape + (3,)
    #%%
    if record_contact_features:
        zarr_gt_contact['observation.env_dtc_map'].append(EE_obj_sdf_on_env_image[np.newaxis, ...])
        zarr_gt_contact['observation.env_normals_map'].append(env_xyz_normals_image[np.newaxis, ...])
        zarr_gt_contact['observation.EE_dtc_map'].append(env_sdf_on_EE_obj_image[np.newaxis, ...])
        zarr_gt_contact['observation.EE_normals_map'].append(EE_object_xyz_normals_image[np.newaxis, ...])

    #%%
    # frames = [env.render_rgb_array()[0].cpu().numpy()]
    # for i in tqdm.tqdm(range(500)):
    start_time = time.perf_counter()
    # while True:
    for i in range(num_steps):
        # action = env.action_space.sample()
        action = zarr_store.data.action[episode_start_idx + i]
        obs, reward, terminated, truncated, info = env.step(action)
        if snap_to_env_state:
            current_env_state_dict = construct_env_state_dict(zarr_store.data, env_state_episode_start_idx + i + 1)
            env.set_state_dict(current_env_state_dict)

        # current_frame = cv2.cvtColor(env.render_sensors()[0,:, :320].cpu().numpy(), cv2.COLOR_RGB2BGR)
        # # current_frame = cv2.cvtColor(env.render_rgb_array()[0].cpu().numpy(), cv2.COLOR_RGB2BGR)
        # # current_frame = obs['sensor_data']['base_camera']['rgb'][0].cpu().numpy()
        # # current_frame = obs['sensor_data']['base_camera']['Color'][0][:,:,:3].cpu().numpy()
        # # recorded_frame = zarr_store.data['observation.rgb'][min(episode_start_idx + i + 1, episode_end_idx - 1)]
        # recorded_frame = zarr_store.data['observation.rgb'][min(episode_start_idx + i + 1, episode_end_idx - 1)]
        # # recorded_frame = zarr_store.data.gt_segmentation['observation.EE_obj_mask'][min(episode_start_idx + i + 1, episode_end_idx - 1)]
        # # recorded_frame = (recorded_frame / max_segmentation_id * 255).astype(np.uint8)
        # current_frame = (current_frame*0.5 + recorded_frame*0.5).astype(np.uint8)
        # # current_frame = (current_frame*recorded_frame).astype(np.uint8)
        # # current_frame = (current_frame*0.5 + obs['extra']['extrinsic_contact_map'][0].cpu().numpy()*255*0.5).astype(np.uint8)
        # # current_frame = cv2.resize(current_frame, desired_viewing_size, interpolation=cv2.INTER_NEAREST)

        # # assert (recorded_frame == current_frame).all(), "mismatch in recorded frame"
        # # print(np.ptp(recorded_frame - current_frame))

        # cv2.imshow("frame", current_frame)
        # # plt.imshow(current_frame)
        # key = cv2.waitKey(1) & 0xFF
        # if key == ord('q') or key == ord('c') or key == ord('r'):
        #     break

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

        # make a book using the sizes and poses from the env

        # EE_object_mesh = env.non_merged_grasped_books_list[0].get_collision_meshes()[0]
        # EE_object_transform = env.non_merged_grasped_books_list[0].pose

        # don't save if its the last step
        if i < num_steps - 1:
            length, width, height = env.grasped_book_sizes[0].tolist()
            binding_thickness = env.binding_thickness
            cover_thickness = env.cover_thickness
            cover_overhang = env.cover_overhang
            EE_object_transform = convert_sapien_pose_to_transform_matrix(env.non_merged_grasped_books_list[0].pose)

            EE_object_mesh_list = get_book_primitive_mesh_list(length, width, height, binding_thickness, cover_thickness, cover_overhang, global_transform=EE_object_transform)
            EE_object_mesh = tm.util.concatenate(EE_object_mesh_list)
            #%%
            env_book_sizes_list = [env.env_book_sizes[0,i].tolist() for i in range(env.non_merged_env_books_list)]
            env_book_poses_list = [convert_sapien_pose_to_transform_matrix(env_book_over_envs[0].pose) for env_book_over_envs in env.non_merged_env_books_list]
            env_object_meshes_list = get_env_object_meshes_list(env_book_sizes_list, env_book_poses_list, binding_thickness, cover_thickness, cover_overhang)
            # table_mesh = env.table_scene.table.get_collision_meshes()

            table_length, table_width, table_height = env.table_scene.table_length, env.table_scene.table_width, env.table_scene.table_height
            table_pose = convert_sapien_pose_to_transform_matrix(env.table_scene.table.pose)
            table_mesh = get_table_primitive_mesh_list(table_length, table_width, table_height, global_transform=table_pose)
            #%%
            env_mesh = tm.util.concatenate(env_object_meshes_list + table_mesh)
            env_mesh_list = env_object_meshes_list + table_mesh
            #%%

            ray_origins, ray_directions, pixels_uv = generate_rays_from_camera(tm_camera)
            env_hit_min_locations, env_hit_min_pixels_uv, env_hit_min_distances, env_hit_min_index_tri, env_hit_min_ray_directions = get_min_grasped_obj_sdf_at_env_hits_data(ray_origins, ray_directions, pixels_uv, env_mesh, EE_object_mesh_list)
            EE_obj_sdf_on_env_image, EE_obj_sdf_on_env_mask = generate_min_distances_image(env_hit_min_pixels_uv, env_hit_min_distances, tm_camera.resolution[::-1])
            EE_obj_sdf_on_env_image = EE_obj_sdf_on_env_image.astype(np.float32)[:240, :320, np.newaxis]
            EE_obj_sdf_on_env_mask = EE_obj_sdf_on_env_mask.astype(bool)[:240, :320]
            assert EE_obj_sdf_on_env_image.shape == image_shape + (1,)
            # plt.imshow(obs['sensor_data']['base_camera']['rgb'][0].cpu().numpy(), alpha=0.5)
            # plt.imshow(EE_obj_sdf_on_env_image)
            # plt.imshow(scene_img, alpha=0.3)
            #%%
            min_env_surface_normals = env_mesh.face_normals[env_hit_min_index_tri]
            env_xyz_normals_image, env_xyz_normals_image_mask = normals_to_xyz_map(min_env_surface_normals, tm_camera.resolution[::-1], env_hit_min_pixels_uv)#, fill_value=1.0/np.sqrt(3.0))
            env_xyz_normals_image = env_xyz_normals_image.astype(np.float32)[:240, :320]
            env_xyz_normals_image_mask = env_xyz_normals_image_mask.astype(bool)[:240, :320]
            # plt.imshow(env_xyz_normals_image, alpha=0.5)
            # plt.imshow(obs['sensor_data']['base_camera']['rgb'][0].cpu().numpy(), alpha=0.5)
            assert env_xyz_normals_image.shape == image_shape + (3,)
            #%%
            EE_obj_hit_min_locations, EE_obj_hit_min_pixels_uv, EE_obj_hit_min_distances, EE_obj_hit_min_index_tri, EE_obj_hit_min_ray_directions = get_min_env_sdf_at_grasped_obj_hits_data(ray_origins, ray_directions, pixels_uv, env_mesh_list, EE_object_mesh)
            env_sdf_on_EE_obj_image, env_sdf_on_EE_obj_mask = generate_min_distances_image(EE_obj_hit_min_pixels_uv, EE_obj_hit_min_distances, tm_camera.resolution[::-1])
            env_sdf_on_EE_obj_image = env_sdf_on_EE_obj_image.astype(np.float32)[:240, :320, np.newaxis]
            env_sdf_on_EE_obj_mask = env_sdf_on_EE_obj_mask.astype(bool)[:240, :320]
            # plt.imshow(obs['sensor_data']['base_camera']['rgb'][0].cpu().numpy(), alpha=0.5)
            # plt.imshow(env_sdf_on_EE_obj_image)
            assert env_sdf_on_EE_obj_image.shape == image_shape + (1,)
            #%%
            min_EE_object_surface_normals = EE_object_mesh.face_normals[EE_obj_hit_min_index_tri] # these are normalized already
            EE_object_xyz_normals_image, EE_object_xyz_normals_image_mask = normals_to_xyz_map(min_EE_object_surface_normals, tm_camera.resolution[::-1], EE_obj_hit_min_pixels_uv)#, fill_value=1.0/np.sqrt(3.0))
            EE_object_xyz_normals_image = EE_object_xyz_normals_image.astype(np.float32)[:240, :320]
            EE_object_xyz_normals_image_mask = EE_object_xyz_normals_image_mask.astype(bool)[:240, :320]

            # plt.imshow(EE_object_xyz_normals_image, alpha=0.5)
            # plt.imshow(obs['sensor_data']['base_camera']['rgb'][0].cpu().numpy(), alpha=0.5)
            assert EE_object_xyz_normals_image.shape == image_shape + (3,)
            #%%
            if record_contact_features:
                zarr_gt_contact['observation.env_dtc_map'].append(EE_obj_sdf_on_env_image[np.newaxis, ...])
                zarr_gt_contact['observation.env_normals_map'].append(env_xyz_normals_image[np.newaxis, ...])
                zarr_gt_contact['observation.EE_dtc_map'].append(env_sdf_on_EE_obj_image[np.newaxis, ...])
                zarr_gt_contact['observation.EE_normals_map'].append(EE_object_xyz_normals_image[np.newaxis, ...])

        # frames.append(current_frame)
        elapsed_timesteps = info["elapsed_steps"].item()
        elapsed_simtime = elapsed_timesteps * sim_dt_bw_step
        elapsed_realtime = time.perf_counter() - start_time
        # time_to_sleep = sim_dt_bw_step - elapsed_time
        time_to_sleep = elapsed_simtime - elapsed_realtime
        # if time_to_sleep > 0:
        #     time.sleep(time_to_sleep)
        if elapsed_timesteps % 50 == 0:
            print(f"realtime_factor: {elapsed_simtime/elapsed_realtime} | elapsed steps: {elapsed_timesteps} | elapsed rt {elapsed_realtime} | elapsed simt {elapsed_simtime}")
    assert len(zarr_gt_contact['observation.env_dtc_map'][:]) == episode_end_idx
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