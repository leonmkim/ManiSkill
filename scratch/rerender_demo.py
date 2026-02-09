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

# path_to_demo_root_dir = Path("/mnt/bighdd/fish_contact_backup/expert_demos/FISH/expert_demos/frankagym/FrankaInsertion-v1")
# demo_name = "1232_sim_w_recovery_demos_leftof4thbook_springbookends_graspedrand_noenvrand_noslotrand_20hz_act"
# task_gym_name = "BookInsertion-v0"
# episode_idx = 0
# cam_extrinsic_rotation_angle_deg = 75

# cam_extrinsic_rotation_angle = np.deg2rad(cam_extrinsic_rotation_angle_deg)

# grasped_object_name = 'grasped_book'
# path_to_demo_dir = path_to_demo_root_dir / demo_name

# path_to_zarr = path_to_demo_dir / "demos.zarr"
# path_to_json = path_to_demo_dir / "demos.json"

# snap_to_env_state = True

# max_num_contact = 50
# #%%
# zarr_store = zarr.open(str(path_to_zarr), mode='r+')
# with open(path_to_json, 'r') as f:
#     json_data = json.load(f)
# #%%
# rerendered_rgb_frames = list()
# for i in tqdm.tqdm(range(zarr_store['episode_data']['episode_0']['rerendered_-50.0_deg_rotation']['observation.rgb'].shape[0])):
#     rgb_image = zarr_store['episode_data']['episode_0']['rerendered_-50.0_deg_rotation']['observation.rgb'][i]
#     rerendered_rgb_frames.append(rgb_image)


# video_path = Path("./rerendered_videos") / demo_name
# video_path.mkdir(parents=True, exist_ok=True)
# video_name = f"episode_{episode_idx}_camera_rotation_{cam_extrinsic_rotation_angle_deg}_degrees.mp4"
# images_to_video(
#     # force_map_overlaid,
#     rerendered_rgb_frames,
#     output_dir=video_path,
#     # video_name="force_map_overlaid",
#     video_name=video_name,
#     fps=20,
#     quality=10,
# )


#%%
@click.command()
@click.argument('episode-idx', type=int)
@click.argument('path-to-demo-root-dir', type=click.Path(exists=True, path_type=Path))
@click.argument('demo-name', type=str)
@click.argument('task-gym-name', type=str)
@click.argument('cam-extrinsic-rotation-angle-deg', type=float)
def main(episode_idx, path_to_demo_root_dir, demo_name, task_gym_name, cam_extrinsic_rotation_angle_deg):
    assert task_gym_name in [
        "BookInsertion-v0",
        "PegInsertionSideCustom-v1",
    ], "Invalid task gym name"
    # path_to_demo_root_dir = Path("/mnt/bighdd/fish_contact_backup/expert_demos/FISH/expert_demos/frankagym/FrankaInsertion-v1")
    # demo_name = "1232_sim_w_recovery_demos_leftof4thbook_springbookends_graspedrand_noenvrand_noslotrand_20hz_act"
    # task_gym_name = "BookInsertion-v0"
    # episode_idx = 0
    # cam_extrinsic_rotation_angle_deg = 75

    # path_to_demo_root_dir = Path(path_to_demo_root_dir)

    cam_extrinsic_rotation_angle = np.deg2rad(cam_extrinsic_rotation_angle_deg)

    grasped_object_name = 'grasped_book'
    path_to_demo_dir = path_to_demo_root_dir / demo_name

    path_to_zarr = path_to_demo_dir / "demos.zarr"
    path_to_json = path_to_demo_dir / "demos.json"

    snap_to_env_state = True

    max_num_contact = 50
    #%%
    zarr_store = zarr.open(str(path_to_zarr), mode='r+')
    with open(path_to_json, 'r') as f:
        json_data = json.load(f)
    #%%
    keys_to_check = list()
    zarr_data = zarr_store['data']
    if 'episode_data' not in zarr_store:
        zarr_store.create_group('episode_data')
    if f'episode_{episode_idx}' not in zarr_store['episode_data']:
        zarr_store['episode_data'].create_group(f'episode_{episode_idx}')
    zarr_episode_data = zarr_store['episode_data'][f'episode_{episode_idx}']

    # things to rerender: 
    # observation.rgb, 
    # observation.depth, 
    # observation.EE_pixel_coord, 
    # gt_segmentation (observation.EE_obj_mask, observation.segmentation), 
    rerender_group_name = f"rerendered_{cam_extrinsic_rotation_angle_deg}_deg_rotation"
    if rerender_group_name not in zarr_episode_data:
        zarr_episode_data.create_group(rerender_group_name)
    zarr_episode_rerendered_data = zarr_episode_data[rerender_group_name]
    keys_to_check.append('observation.rgb')
    original_rgb_type = zarr_store['data']['observation.rgb'].dtype
    original_rgb_compressors = zarr_store['data']['observation.rgb'].compressors[0]
    original_rgb_shape = zarr_store['data']['observation.rgb'].shape[1:]
    zarr_episode_rerendered_data.create_array(
        'observation.rgb',
        shape=(0,) + original_rgb_shape,
        dtype=original_rgb_type,
        compressor=original_rgb_compressors,
        chunks=(1,) + original_rgb_shape,
        overwrite=True,
    )
    keys_to_check.append('observation.depth')
    original_depth_shape = zarr_store['data']['observation.depth'].shape[1:]
    original_depth_type = zarr_store['data']['observation.depth'].dtype
    original_depth_compressors = zarr_store['data']['observation.depth'].compressors[0]
    zarr_episode_rerendered_data.create_array(
        'observation.depth',
        shape=(0,) + original_depth_shape,
        dtype=original_depth_type,
        compressor=original_depth_compressors,
        chunks=(1,) + original_depth_shape,
        overwrite=True,
    )
    keys_to_check.append('observation.EE_pixel_coord')
    original_EE_pixel_coord_type = zarr_store['data']['observation.EE_pixel_coord'].dtype
    original_EE_pixel_coord_compressors = zarr_store['data']['observation.EE_pixel_coord'].compressors[0]
    original_EE_pixel_coord_shape = zarr_store['data']['observation.EE_pixel_coord'].shape[1:]
    zarr_episode_rerendered_data.create_array(
        'observation.EE_pixel_coord',
        shape=(0,) + original_EE_pixel_coord_shape,
        dtype=original_EE_pixel_coord_type,
        compressor=original_EE_pixel_coord_compressors,
        chunks=(1,) + original_EE_pixel_coord_shape,
        overwrite=True,
    )

    if 'gt_segmentation' not in zarr_episode_rerendered_data:
        zarr_episode_rerendered_data.create_group('gt_segmentation')
    keys_to_check.append('gt_segmentation/observation.EE_obj_mask')
    original_EE_obj_mask_type = zarr_data['gt_segmentation']['observation.EE_obj_mask'].dtype
    original_EE_obj_mask_compressors = zarr_data['gt_segmentation']['observation.EE_obj_mask'].compressors[0]
    original_EE_obj_mask_shape = zarr_data['gt_segmentation']['observation.EE_obj_mask'].shape[1:]
    zarr_episode_rerendered_data['gt_segmentation'].create_array(
        'observation.EE_obj_mask',
        shape=(0,) + original_EE_obj_mask_shape,
        dtype=original_EE_obj_mask_type,
        compressor=original_EE_obj_mask_compressors,
        chunks=(1,) + original_EE_obj_mask_shape,
        overwrite=True,
    )
    keys_to_check.append('gt_segmentation/observation.segmentation')
    original_segmentation_type = zarr_data['gt_segmentation']['observation.segmentation'].dtype
    original_segmentation_compressors = zarr_data['gt_segmentation']['observation.segmentation'].compressors[0]
    original_segmentation_shape = zarr_data['gt_segmentation']['observation.segmentation'].shape[1:]
    zarr_episode_rerendered_data['gt_segmentation'].create_array(
        'observation.segmentation',
        shape=(0,) + original_segmentation_shape,
        dtype=original_segmentation_type,
        compressor=original_segmentation_compressors,
        chunks=(1,) + original_segmentation_shape,
        overwrite=True,
    )

    original_episode_cam_tf_world_type = zarr_store['meta']['episode_cam_tf_world'].dtype
    original_episode_cam_tf_world_compressors = zarr_store['meta']['episode_cam_tf_world'].compressors[0]
    original_episode_cam_tf_world_shape = zarr_store['meta']['episode_cam_tf_world'].shape[1:]
    zarr_episode_rerendered_data.create_array(
        'episode_cam_tf_world',
        shape=(0,) + original_episode_cam_tf_world_shape,
        dtype=original_episode_cam_tf_world_type,
        compressor=original_episode_cam_tf_world_compressors,
        chunks=(1,) + original_episode_cam_tf_world_shape,
        overwrite=True,
    )

    #%%
    joint_stiffness = 100.0
    joint_damping = 2*np.sqrt(joint_stiffness)
    ## testing book insertion task
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
        cam_extrinsic_rotation_angle=cam_extrinsic_rotation_angle,
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
        # obs_mode="rgb",
        obs_mode="rgb+depth+segmentation",
        # obs_mode="none",
        control_mode="pd_ee_target_delta_pose",
        # control_mode='pd_ee_target_pose',
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
    segmentation_id_map = dict()
    for key, value in env.segmentation_id_map.items():
        entity_name = value.name
        segmentation_id_map[entity_name] = key
    #%%
    sim_dt = 1.0 / env.sim_config.sim_freq
    sim_dt_bw_step = sim_dt * (env.sim_config.sim_freq / env.sim_config.control_freq)

    human_render_cam_params = env.scene.human_render_cameras['render_camera'].get_params()
    human_render_cam_intrisic = human_render_cam_params['intrinsic_cv'][0]
    human_render_cam_cam2world_gl = human_render_cam_params['cam2world_gl'][0][:3, :4]
    human_render_cam_extrinsic_cv = human_render_cam_params['extrinsic_cv'][0]

    #%%

    episode_dict = json_data['episodes'][episode_idx]
    assert episode_dict['episode_id'] == episode_idx
    seed = episode_dict['episode_seed']
    num_steps = episode_dict['elapsed_steps']
    episode_start_idx = 0 if episode_idx == 0 else zarr_store['meta']['episode_ends'][episode_idx - 1]
    env_state_episode_start_idx = episode_start_idx + episode_idx
    episode_end_idx = zarr_store['meta']['episode_ends'][episode_idx]
    env_state_episode_end_idx = episode_end_idx + episode_idx + 1
    assert episode_end_idx - episode_start_idx == num_steps, f"mismatch in episode steps: {episode_end_idx - episode_start_idx} vs {num_steps}"
    #%%
    rerendered_frames = list()
    obs, info = env.reset(seed=seed)

    cam_tf_world = obs['sensor_param']['base_camera']['extrinsic_cv'] # num_envs x 3 x 4 
    # need to add the last row for the homogeneous coordinates
    cam_tf_world = np.concatenate([cam_tf_world, np.zeros((1, 1, 4), dtype=np.float32)], axis=1)
    cam_tf_world[:, 3, 3] = 1.0
    zarr_episode_rerendered_data['episode_cam_tf_world'].append(cam_tf_world)

    if snap_to_env_state:
        current_env_state_dict = construct_env_state_dict(zarr_store['data'], env_state_episode_start_idx)
        env.set_state_dict(current_env_state_dict)

    zarr_episode_rerendered_data['observation.rgb'].append(obs['sensor_data']['base_camera']['rgb'].cpu().numpy())
    zarr_episode_rerendered_data['observation.depth'].append(obs['sensor_data']['base_camera']['depth'].cpu().numpy())
    zarr_episode_rerendered_data['observation.EE_pixel_coord'].append(obs['extra']['end_effector_pixel_coordinates'].cpu().numpy())

    segmentation = obs['sensor_data']['base_camera']['segmentation'].cpu().numpy()
    EE_obj_mask = (segmentation == segmentation_id_map[f"{grasped_object_name}_0"]).astype(np.uint8)
    zarr_episode_rerendered_data['gt_segmentation']['observation.EE_obj_mask'].append(EE_obj_mask)
    zarr_episode_rerendered_data['gt_segmentation']['observation.segmentation'].append(segmentation)
    # rgb_image = obs['sensor_data']['base_camera']['rgb'][0].cpu().numpy()
    # segmentation_mask = obs['sensor_data']['base_camera']['segmentation'][0].cpu().numpy()
    # rerendered_frames.append(rgb_image)
    #%%
    # show the segmentation mask as an image to verify that the camera is looking at the scene correctly
    # plt.imshow(segmentation_mask, alpha=0.8)
    # plt.imshow(rgb_image, alpha=0.2)
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
            # rgb_image = obs['sensor_data']['base_camera']['rgb'][0].cpu().numpy()
            # rerendered_frames.append(rgb_image)
            zarr_episode_rerendered_data['observation.rgb'].append(obs['sensor_data']['base_camera']['rgb'].cpu().numpy())
            zarr_episode_rerendered_data['observation.depth'].append(obs['sensor_data']['base_camera']['depth'].cpu().numpy())
            zarr_episode_rerendered_data['observation.EE_pixel_coord'].append(obs['extra']['end_effector_pixel_coordinates'].cpu().numpy())

            segmentation = obs['sensor_data']['base_camera']['segmentation'].cpu().numpy()
            EE_obj_mask = (segmentation == segmentation_id_map[f"{grasped_object_name}_0"]).astype(np.uint8)
            zarr_episode_rerendered_data['gt_segmentation']['observation.EE_obj_mask'].append(EE_obj_mask)
            zarr_episode_rerendered_data['gt_segmentation']['observation.segmentation'].append(segmentation)
    #%%
    # video_path = Path("./rerendered_videos") / demo_name
    # video_path.mkdir(parents=True, exist_ok=True)
    # video_name = f"episode_{episode_idx}_camera_rotation_{np.rad2deg(env.cam_extrinsic_rotation_angle):.1f}_degrees.mp4"
    # images_to_video(
    #     # force_map_overlaid,
    #     rerendered_frames,
    #     output_dir=video_path,
    #     # video_name="force_map_overlaid",
    #     video_name=video_name,
    #     fps=20,
    #     quality=10,
    # )
    #%%
    for key in keys_to_check:
        assert key in zarr_episode_rerendered_data, f"key {key} not found in zarr_episode_rerendered_data"
        assert zarr_episode_rerendered_data[key].shape[0] == num_steps, f"mismatch in number of contact features for {key}: {zarr_episode_rerendered_data[key].shape[0]} vs {num_steps}"

    env.close()
    del env

if __name__ == "__main__":
    main()
    print(f"Saved rerendered obs to {path_to_zarr}.")