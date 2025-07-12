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
desired_viewing_size = (256, 256)
# path_to_demo_root_dir = Path("/mnt/crucialSSD/datasetsSSD/fish_datasets/simulated/teleop/424_sim_w_recovery_demos_leftof4thbook_springbookends_graspedrand_noenvrand_slotrand_20hz_act")
path_to_demo_root_dir = Path("/mnt/crucialSSD/datasetsSSD/fish_datasets/simulated/teleop/1_demo_test")

path_to_zarr = path_to_demo_root_dir / "demos.zarr"
path_to_json = path_to_demo_root_dir / "demos.json"

snap_to_env_state = True
record_contact_features = True
#%%
zarr_store = zarr.open(str(path_to_zarr), mode='r+')
with open(path_to_json, 'r') as f:
    json_data = json.load(f)
#%%
compressors = zarr_store['data']['observation.rgb'].compressors[0]
# create the datasets for contact features if they don't exist
image_shape = zarr_store['data']['observation.rgb'].shape[1:3]
zarr_gt_contact = zarr_store['data']['gt_contact']
if record_contact_features:
    # if 'observation.EE_dtc_map' not in zarr_gt_contact:
    zarr_gt_contact.create_array('observation.EE_dtc_map', shape=(0,) + image_shape + (1,), chunks=(1,) + image_shape + (1,), dtype=np.float32, compressor=compressors, overwrite=True)
    # if 'observation.EE_normals_map' not in zarr_gt_contact:
    zarr_gt_contact.create_array('observation.EE_normals_map', shape=(0,) + image_shape + (3,), chunks=(1,) + image_shape + (3,), dtype=np.float32, compressor=compressors, overwrite=True)
    # if 'observation.env_dtc_map' not in zarr_gt_contact:
    zarr_gt_contact.create_array('observation.env_dtc_map', shape=(0,) + image_shape + (1,), chunks=(1,) + image_shape + (1,), dtype=np.float32, compressor=compressors, overwrite=True)
    # if 'observation.env_normals_map' not in zarr_gt_contact:
    zarr_gt_contact.create_array('observation.env_normals_map', shape=(0,) + image_shape + (3,), chunks=(1,) + image_shape + (3,), dtype=np.float32, compressor=compressors, overwrite=True)
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
        y_randomization_bounds=[-0.05, 0.05],
        # y_randomization_bounds=0.0,
    ),
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
        contact_features_dict = env.get_extra_contact_features(True, True)
        zarr_gt_contact['observation.env_dtc_map'].append(contact_features_dict['env_dtc_map'].cpu().numpy())
        zarr_gt_contact['observation.env_normals_map'].append(contact_features_dict['env_normals_map'].cpu().numpy())
        zarr_gt_contact['observation.EE_dtc_map'].append(contact_features_dict['EE_dtc_map'].cpu().numpy())
        zarr_gt_contact['observation.EE_normals_map'].append(contact_features_dict['EE_normals_map'].cpu().numpy())

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
                contact_features_dict = env.get_extra_contact_features(True, True)
                zarr_gt_contact['observation.env_dtc_map'].append(contact_features_dict['env_dtc_map'].cpu().numpy())
                zarr_gt_contact['observation.env_normals_map'].append(contact_features_dict['env_normals_map'].cpu().numpy())
                zarr_gt_contact['observation.EE_dtc_map'].append(contact_features_dict['EE_dtc_map'].cpu().numpy())
                zarr_gt_contact['observation.EE_normals_map'].append(contact_features_dict['EE_normals_map'].cpu().numpy())


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
    assert zarr_gt_contact['observation.env_dtc_map'].shape[0] == episode_end_idx, f"mismatch in number of contact features: {zarr_gt_contact['observation.env_dtc_map'].shape[0]} vs {episode_end_idx}"
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