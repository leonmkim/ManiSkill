#%%
import gymnasium as gym
import time
import tqdm
import numpy as np

import sys
import torch
import matplotlib.pyplot as plt
import natsort
import os

import zarr
ZARR_VERSION=int(zarr.__version__.split('.')[0])

from pathlib import Path

import time
import json

from pathlib import Path

import tqdm

from pytorch3d.transforms import quaternion_to_matrix
from torch.utils.data import DataLoader

from tqdm import tqdm
import einops
import imageio

import click
#%%
# path to fish_leon dir 
path_to_fish_leon = Path(__file__).parents[2]
path_to_contact_estimation = path_to_fish_leon / "contact_estimation"
assert path_to_contact_estimation.exists(), f"Path {path_to_contact_estimation} does not exist. Please check the path."
sys.path.append(str(path_to_contact_estimation))
from src.dataset_processing.mask_predictor import MaskPredictor
from src.utils import cv2_debugger_header

from src.dataset.contact_dataset_episodic import ContactDatasetEpisodic
from src.utils.viz_utils import (
    masked_overlay_im_list,
)
from src.dataset_processing.extract_trimesh_data import save_bool_mask_as_uint8
from PIL import Image
#%%
path_to_FISH = path_to_fish_leon / "FISH"
assert path_to_FISH.exists(), f"Path {path_to_FISH} does not exist. Please check the path."
sys.path.append(str(path_to_FISH))
from agent.encoder import VisualFeatureSet
from lerobot.common.policies.diffusion.configuration_diffusion import ActionConfig, ActionHistoryConfig
from agent.encoder import MaskInputDict
from dataset.expert_dataset import ExpertDatasetZarr
#%%
# path_to_demo_root_dir = Path("/mnt/crucialSSD/datasetsSSD/fish_datasets/simulated/teleop/FISH/expert_demos/frankagym/FrankaInsertion-v1")
# demo_name = "1232_sim_w_recovery_demos_leftof4thbook_springbookends_graspedrand_noenvrand_noslotrand_20hz_act"
# path_to_demo_dir = path_to_demo_root_dir / demo_name
# assert path_to_demo_dir.exists(), f"Path {path_to_demo_dir} does not exist. Please check the path."

# path_to_zarr = path_to_demo_dir / "demos.zarr"
# path_to_json = path_to_demo_dir / "demos.json"

# assert path_to_zarr.exists(), f"Path {path_to_zarr} does not exist. Please check the path."
# assert path_to_json.exists(), f"Path {path_to_json} does not exist. Please check the path."
# zarr_dataset = zarr.open(str(path_to_zarr), mode='r+')
#%%
@click.command()
@click.argument('episode-idx', type=int)
@click.argument('path-to-demo-root-dir', type=click.Path(exists=True, path_type=Path))
@click.argument('demo-name', type=str)
def main(episode_idx, path_to_demo_root_dir, demo_name):
    print(f"starting to process episode {episode_idx}...")
    device = 'cuda'
    assert path_to_demo_root_dir.exists(), f"Path {path_to_demo_root_dir} does not exist. Please check the path."

    # path_to_demo_root_dir = Path("/mnt/crucialSSD/datasetsSSD/fish_datasets/simulated/teleop/FISH/expert_demos/frankagym/FrankaInsertion-v1")
    # demo_name = "1232_sim_w_recovery_demos_leftof4thbook_springbookends_graspedrand_noenvrand_noslotrand_20hz_act"
    path_to_demo_dir = path_to_demo_root_dir / demo_name
    assert path_to_demo_dir.exists(), f"Path {path_to_demo_dir} does not exist. Please check the path."

    path_to_zarr = path_to_demo_dir / "demos.zarr"
    path_to_json = path_to_demo_dir / "demos.json"

    assert path_to_zarr.exists(), f"Path {path_to_zarr} does not exist. Please check the path."
    assert path_to_json.exists(), f"Path {path_to_json} does not exist. Please check the path."
    zarr_dataset = zarr.open(str(path_to_zarr), mode='r+')
    #%%
    if 'episode_data' not in zarr_dataset:
        zarr_dataset.create_group('episode_data')
        print("Created group 'episode_data' in zarr dataset.")
    #%%
    sam2_options_string = "sam2-hiera-base-plus"
    # sam2_options_string = "sam2.1-hiera-base-plus"
    hf_pretrained_model_name = f"facebook/{sam2_options_string}"

    '''
    from book_insertion env config
    density_randomization_bounds: [650, 850]
    height_randomization_bounds: [0.165, 0.25]
    width_randomization_bounds: [0.03, 0.065]
    length_randomization_bounds: [0.1, 0.15]
    '''

    max_grasped_object_length_x = 0.15 + .01
    max_grasped_object_width_y = 0.065 + .01
    max_grasped_object_height_z = 0.25 + .01
    mask_predictor = MaskPredictor(hf_pretrained_model_name=hf_pretrained_model_name, device=device, 
                                max_grasped_object_length_x=max_grasped_object_length_x,
                                max_grasped_object_width_y=max_grasped_object_width_y,
                                max_grasped_object_height_z=max_grasped_object_height_z,)
    #%%
    if sam2_options_string not in zarr_dataset['data']:
        zarr_dataset.create_group('data/' + sam2_options_string)
        print(f"Created group {sam2_options_string} in zarr dataset.")
    #%%
    # for episode_idx in episode_idxs:
    episode_data_group_name = f"episode_data/episode_{episode_idx}"
    if episode_data_group_name not in zarr_dataset:
        zarr_dataset.create_group(episode_data_group_name)
        print(f"Created group {episode_data_group_name} in zarr dataset.")
    episode_data_mask_group_name = episode_data_group_name + '/' + sam2_options_string
    if sam2_options_string not in zarr_dataset[episode_data_group_name]:
        zarr_dataset.create_group(episode_data_mask_group_name)
        print(f"Created group {sam2_options_string} in episode data group {episode_data_group_name} in zarr dataset.")

    episode_data_mask_array_name = episode_data_mask_group_name + '/observation.EE_obj_mask'

    #%%
    mask_input_dict = MaskInputDict(enable=True, mask_list=['EE_obj_mask'], representation='channels', segmentation_model_name='gt_segmentation')
    observation_cfg = VisualFeatureSet(use_color=True, use_depth=True, mask_input_dict=mask_input_dict, use_contact_map=False, use_sdf_maps=False, use_normals_maps=False)
    action_horizon_length = 1
    action_history_length = 1
    action_config = ActionConfig(horizon_length=action_horizon_length, action_frame_expression='delta', input_rotation_representation='euler_angles')
    action_history_config = ActionHistoryConfig(enable=False, history_length=action_history_length, action_frame_expression='delta', action_frame='current_end_effector', rotation_representation='euler_angles')
    episode_dataset = ExpertDatasetZarr(path_to_zarr, 
                                        demos_idxs_list_or_num=[episode_idx], 
                                        observation_cfg=observation_cfg, 
                                        action_config=action_config, 
                                        action_history_config=action_history_config, 
                                        action_key='action',
                                        n_obs_steps=1,
                                        load_to_memory=False,
                                        pad_after=0,
                                        action_indices_same_as_indices=False,
                                        set_close_gripper_action_for_padding=True,
                                        include_target_pose_observations=True,
                                        repeat_padding_for_actions=True,
                                        action_using_env_state_indices=False,
                                        stored_action_frame_expression='absolute',
                                        )
    dataloader = DataLoader(episode_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    #%%
    gt_mask_shape = zarr_dataset['data']['gt_segmentation']['observation.EE_obj_mask'].shape[1:]
    gt_mask_dtype = zarr_dataset['data']['gt_segmentation']['observation.EE_obj_mask'].dtype
    gt_mask_compressors = zarr_dataset['data']['gt_segmentation']['observation.EE_obj_mask'].compressors[0]

    #%%
    ## ##############
    ## generate masks and write to disk
    ## ##############
    mask_predictor.reset()
    camera_K = zarr_dataset['meta']['episode_cam_K'][episode_idx]
    cam_tf_world = zarr_dataset['meta']['episode_cam_tf_world'][episode_idx]
    zarr_dataset.create_array(episode_data_mask_array_name, shape=(0, *gt_mask_shape), chunks=(1, *gt_mask_shape), dtype=gt_mask_dtype, compressors=gt_mask_compressors, overwrite=True)
    #%%
    for i, batch in tqdm(enumerate(dataloader)):
    # batch = next(iter(dataloader))
        # i = 0
        color_image = einops.rearrange(batch['observation.rgb'][0,0], 'c h w -> h w c').cpu().numpy()
        depth_image = einops.rearrange(batch['observation.depth'][0,0], 'c h w -> h w c').cpu().numpy()
        EE_pose = batch['observation.state'][0,0,:7].cpu().numpy()
        gripper_width = batch['observation.state'][0,0,7].cpu().numpy()

        if i == 0: #initialize cutie
            # mask = batch['observation.EE_obj_mask'][0,0,0].cpu().numpy()
            # mask = mask_predictor.start_mask_tracking_from_mask(color_image, mask)
            mask = mask_predictor.start_mask_tracking(color_image, depth_image, camera_K, cam_tf_world, EE_pose, gripper_width)
        else:
            # with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
            #     out_obj_ids, out_mask_logits = sam2_online_predictor.track(color_image)
            mask = mask_predictor.track_mask(color_image)
        # if episode_data_mask_array_name not in zarr_dataset:

        zarr_dataset[episode_data_mask_array_name].append(mask[np.newaxis, :, :, np.newaxis].astype(gt_mask_dtype))

if __name__ == "__main__":
    main()
    print(f"Saved masks to {path_to_zarr}.")
    print("Done.")