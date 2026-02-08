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
import click
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
from lerobot.common.policies.diffusion.configuration_diffusion import ActionConfig, ActionHistoryConfig, EndEffectorWrenchHistoryConfig
from agent.encoder import MaskInputDict, TheiaInputConfig
from dataset.expert_dataset import ExpertDatasetZarr
#%%
@click.command()
@click.option('--path_to_demo_root_dir', type=str, default="/mnt/bighdd/fish_contact_backup/expert_demos/FISH/expert_demos/frankagym/FrankaInsertion-v1", help='Path to the root directory containing the demo directories.')
@click.option('--demo_name', type=str, default="1232_sim_w_recovery_demos_leftof4thbook_springbookends_graspedrand_noenvrand_noslotrand_20hz_act", help='Name of the demo directory to process.')
@click.option('--segmentation_model_name', type=str, default='sam2-hiera-base-plus', help='Name of the segmentation model used for generating masks. This should match the name used in the observation configuration when the datasetwas created.')
@click.option('--contact_model_name', type=str, default='contact_model_175604_2_epoch_9', help='Name of the contact model used for generating contact maps and related features. This should match the name used in the observation configuration when the dataset was created.')
def main(path_to_demo_root_dir: str, demo_name: str, segmentation_model_name: str, contact_model_name: str):
    # path_to_demo_root_dir = Path("/mnt/bighdd/fish_contact_backup/expert_demos/FISH/expert_demos/frankagym/FrankaInsertion-v1")
    path_to_demo_root_dir = Path(path_to_demo_root_dir)
    # demo_name = "1232_sim_w_recovery_demos_leftof4thbook_springbookends_graspedrand_noenvrand_noslotrand_20hz_act"
    zarr_dataset = zarr.open(str(path_to_demo_root_dir / demo_name / "demos.zarr"), mode='r')
    num_episodes = zarr_dataset['meta']['episode_ends'].shape[0]

    #%%
    assert path_to_demo_root_dir.exists(), f"Path {path_to_demo_root_dir} does not exist. Please check the path."

    path_to_demo_dir = path_to_demo_root_dir / demo_name
    assert path_to_demo_dir.exists(), f"Path {path_to_demo_dir} does not exist. Please check the path."

    path_to_zarr = path_to_demo_dir / "demos.zarr"
    path_to_json = path_to_demo_dir / "demos.json"

    assert path_to_zarr.exists(), f"Path {path_to_zarr} does not exist. Please check the path."
    assert path_to_json.exists(), f"Path {path_to_json} does not exist. Please check the path."
    #%%
    mask_input_dict = MaskInputDict(enable=True, mask_list=['EE_obj_mask'], representation='channels', segmentation_model_name=segmentation_model_name)
    theia_input_config = TheiaInputConfig(enable=True, model_name='theia-base-patch16-224-cdiv')
    observation_cfg = VisualFeatureSet(use_color=True, use_depth=True, 
                                    mask_input_dict=mask_input_dict, 
                                    theia_input_config=theia_input_config,
                                    contact_model_name=contact_model_name,
                                    use_contact_map=True, use_sdf_maps=True, use_normals_maps=True, use_contact_forces_map=True,)
    action_horizon_length = 1
    action_history_length = 1
    action_config = ActionConfig(horizon_length=action_horizon_length, action_frame_expression='delta', input_rotation_representation='euler_angles')
    action_history_config = ActionHistoryConfig(enable=False, history_length=action_history_length, action_frame_expression='delta', action_frame='current_end_effector', rotation_representation='euler_angles')
    end_effector_wrench_history_config = EndEffectorWrenchHistoryConfig(enable=False)

    #%%
    # episode_idx = 0
    # for episode_idx in range(num_episodes):
    for episode_idx in tqdm(range(num_episodes), desc="Processing episodes", position=0, leave=True):
        episode_dataset = ExpertDatasetZarr(path_to_zarr, 
                                            demos_idxs_list_or_num=[episode_idx], 
                                            observation_cfg=observation_cfg, 
                                            action_config=action_config, 
                                            action_history_config=action_history_config, 
                                            end_effector_wrench_history_config=end_effector_wrench_history_config,
                                            action_key='action',
                                            n_obs_steps=1,
                                            load_to_memory=False,
                                            pad_after=0,
                                            action_indices_same_as_indices=False,
                                            set_close_gripper_action_for_padding=True,
                                            include_target_pose_observations=True,
                                            # repeat_padding_for_actions=True,
                                            # action_using_env_state_indices=False,
                                            # stored_action_frame_expression='absolute',
                                            )
        dataloader = DataLoader(episode_dataset, batch_size=8, shuffle=False, num_workers=8, pin_memory=True)

        #%%
        ## ##############
        ## generate masks and write to disk
        ## ##############
        for i, batch in tqdm(enumerate(dataloader), desc="Processing batches", position=1, leave=False):
            pass
            # color_image = einops.rearrange(batch['observation.rgb'][0,0], 'c h w -> h w c').cpu().numpy()
            # depth_image = einops.rearrange(batch['observation.depth'][0,0], 'c h w -> h w c').cpu().numpy()
            # EE_pose = batch['observation.state'][0,0,:7].cpu().numpy()
            # gripper_width = batch['observation.state'][0,0,7].cpu().numpy()

    # %%
if __name__ == "__main__":
    main()
