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
from src.utils import cv2_debugger_header
from src.dataset_processing.contact_predictor import ContactPredictor
#%%
path_to_FISH = path_to_fish_leon / "FISH"
assert path_to_FISH.exists(), f"Path {path_to_FISH} does not exist. Please check the path."
sys.path.append(str(path_to_FISH))
from agent.encoder import VisualFeatureSet
from lerobot.common.policies.diffusion.configuration_diffusion import ActionConfig, ActionHistoryConfig, EndEffectorWrenchHistoryConfig
from agent.encoder import MaskInputDict
from dataset.expert_dataset import ExpertDatasetZarr
# #%%
# path_to_demo_root_dir = Path("/mnt/crucialSSD/datasetsSSD/fish_datasets/simulated/teleop/FISH/expert_demos/frankagym/FrankaInsertion-v1")
# path_to_zarr = path_to_demo_root_dir / "2_demo_test" / "demos.zarr"
# zarr_store = zarr.open(str(path_to_zarr), mode='r')
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

# #%%
# episode_idx = 0
# episode_start_idx = 0 if episode_idx == 0 else zarr_dataset['meta']['episode_ends'][episode_idx - 1]
# episode_end_idx = zarr_dataset['meta']['episode_ends'][episode_idx]
# episode_length = episode_end_idx - episode_start_idx
# import imageio
# import cv2
# gt_contact_images = []
# estimated_contact_images = []
# gt_EE_dtc_images = []
# estimated_EE_dtc_images = []
# gt_env_dtc_images = []
# estimated_env_dtc_images = []
# contact_threshold_viz = 0.01
# dtc_threshold_viz = 0.05
# # contact_model_group_name = f"contact_model_175604_2_epoch_9"
# contact_model_group_name = f"contact_model_197406_2_epoch_8"
# for i, idx in enumerate(range(episode_start_idx, episode_end_idx)):
#     gt_contact_map = zarr_dataset['data']['gt_contact']['observation.contact_map'][idx]
#     estimated_contact_map = zarr_dataset['episode_data']['episode_0'][contact_model_group_name]['observation.contact_map'][i]
#     rgb_image = zarr_dataset['data']['observation.rgb'][idx]


#     gt_contact_map_mask = (gt_contact_map > contact_threshold_viz)
#     estimated_contact_map_mask = (estimated_contact_map > contact_threshold_viz)

#     # gt_contact_map_viz = cv2.applyColorMap(
#     #     np.clip((gt_contact_map_mask * 255).astype(np.uint8), 0, 255),
#     #     cv2.COLORMAP_JET
#     # )
#     gt_contact_indices = np.where(gt_contact_map > contact_threshold_viz)
#     gt_contact_map_viz = rgb_image.copy()
#     gt_contact_map_viz[gt_contact_indices[0], gt_contact_indices[1], :] = np.array([255, 0, 0])  # Red color for GT contact
#     estimated_contact_map_viz = cv2.applyColorMap(
#         np.clip((estimated_contact_map_mask * (255/.2)).astype(np.uint8), 0, 255),
#         cv2.COLORMAP_JET
#     )

#     gt_contact_map_viz = cv2.addWeighted(gt_contact_map_viz, 0.5, rgb_image, 0.5, 0)
#     estimated_contact_map_viz = cv2.addWeighted(estimated_contact_map_viz, 0.5, rgb_image, 0.5, 0)
#     gt_contact_images.append(gt_contact_map_viz)
#     estimated_contact_images.append(estimated_contact_map_viz)

#     gt_EE_dtc_map = zarr_dataset['data']['gt_contact']['observation.EE_dtc_map'][idx]
#     estimated_EE_dtc_map = zarr_dataset['episode_data']['episode_0'][contact_model_group_name]['observation.EE_dtc_map'][i]

#     gt_EE_dtc_viz = cv2.applyColorMap(
#         np.clip(
#             (dtc_threshold_viz - np.clip(gt_EE_dtc_map, 0, dtc_threshold_viz))*(255/dtc_threshold_viz),
#             0,
#             255).astype(np.uint8),
#         cv2.COLORMAP_JET
#     )
#     estimated_EE_dtc_viz = cv2.applyColorMap(
#         np.clip(
#             (dtc_threshold_viz - np.clip(estimated_EE_dtc_map, 0, dtc_threshold_viz))*(255/dtc_threshold_viz),
#             0,
#             255).astype(np.uint8),
#         cv2.COLORMAP_JET
#     )
#     gt_EE_dtc_viz = cv2.addWeighted(gt_EE_dtc_viz, 0.5, rgb_image, 0.5, 0)
#     estimated_EE_dtc_viz = cv2.addWeighted(estimated_EE_dtc_viz, 0.5, rgb_image, 0.5, 0)
#     gt_EE_dtc_images.append(gt_EE_dtc_viz)
#     estimated_EE_dtc_images.append(estimated_EE_dtc_viz)

#     gt_env_dtc_map = zarr_dataset['data']['gt_contact']['observation.env_dtc_map'][idx]
#     estimated_env_dtc_map = zarr_dataset['episode_data']['episode_0'][contact_model_group_name]['observation.env_dtc_map'][i]

#     gt_env_dtc_viz = cv2.applyColorMap(
#         np.clip(
#             (dtc_threshold_viz - np.clip(gt_env_dtc_map, 0, dtc_threshold_viz))*(255/dtc_threshold_viz),
#             0,
#             255).astype(np.uint8),
#         cv2.COLORMAP_JET
#     )
#     estimated_env_dtc_viz = cv2.applyColorMap(
#         np.clip(
#             (dtc_threshold_viz - np.clip(estimated_env_dtc_map, 0, dtc_threshold_viz))*(255/dtc_threshold_viz),
#             0,
#             255).astype(np.uint8),
#         cv2.COLORMAP_JET
#     )
#     gt_env_dtc_viz = cv2.addWeighted(gt_env_dtc_viz, 0.5, rgb_image, 0.5, 0)
#     estimated_env_dtc_viz = cv2.addWeighted(estimated_env_dtc_viz, 0.5, rgb_image, 0.5, 0)
#     gt_env_dtc_images.append(gt_env_dtc_viz)
#     estimated_env_dtc_images.append(estimated_env_dtc_viz)
# #%%
# gt_env_dtc_video_path = Path("./gt_env_dtc_video.mp4")
# estimated_env_dtc_video_path = Path(f"./{contact_model_group_name}_estimated_env_dtc_video.mp4")
# imageio.mimwrite(gt_env_dtc_video_path, gt_env_dtc_images, fps=20, quality=8, macro_block_size=None)
# imageio.mimwrite(estimated_env_dtc_video_path, estimated_env_dtc_images, fps=20, quality=8, macro_block_size=None)
# #%%
# gt_EE_dtc_video_path = Path("./gt_EE_dtc_video.mp4")
# estimated_EE_dtc_video_path = Path(f"./{contact_model_group_name}_estimated_EE_dtc_video.mp4")
# imageio.mimwrite(gt_EE_dtc_video_path, gt_EE_dtc_images, fps=20, quality=8, macro_block_size=None)
# imageio.mimwrite(estimated_EE_dtc_video_path, estimated_EE_dtc_images, fps=20, quality=8, macro_block_size=None)
# #%%
# gt_contact_video_path = Path("./gt_contact_video.mp4")
# estimated_contact_video_path = Path(f"./{contact_model_group_name}_estimated_contact_video.mp4")
# imageio.mimwrite(gt_contact_video_path, gt_contact_images, fps=20, quality=8, macro_block_size=None)
# imageio.mimwrite(estimated_contact_video_path, estimated_contact_images, fps=20, quality=8, macro_block_size=None)

#%%
@click.command()
@click.argument('episode-idx', type=int)
@click.argument('path-to-demo-root-dir', type=click.Path(exists=True, path_type=Path))
@click.argument('demo-name', type=str)
@click.argument('contact_model_id', type=str)
@click.argument('checkpoint_name', type=str)
@click.argument('segmentation_model_name', type=str)
def main(episode_idx, path_to_demo_root_dir, demo_name, contact_model_id, checkpoint_name, segmentation_model_name):
    # episode_idx = 0
    # path_to_demo_root_dir = Path("/mnt/crucialSSD/datasetsSSD/fish_datasets/simulated/teleop/FISH/expert_demos/frankagym/FrankaInsertion-v1")
    # demo_name = "1232_sim_w_recovery_demos_leftof4thbook_springbookends_graspedrand_noenvrand_noslotrand_20hz_act"
    # # contact_model_id = "175604_2" # no mask, no flow, just context
    # # checkpoint_name = "epoch=09-val_loss=0.00.ckpt"
    # contact_model_id = "197406_2" # w/ mask no flow, just context
    # checkpoint_name = "epoch=08-val_loss=0.00.ckpt" # w/
    epoch = int(checkpoint_name[checkpoint_name.find('epoch=') + len('epoch='):checkpoint_name.find('-')])
    contact_model_data_group_name = f"contact_model_{contact_model_id}_epoch_{epoch}"
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
    path_to_contact_estimation = Path(__file__).parents[2] / "contact_estimation"
    assert path_to_contact_estimation.exists(), f"Path {path_to_contact_estimation} does not exist. Please check the path."
    path_to_artifacts = path_to_contact_estimation / "artifacts"
    assert path_to_artifacts.exists(), f"Path {path_to_artifacts} does not exist. Please check the path."
    # contact_model_id = "175604_2"
    # checkpoint_name = "epoch=09-val_loss=0.00.ckpt"
    contact_model_path = path_to_artifacts / contact_model_id / "checkpoints" / checkpoint_name
    assert contact_model_path.exists(), f"Path {contact_model_path} does not exist. Please check the path."
    camera_K = torch.from_numpy(zarr_dataset['meta']['episode_cam_K'][episode_idx]).to(device)
    cam_tf_world = torch.from_numpy(zarr_dataset['meta']['episode_cam_tf_world'][episode_idx]).to(device)
    raw_height, raw_width = zarr_dataset['data']['observation.depth'].shape[1:3]
    contact_estimation_model = ContactPredictor(
        contact_model_path,
        camera_K,
        cam_tf_world,
        raw_height,
        raw_width,
        device=device,
        fill_depth_zeros=True,
    )
    #%%
    # for episode_idx in episode_idxs:
    episode_data_group_name = f"episode_data/episode_{episode_idx}"
    if episode_data_group_name not in zarr_dataset:
        zarr_dataset.create_group(episode_data_group_name)
        print(f"Created group {episode_data_group_name} in zarr dataset.")
    episode_data_contact_group_name = episode_data_group_name + '/' + contact_model_data_group_name
    if episode_data_contact_group_name not in zarr_dataset:
        zarr_dataset.create_group(episode_data_contact_group_name)
        print(f"Created group {episode_data_contact_group_name} in episode data group {episode_data_group_name} in zarr dataset.")

    episode_data_contact_map_array_name = episode_data_contact_group_name + '/observation.contact_map'
    episode_data_EE_dtc_map_array_name = episode_data_contact_group_name + '/observation.EE_dtc_map'
    episode_data_EE_normals_map_array_name = episode_data_contact_group_name + '/observation.EE_normals_map'
    episode_data_env_dtc_map_array_name = episode_data_contact_group_name + '/observation.env_dtc_map'
    episode_data_env_normals_map_array_name = episode_data_contact_group_name + '/observation.env_normals_map'

    #%%
    # check if already exists
    episode_length = zarr_dataset['meta']['episode_ends'][episode_idx] - (0 if episode_idx == 0 else zarr_dataset['meta']['episode_ends'][episode_idx - 1])
    if episode_data_contact_map_array_name in zarr_dataset:
        episode_data_array_length = zarr_dataset[episode_data_contact_map_array_name].shape[0]
        assert episode_data_array_length == zarr_dataset[episode_data_EE_dtc_map_array_name].shape[0] == zarr_dataset[episode_data_EE_normals_map_array_name].shape[0] == zarr_dataset[episode_data_env_dtc_map_array_name].shape[0] == zarr_dataset[episode_data_env_normals_map_array_name].shape[0], "All episode data arrays must have the same length."
        # if exists then check if length matches episode length
        if episode_data_array_length == episode_length:
            print(f"Episode data arrays already exist for episode {episode_idx} and have the correct length {episode_length}. Skipping mask generation.")
            return
        else:
            print(f"Episode data arrays already exist for episode {episode_idx} but have length {episode_data_array_length} which does not match episode length {episode_length}. Re-generating masks.")
        

    #%%
    mask_input_dict = MaskInputDict(enable=contact_estimation_model.contact_model_uses_mask, mask_list=['EE_obj_mask'], representation='channels', segmentation_model_name=segmentation_model_name)
    # mask_input_dict = MaskInputDict(enable=contact_estimation_model.contact_model_uses_mask, mask_list=['EE_obj_mask'], representation='channels', segmentation_model_name='gt_segmentation')
    observation_cfg = VisualFeatureSet(use_color=True, use_depth=True, mask_input_dict=mask_input_dict, use_contact_map=False, use_sdf_maps=False, use_normals_maps=False, use_contact_forces_map=False)
    action_horizon_length = 1
    action_history_length = 1
    action_config = ActionConfig(horizon_length=action_horizon_length, action_frame_expression='delta', input_rotation_representation='euler_angles')
    action_history_config = ActionHistoryConfig(enable=False, history_length=action_history_length, action_frame_expression='delta', action_frame='current_end_effector', rotation_representation='euler_angles')
    end_effector_wrench_history_config = EndEffectorWrenchHistoryConfig(enable=False)
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
                                        check_data_is_sum_of_episode_lengths=False,
                                        )
    dataloader = DataLoader(episode_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    #%%
    gt_contact_map_shape = zarr_dataset['data']['gt_contact']['observation.contact_map'].shape[1:]
    gt_contact_map_dtype = zarr_dataset['data']['gt_contact']['observation.contact_map'].dtype
    gt_contact_map_compressors = zarr_dataset['data']['gt_contact']['observation.contact_map'].compressors[0]

    gt_EE_dtc_map_shape = gt_contact_map_shape
    gt_EE_dtc_map_dtype = np.float32
    gt_EE_dtc_map_compressors = gt_contact_map_compressors
    # gt_EE_dtc_map_shape = zarr_dataset['data']['gt_contact']['observation.EE_dtc_map'].shape[1:]
    # gt_EE_dtc_map_dtype = zarr_dataset['data']['gt_contact']['observation.EE_dtc_map'].dtype
    # gt_EE_dtc_map_compressors = zarr_dataset['data']['gt_contact']['observation.EE_dtc_map'].compressors[0]

    gt_EE_normals_map_shape = gt_contact_map_shape
    gt_EE_normals_map_shape[-1] = 3
    gt_EE_normals_map_dtype = np.float32
    gt_EE_normals_map_compressors = gt_contact_map_compressors
    # gt_EE_normals_map_shape = zarr_dataset['data']['gt_contact']['observation.EE_normals_map'].shape[1:]
    # gt_EE_normals_map_dtype = zarr_dataset['data']['gt_contact']['observation.EE_normals_map'].dtype
    # gt_EE_normals_map_compressors = zarr_dataset['data']['gt_contact']['observation.EE_normals_map'].compressors[0]     

    gt_env_dtc_map_shape = gt_contact_map_shape
    gt_env_dtc_map_dtype = np.float32
    gt_env_dtc_map_compressors = gt_contact_map_compressors
    # gt_env_dtc_map_shape = zarr_dataset['data']['gt_contact']['observation.env_dtc_map'].shape[1:]
    # gt_env_dtc_map_dtype = zarr_dataset['data']['gt_contact']['observation.env_dtc_map'].dtype
    # gt_env_dtc_map_compressors = zarr_dataset['data']['gt_contact']['observation.env_dtc_map'].compressors[0]       

    gt_env_normals_map_shape = gt_contact_map_shape
    gt_env_normals_map_shape[-1] = 3
    gt_env_normals_map_dtype = np.float32
    gt_env_normals_map_compressors = gt_contact_map_compressors
    # gt_env_normals_map_shape = zarr_dataset['data']['gt_contact']['observation.env_normals_map'].shape[1:]
    # gt_env_normals_map_dtype = zarr_dataset['data']['gt_contact']['observation.env_normals_map'].dtype
    # gt_env_normals_map_compressors = zarr_dataset['data']['gt_contact']['observation.env_normals_map'].compressors[0]

    #%%
    ## ##############
    ## generate masks and write to disk
    ## ##############
    contact_estimation_model.reset()
    zarr_dataset.create_array(episode_data_contact_map_array_name, shape=(0, *gt_contact_map_shape), chunks=(1, *gt_contact_map_shape), dtype=gt_contact_map_dtype, compressors=gt_contact_map_compressors, overwrite=True)
    zarr_dataset.create_array(episode_data_EE_dtc_map_array_name, shape=(0, *gt_EE_dtc_map_shape), chunks=(1, *gt_EE_dtc_map_shape), dtype=gt_EE_dtc_map_dtype, compressors=gt_EE_dtc_map_compressors, overwrite=True)
    zarr_dataset.create_array(episode_data_EE_normals_map_array_name, shape=(0, *gt_EE_normals_map_shape), chunks=(1, *gt_EE_normals_map_shape), dtype=gt_EE_normals_map_dtype, compressors=gt_EE_normals_map_compressors, overwrite=True)
    zarr_dataset.create_array(episode_data_env_dtc_map_array_name, shape=(0, *gt_env_dtc_map_shape), chunks=(1, *gt_env_dtc_map_shape), dtype=gt_env_dtc_map_dtype, compressors=gt_env_dtc_map_compressors, overwrite=True)
    zarr_dataset.create_array(episode_data_env_normals_map_array_name, shape=(0, *gt_env_normals_map_shape), chunks=(1, *gt_env_normals_map_shape), dtype=gt_env_normals_map_dtype, compressors=gt_env_normals_map_compressors, overwrite=True)

    #%%
    for i, batch in tqdm(enumerate(dataloader)):
    # batch = next(iter(dataloader))
    # i = 1
        # put all batch items on the device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        depth_image = batch['observation.depth'][0]
        EE_pose = batch['observation.state'][0,0,:7]
        mask = None
        if contact_estimation_model.contact_model_uses_mask:
            mask = batch['observation.EE_obj_mask'][0]
        #%%
        if i == 0: #initialize cutie
            # mask = batch['observation.EE_obj_mask'][0,0,0].cpu().numpy()
            # mask = mask_predictor.start_mask_tracking_from_mask(color_image, mask)
            contact_model_output_dict = contact_estimation_model.start_contact_prediction(depth_image, EE_pose, grasped_obj_mask=mask)
        else:
            # with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
            #     out_obj_ids, out_mask_logits = sam2_online_predictor.track(color_image)
            contact_model_output_dict = contact_estimation_model.continue_contact_prediction(depth_image, EE_pose, mask=mask)
        # if episode_data_mask_array_name not in zarr_dataset:
        #%%
        contact_map = einops.rearrange(torch.sigmoid(contact_model_output_dict['contact_prob_map_logit']), 'c h w -> h w c').cpu().numpy()
        EE_dtc_map = einops.rearrange(contact_model_output_dict['envsdf_map'], 'c h w -> h w c').cpu().numpy()
        EE_normals_map = einops.rearrange(contact_model_output_dict['EE_obj_normals_map'], 'b c h w -> b h w c').cpu().numpy()
        env_dtc_map = einops.rearrange(contact_model_output_dict['EEsdf_map'], 'c h w -> h w c').cpu().numpy()
        env_normals_map = einops.rearrange(contact_model_output_dict['env_normals_map'], 'b c h w -> b h w c').cpu().numpy()
        #%%
        zarr_dataset[episode_data_contact_map_array_name].append(contact_map[np.newaxis, ...].astype(gt_contact_map_dtype))
        zarr_dataset[episode_data_EE_dtc_map_array_name].append(EE_dtc_map[np.newaxis, ...].astype(gt_EE_dtc_map_dtype))
        zarr_dataset[episode_data_EE_normals_map_array_name].append(EE_normals_map.astype(gt_EE_normals_map_dtype))
        zarr_dataset[episode_data_env_dtc_map_array_name].append(env_dtc_map[np.newaxis, ...].astype(gt_env_dtc_map_dtype))
        zarr_dataset[episode_data_env_normals_map_array_name].append(env_normals_map.astype(gt_env_normals_map_dtype))
#%%
if __name__ == "__main__":
    main()
    print(f"Saved masks to {path_to_zarr}.")
    print("Done.")