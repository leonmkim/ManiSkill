#%%
import transformers
from transformers import AutoModel, AutoConfig
import gymnasium as gym
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

import torchvision.transforms.v2 as T
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionRgbEncoder, _replace_submodules
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig, ActionConfig, ActionHistoryConfig
import torchvision
from torch import nn

# from theia.decoding import load_feature_stats, prepare_depth_decoder
#%%
path_to_fish_leon = Path(__file__).parents[2]
path_to_FISH = path_to_fish_leon / "FISH"
assert path_to_FISH.exists(), f"Path {path_to_FISH} does not exist. Please check the path."
sys.path.append(str(path_to_FISH))
from agent.encoder import VisualFeatureSet, VisualFeaturePreprocessor
from lerobot.common.policies.diffusion.configuration_diffusion import ActionConfig, ActionHistoryConfig
from agent.encoder import MaskInputDict
from dataset.expert_dataset import ExpertDatasetZarr
#%%
path_to_demo_root_dir = Path("/mnt/crucialSSD/datasetsSSD/fish_datasets/simulated/teleop/FISH/expert_demos/frankagym/FrankaInsertion-v1")
demo_name = "1232_sim_w_recovery_demos_leftof4thbook_springbookends_graspedrand_noenvrand_noslotrand_20hz_act"
path_to_demo_dir = path_to_demo_root_dir / demo_name
assert path_to_demo_dir.exists(), f"Path {path_to_demo_dir} does not exist. Please check the path."

path_to_zarr = path_to_demo_dir / "demos.zarr"
path_to_json = path_to_demo_dir / "demos.json"

assert path_to_zarr.exists(), f"Path {path_to_zarr} does not exist. Please check the path."
assert path_to_json.exists(), f"Path {path_to_json} does not exist. Please check the path."
zarr_dataset = zarr.open(str(path_to_zarr), mode='r+')

#%%
@click.command()
@click.argument('episode-idx', type=int)
@click.argument('path-to-demo-root-dir', type=click.Path(exists=True, path_type=Path))
@click.argument('demo-name', type=str)
@click.argument('theia-model-string', type=str)
@click.argument('desired-theia-size', type=(click.Tuple([int, int])))
def main(episode_idx, path_to_demo_root_dir, demo_name, theia_model_string, desired_theia_size):
# episode_idx = 0
# path_to_demo_root_dir = Path("/mnt/crucialSSD/datasetsSSD/fish_datasets/simulated/teleop/FISH/expert_demos/frankagym/FrankaInsertion-v1")
# # demo_name = "1232_sim_w_recovery_demos_leftof4thbook_springbookends_graspedrand_noenvrand_noslotrand_20hz_act"
# demo_name = "2_demo_test"
# theia_model_string = 'theia-base-patch16-224-cdiv'
# # theia_model_string = 'theia-base-patch16-224-cddsv'

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
    # theia_config = AutoConfig.from_pretrained(f'theaiinstitute/{theia_model_string}', trust_remote_code=True)
    # theia_config.pretrained = True
    # for key in theia_config.target_feature_sizes:
    #     theia_config.target_feature_sizes[key][1:] = desired_theia_size
    # theia_config.num_reg_tokens = 1

    #%%
    theia_model = AutoModel.from_pretrained(f'theaiinstitute/{theia_model_string}', trust_remote_code=True)
    # theia_model = AutoModel.from_pretrained(f'theaiinstitute/{theia_model_string}', config=theia_config, trust_remote_code=True)
    # theia_model = AutoModel.from_config(theia_config, trust_remote_code=True)
    theia_model.eval()
    theia_model.to(device)

    #%%
    patch_size = theia_model.backbone.model.config.patch_size
    # patch_size = 16
    original_image_size = zarr_dataset['data']['observation.rgb'].shape[1:3]
    desired_theia_size = (15,20)
    assert isinstance(desired_theia_size, tuple) and len(desired_theia_size) == 2, "desired_theia_size should be a tuple of two integers."
    # compute the rescale factor based on the original image size and desired theia size
    rescale_factor = (desired_theia_size[0] * patch_size) / original_image_size[0]
    assert rescale_factor == (desired_theia_size[1] * patch_size) / original_image_size[1], "Rescale factor should be the same for both dimensions."
    # rescale_factor = 0.875 # if concatenating at the end of the 3rd resnet block (15x20 for original 240x320 image)
    # rescale_factor = 1.75 # if concatenating at the end of the 2nd resnet block (30x40 for original 240x320 image)
    # theia_model_string_with_rescale = f"{theia_model_string}_rescale_{rescale_factor}"
    theia_model_string_with_rescale = f"{theia_model_string}_{desired_theia_size[0]}x{desired_theia_size[1]}"
    #%%
    theia_resize = (((original_image_size[0]*rescale_factor) // patch_size)*patch_size, ((original_image_size[1]*rescale_factor) // patch_size)*patch_size)
    theia_output_size = (int(theia_resize[0] // patch_size), int(theia_resize[1] // patch_size))
    # convert back to int
    theia_resize = (int(theia_resize[0]), int(theia_resize[1]))
    #%%
    # # #################################
    # # playing with the resnet encoder
    # # #################################
    # # load a trained model
    # path_to_checkpoint = Path("/mnt/bighdd/fish_contact_backup/exp_local/frankagym_pixels/FrankaInsertion-v1/109581_0/snapshot_150000.pt")
    # checkpoint = torch.load(path_to_checkpoint, map_location=device)
    # #%%
    # rgb_encoder_dict = {k: v for k, v in checkpoint['diffusion'].items() if 'rgb_encoder.backbone' in k}
    # # strip the 'rgb_encoder.backbone.' prefix from the keys
    # rgb_encoder_dict = {k.replace('rgb_encoder.backbone.', ''): v for k, v in rgb_encoder_dict.items()}

    # #%%
    # full_resnet_encoder = getattr(torchvision.models, 'resnet18')(
    #             weights=None
    # )
    # # resnet_encoder = resnet_encoder.to(device)
    # #%%
    # full_resnet_layers = list(full_resnet_encoder.children())[:-2] # get to third block
    # full_resnet = nn.Sequential(*(full_resnet_layers))
    # full_resnet = _replace_submodules(
    #     root_module=full_resnet,
    #     predicate=lambda x: isinstance(x, nn.BatchNorm2d),
    #     func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
    # )
    # # change first conv layer to take in 4 channels
    # full_resnet[0] = nn.Conv2d(
    #     in_channels=4,
    #     out_channels=full_resnet[0].out_channels,
    #     kernel_size=full_resnet[0].kernel_size,
    #     stride=full_resnet[0].stride,
    #     padding=full_resnet[0].padding,
    #     bias=False,
    # )
    # #%%
    # # load rgb_encoder weights into full_resnet
    # full_resnet.load_state_dict(rgb_encoder_dict, strict=False)
    # full_resnet = full_resnet.to(device)
    # #%%
    # # count number of parameters in full_resnet
    # num_params_full_resnet = sum(p.numel() for p in full_resnet.parameters() if p.requires_grad)
    # print(f"Number of parameters in full_resnet: {num_params_full_resnet}")
    # #%%
    # # backbone_layers = list(resnet_encoder.children())[:-2] # gets rid of the adaptiveavgpool and linear layer
    # # backbone_pre_dino_layers = list(resnet_encoder.children())[:-4]
    # backbone_pre_dino_encoder = getattr(torchvision.models, 'resnet18')(
    #             weights=None
    # )
    # resnet_fusion_index = -3
    # backbone_pre_dino_layers = list(backbone_pre_dino_encoder.children())[:resnet_fusion_index] # get to third block
    # backbone_pre_dino = nn.Sequential(*(backbone_pre_dino_layers))
    # backbone_pre_dino = _replace_submodules(
    #     root_module=backbone_pre_dino,
    #     predicate=lambda x: isinstance(x, nn.BatchNorm2d),
    #     func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
    # )
    # # change first conv layer to take in 4 channels
    # backbone_pre_dino[0] = nn.Conv2d(
    #     in_channels=4,
    #     out_channels=backbone_pre_dino[0].out_channels,
    #     kernel_size=backbone_pre_dino[0].kernel_size,
    #     stride=backbone_pre_dino[0].stride,
    #     padding=backbone_pre_dino[0].padding,
    #     bias=False,
    # )
    # #%%
    # # load rgb_encoder weights into backbone_pre_dino
    # backbone_pre_dino.load_state_dict(rgb_encoder_dict, strict=False)
    # backbone_pre_dino = backbone_pre_dino.to(device)

    # #%%
    # pre_dino_feature_dim = backbone_pre_dino_layers[-1][-1].bn2.num_channels
    # dinov2_feature_dim = dinov2_model.num_features
    # dino_adapter_output_dim = dinov2_feature_dim // 2
    # # dino_adapter_output_dim = 512
    # # dino_adapter_output_dim = dinov2_feature_dim

    # #%%
    # post_dino_input_feature_dim = pre_dino_feature_dim + dino_adapter_output_dim
    # backbone_post_dino_encoder = getattr(torchvision.models, 'resnet18')(
    #             weights=None
    # )
    # backbone_post_dino_layers = list(backbone_post_dino_encoder.children())[resnet_fusion_index:-2] # gets rid of the adaptiveavgpool and linear layer
    # backbone_post_dino = nn.Sequential(*(backbone_post_dino_layers))
    # backbone_post_dino = _replace_submodules(
    #     root_module=backbone_post_dino,
    #     predicate=lambda x: isinstance(x, nn.BatchNorm2d),
    #     func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
    # )
    # # replace the input conv layers with a conv layer that takes in the concatenated features
    # backbone_post_dino[0][0].conv1 = nn.Conv2d(
    #     in_channels=post_dino_input_feature_dim,
    #     out_channels=backbone_post_dino[0][0].conv1.out_channels,
    #     kernel_size=backbone_post_dino[0][0].conv1.kernel_size,
    #     stride=backbone_post_dino[0][0].conv1.stride,
    #     padding=backbone_post_dino[0][0].conv1.padding,
    #     bias=False,
    # )
    # backbone_post_dino[0][0].downsample[0] = nn.Conv2d(
    #     in_channels=post_dino_input_feature_dim,
    #     out_channels=backbone_post_dino[0][0].downsample[0].out_channels,
    #     kernel_size=backbone_post_dino[0][0].downsample[0].kernel_size,
    #     stride=backbone_post_dino[0][0].downsample[0].stride,
    #     padding=backbone_post_dino[0][0].downsample[0].padding,
    #     bias=False,
    # )

    # backbone_post_dino = backbone_post_dino.to(device)
    # #%%
    # dino_patch_feature_to_resnet_adapter = nn.Sequential(
    #     nn.Conv2d(
    #         in_channels=dinov2_feature_dim,
    #         out_channels=dino_adapter_output_dim,
    #         kernel_size=(1,1),
    #         stride=(1,1),
    #         padding=(0,0),
    #         bias=False,
    #     ),
    #     nn.GroupNorm(num_groups=backbone_post_dino[0][0].bn1.num_groups, num_channels=dino_adapter_output_dim, eps=backbone_post_dino[0][0].bn1.eps, affine=True),
    #     nn.ReLU(inplace=True),
    # )
    # dino_patch_feature_to_resnet_adapter = dino_patch_feature_to_resnet_adapter.to(device)

    # #%%
    # # count number of parameters in backbone_pre and backbone_post_dino
    # num_params_resnet_fuse_dino = sum(p.numel() for p in backbone_pre_dino.parameters() if p.requires_grad) + sum(p.numel() for p in backbone_post_dino.parameters() if p.requires_grad) + sum(p.numel() for p in dino_patch_feature_to_resnet_adapter.parameters() if p.requires_grad)
    # print(f"Number of parameters in backbone_pre_dino, adapter, and backbone_post_dino: {num_params_resnet_fuse_dino}")

    # #%%
    # num_params_greater_than_full_resnet = num_params_resnet_fuse_dino - num_params_full_resnet
    # print(f"Number of parameters in backbone_pre_dino, adapter, and backbone_post_dino greater than full_resnet: {num_params_greater_than_full_resnet}")

    # # #################################
    # # playing with the resnet encoder
    # # #################################

    #%%
    # # #################################
    # # PCA sandbox
    # # #################################
    # transform = T.Compose([
    #     T.ToPILImage(),
    #     T.Resize(theia_resize, interpolation=T.InterpolationMode.BICUBIC),
    #     # T.CenterCrop((238, 308)),
    #     T.ToImage(),
    #     # T.ToDtype(torch.float32, scale=True),
    #     # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])
    # theia_output_size = (theia_resize[0] // patch_size, theia_resize[1] // patch_size)
    # from sklearn.decomposition import PCA
    # rgb_image = zarr_dataset['data']['observation.rgb'][0]
    # plt.imshow(rgb_image)
    # #%%
    # rgb_image = einops.rearrange(transform(rgb_image).unsqueeze(0).to(device), 'b c h w -> b h w c')
    # with torch.inference_mode():
    #     theia_output = theia_model.forward_feature(rgb_image, do_resize=False, interpolate_pos_encoding=True)
    # #%%
    # with torch.inference_mode():
    #     predicted_features = theia_model(rgb_image)
    # #%%
    # # dinov2_output = predicted_features['facebook/dinov2-large']
    # depth_output = predicted_features['LiheYoung/depth-anything-large-hf']

    # #%%
    # pca = PCA(n_components=3)
    # pca_features = pca.fit_transform(theia_output[0].cpu().numpy())
    # # pca_features = pca.fit_transform(dinov2_output[0].cpu().numpy())

    # #%%
    # pca_features_rgb = (pca_features - np.min(pca_features)) / (np.max(pca_features) - np.min(pca_features))
    # pca_features_rgb = (pca_features_rgb * 255).astype(np.uint8)
    # pca_features_rgb = pca_features_rgb.reshape(*theia_output_size, 3)
    # # pca_features_rgb = pca_features_rgb.reshape(16, 16, 3)
    # plt.imshow(pca_features_rgb)
    # #%%
    # pca_features_single_channel = pca_features[:, 2].reshape(*theia_output_size)
    # pca_features_single_channel = (pca_features_single_channel - np.min(pca_features_single_channel)) / (np.max(pca_features_single_channel) - np.min(pca_features_single_channel))
    # pca_features_single_channel = (pca_features_single_channel * 255).astype(np.uint8)
    # pca_features_single_channel = pca_features_single_channel.reshape(*theia_output_size, 1)
    # plt.imshow(pca_features_single_channel, cmap='gray')
    # plt.colorbar()
    # #%%
    # pca_features_single_channel_thresholded = pca_features_single_channel > 150
    # # extract pca features from the mask
    # plt.imshow(pca_features_single_channel_thresholded, cmap='gray')
    # #%%
    # dino_features_foreground = features['x_norm_patchtokens'][0].cpu().numpy()[pca_features_single_channel_thresholded.flatten()]
    # pca_features_foreground = pca.fit_transform(dino_features_foreground)
    # #%%
    # pca_features_foreground_rgb = (pca_features_foreground - np.min(pca_features_foreground)) / (np.max(pca_features_foreground) - np.min(pca_features_foreground))
    # pca_features_foreground_rgb = (pca_features_foreground_rgb * 255).astype(np.uint8)
    # pca_features_foreground_rgb_image = np.zeros((*dinov2_output_size, 3), dtype=np.uint8)
    # pca_features_foreground_rgb_image[pca_features_single_channel_thresholded.reshape(*dinov2_output_size)] = pca_features_foreground_rgb.reshape(-1, 3)
    # plt.imshow(pca_features_foreground_rgb_image)
    # #%%
    # plt.imshow(pca_features_foreground_rgb_image[..., 2], cmap='gray')
    # plt.colorbar()
    # # #################################
    # # PCA sandbox
    # # #################################
    #%%
    # from transformers import AutoImageProcessor, AutoModel
    # from PIL import Image
    # processor = AutoImageProcessor.from_pretrained("facebook/dinov2-with-registers-base")
    # model = AutoModel.from_pretrained("facebook/dinov2-with-registers-base")

    # #%%
    if 'episode_data' not in zarr_dataset:
        zarr_dataset.create_group('episode_data')
        print("Created group 'episode_data' in zarr dataset.")
    #%%
    compressors = zarr_dataset['data']['action'].compressors
    #%%
    episode_data_group_name = f"episode_data/episode_{episode_idx}"
    if episode_data_group_name not in zarr_dataset:
        zarr_dataset.create_group(episode_data_group_name)
        print(f"Created group {episode_data_group_name} in zarr dataset.")
    episode_data_theia_group_name = episode_data_group_name + '/' + theia_model_string_with_rescale
    if theia_model_string_with_rescale not in zarr_dataset[episode_data_group_name]:
        zarr_dataset.create_group(episode_data_theia_group_name)
        print(f"Created group {theia_model_string_with_rescale} in episode data group {episode_data_group_name} in zarr dataset.")

    # x_norm_clstoken_array_name = episode_data_theia_group_name + '/observation.x_norm_clstoken'
    # # if x_norm_clstoken_array_name not in zarr_dataset:
    # zarr_dataset.create_array(x_norm_clstoken_array_name, shape=(0, 1, theia_model.num_features), dtype=np.float32, chunks=(1, 1, theia_model.num_features), compressors=compressors, overwrite=True)
    # print(f"Created dataset {x_norm_clstoken_array_name} in zarr dataset.")
    # x_norm_regtokens_array_name = episode_data_theia_group_name + '/observation.x_norm_regtokens'
    # # if x_norm_regtokens_array_name not in zarr_dataset:
    # zarr_dataset.create_array(x_norm_regtokens_array_name, shape=(0, theia_model.num_register_tokens, theia_model.num_features), dtype='float32', chunks=(1, theia_model.num_register_tokens, theia_model.num_features), compressors=compressors, overwrite=True)
    # print(f"Created dataset {x_norm_regtokens_array_name} in zarr dataset.")
    x_norm_patchtokens_array_name = episode_data_theia_group_name + '/observation.x_norm_patchtokens'
    # if x_norm_patchtokens_array_name not in zarr_dataset:
    zarr_dataset.create_array(x_norm_patchtokens_array_name, shape=(0, *theia_output_size, theia_model.backbone.get_feature_size()[0]), dtype='float32', chunks=(1, *theia_output_size, theia_model.backbone.get_feature_size()[0]), compressors=compressors, overwrite=True)
    print(f"Created dataset {x_norm_patchtokens_array_name} in zarr dataset.")

    #%%
    mask_input_dict = MaskInputDict(enable=False, mask_list=['EE_obj_mask'], representation='channels', segmentation_model_name='gt_segmentation')
    observation_cfg = VisualFeatureSet(use_color=True, use_depth=False, 
                                    mask_input_dict=mask_input_dict, 
                                    use_contact_map=False, use_sdf_maps=False, use_normals_maps=False,
                                    zero_centered=True, z_score_normalize_rgb=True,
                                    )
    # visual_preprocessor = VisualFeaturePreprocessor(
    #     cfg = observation_cfg,
    #     device=device,
    # )
    #%%
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
                                        # repeat_padding_for_actions=True,
                                        # action_using_env_state_indices=False,
                                        # stored_action_frame_expression='absolute',
                                        )
    dataloader = DataLoader(episode_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    #%%
    ## ##############
    ## generate masks and write to disk
    ## ##############
    #%%
    theia_color_transform = T.Compose([
        # T.ToPILImage(),
        # T.Resize(theia_resize, interpolation=T.InterpolationMode.BICUBIC),
        # T.CenterCrop((238, 308)),
        T.ToImage(),
        # T.ToDtype(torch.float32, scale=True),
        # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # batch = next(iter(dataloader))
    for i, batch in tqdm(enumerate(dataloader)):
        color_image = batch['observation.rgb'][:, 0].to(torch.uint8).to(device)
        # depth_image = batch['observation.depth'][:, 0].to(device)
        #%%

        # resnet_transform = T.Compose([
        #     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ])
        #%%
        # inputs, _ = visual_preprocessor.normalize_images(batch)
        # inputs = inputs[:, 0]
        # inputs = inputs.to(device)
        # #%%
        # color_to_grayscale = T.Grayscale(num_output_channels=3)
        # inputs[:, :3] = color_to_grayscale(inputs[:, :3])
        # #%%
        # inputs = visual_preprocessor.post_augmentation_normalize_images(inputs.unsqueeze(0).unsqueeze(0), checkpoint['dataset_statistics']).squeeze(0).squeeze(0)
        # #%%

        with torch.inference_mode():
            color_image_for_theia = theia_color_transform(color_image)
            patch_features = theia_model.forward_feature(color_image_for_theia, do_resize=False, interpolate_pos_encoding=True)
            patch_features = einops.rearrange(patch_features, 'b (h w) c -> b h w c', h=theia_output_size[0], w=theia_output_size[1])
            # rgb_image_tensor = transform(rgb_image).unsqueeze(0).to(device)
            zarr_dataset[x_norm_patchtokens_array_name].append(patch_features.cpu().numpy())

            # clstoken_features = features['x_norm_clstoken'].unsqueeze(1)
            # zarr_dataset[x_norm_clstoken_array_name].append(clstoken_features.cpu().numpy())

            # regtokens_features = features['x_norm_regtokens']
            # zarr_dataset[x_norm_regtokens_array_name].append(regtokens_features.cpu().numpy())
            
            # intermediate_features = dinov2_model.get_intermediate_layers(color_image)

            # patch_features = dino_patch_feature_to_resnet_adapter(patch_features)

            # resnet_output = resnet_encoder(color_image)
            # color_image_for_resnet = resnet_transform(color_image)
            # pre_dino_resnet_output = backbone_pre_dino(inputs)

            # concatenate the features
            # concatenated_features = torch.cat((pre_dino_resnet_output, patch_features), dim=1)
            # pass through the post-dino backbone
            # post_dino_resnet_output = backbone_post_dino(concatenated_features)

            # full_resnet_output = full_resnet(inputs)
    #%%
    episode_end = zarr_dataset['meta']['episode_ends'][episode_idx]
    episode_start = zarr_dataset['meta']['episode_starts'][episode_idx-1] if episode_idx > 0 else 0
    episode_length = episode_end - episode_start
    #%%
    # assert zarr_dataset[x_norm_patchtokens_array_name].shape[0] == zarr_dataset[x_norm_clstoken_array_name].shape[0] == zarr_dataset[x_norm_regtokens_array_name].shape[0], "The number of elements in the x_norm_patchtokens, x_norm_clstoken, and x_norm_regtokens arrays should be the same."
    assert zarr_dataset[x_norm_patchtokens_array_name].shape[0] == episode_length, "The number of elements in the x_norm_patchtokens array should be equal to the episode length."
    #%%
    # visualize color image for dino
    # unnormalize the image
    # color_image_for_dino = einops.rearrange(color_image_for_dino[0], 'c h w -> h w c').cpu().numpy()
    # plt.figure(figsize=(10, 10))
    # plt.imshow(color_image_for_dino)

#%%
if __name__ == "__main__":
    main()
    # print(f"Saved masks to {path_to_zarr}.")
    print("Done.")
# %%
