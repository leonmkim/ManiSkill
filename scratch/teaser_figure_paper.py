#%%
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import natsort
import einops
from sklearn.decomposition import PCA
import zarr
import torch
from transformers import AutoModel, AutoConfig
import torchvision.transforms.v2 as T
#%%
path_to_zarr = Path("/mnt/crucialSSD/datasetsSSD/fish_datasets/real/384_240x320_all_twodim_bookends_left_to_right_annotated_start_idx_5hz_expert_demos_imp_act/demos.zarr")
zarr_root = zarr.open(str(path_to_zarr), mode='r')

#%%
device = 'cuda'
theia_model = AutoModel.from_pretrained(f'theaiinstitute/theia-base-patch16-224-cdiv', trust_remote_code=True)
# theia_model = AutoModel.from_pretrained(f'theaiinstitute/{theia_model_string}', config=theia_config, trust_remote_code=True)
# theia_model = AutoModel.from_config(theia_config, trust_remote_code=True)
theia_model.eval()
theia_model.to(device)
#%%
patch_size = theia_model.backbone.model.config.patch_size

#%%
path_to_root_episode_dir = Path("/mnt/crucialSSD/datasetsSSD/fish_datasets/real/")
dataset_name = "cook_twodim_bookends_nominal"
path_to_episode_dir = path_to_root_episode_dir / dataset_name
#%%
list_of_episodes = natsort.natsorted([x for x in path_to_episode_dir.iterdir() if x.is_dir()])
print(f"Number of episodes: {len(list_of_episodes)}")
# %%
path_to_episode_dir = list_of_episodes[0]
#%%
path_to_episode_rgb_dir = path_to_episode_dir / "color"
list_of_rgb_images = natsort.natsorted([x for x in path_to_episode_rgb_dir.iterdir() if x.suffix == ".png"])
print(f"Number of RGB images: {len(list_of_rgb_images)}")

path_to_episode_mask_dir = path_to_episode_dir / "EE_obj_maskbase_plus"
list_of_mask_images = natsort.natsorted([x for x in path_to_episode_mask_dir.iterdir() if x.suffix == ".png"])

path_to_episode_contact_map_dir = path_to_episode_dir / "local_multitask_outhd64all_mnt_crop_h144w144d48_mask_ctxtmask" / "seed_220979" / "epoch_9" / "pred_contact_prob_map"
list_of_contact_map_images = natsort.natsorted([x for x in path_to_episode_contact_map_dir.iterdir() if x.suffix == ".pkl"])

path_to_episode_theia_zarr = path_to_episode_dir / "theia-base-patch16-224-cdiv_15x20.zarr"
#%%
dir_for_visualizations = Path("./teaser_figure_visualizations")
dir_for_visualizations.mkdir(exist_ok=True, parents=True)

#%%
scratch_image_idx = 120
raw_rgb_frame = cv2.imread(str(list_of_rgb_images[scratch_image_idx]))
original_height, original_width = raw_rgb_frame.shape[0], raw_rgb_frame.shape[1]
crop_lower_x = 180
crop_upper_x = 450
crop_lower_y = 100
crop_upper_y = 420
raw_rgb_frame = cv2.cvtColor(raw_rgb_frame, cv2.COLOR_BGR2RGB)
raw_rgb_frame = raw_rgb_frame[crop_lower_y:crop_upper_y, crop_lower_x:crop_upper_x]
plt.imshow(raw_rgb_frame)
#%%
# save the figure
plt.imsave(dir_for_visualizations / "raw_rgb_frame.png", raw_rgb_frame)

#%%
contact_image_idx = 240
contact_map = np.load(str(list_of_contact_map_images[contact_image_idx]), allow_pickle=True)
contact_map = cv2.resize(contact_map, (original_width, original_height), interpolation=cv2.INTER_LANCZOS4)
contact_underlay_rgb = cv2.imread(str(list_of_rgb_images[contact_image_idx]))
contact_underlay_rgb = cv2.cvtColor(contact_underlay_rgb, cv2.COLOR_BGR2RGB)
# contact_underlay_rgb = contact_underlay_rgb[crop_lower_y:crop_upper_y, crop_lower_x:crop_upper_x]
# plt.imshow(contact_underlay_rgb)
no_contact_mask = contact_map < 0.001
max_contact = 0.07
contact_blend_alpha = 0.7
# use colormap to visualize contact map
contact_map_viz = cv2.applyColorMap((contact_map * (255/max_contact)).astype(np.uint8), cv2.COLORMAP_WINTER)
contact_map_viz = cv2.cvtColor(contact_map_viz, cv2.COLOR_BGR2RGB)
contact_underlay_rgb[~no_contact_mask] = (contact_blend_alpha*contact_map_viz[~no_contact_mask]) + (1-contact_blend_alpha)*contact_underlay_rgb[~no_contact_mask]
contact_underlay_rgb = contact_underlay_rgb[crop_lower_y:crop_upper_y, crop_lower_x:crop_upper_x].astype(np.uint8)
# plt.imshow(contact_map_viz, alpha=0.7)
plt.imshow(contact_underlay_rgb)
#%%
# save the figure
plt.imsave(dir_for_visualizations / "contact_map_viz.png", contact_underlay_rgb)

# %%
mask_image_idx = 310
mask_blend_alpha = 0.5
obj_mask = cv2.imread(str(list_of_mask_images[mask_image_idx]), cv2.IMREAD_GRAYSCALE)
obj_mask = cv2.resize(obj_mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
obj_mask = obj_mask[crop_lower_y:crop_upper_y, crop_lower_x:crop_upper_x]
mask_underlay_rgb = cv2.imread(str(list_of_rgb_images[mask_image_idx]))
mask_underlay_rgb = cv2.cvtColor(mask_underlay_rgb, cv2.COLOR_BGR2RGB)
mask_underlay_rgb = mask_underlay_rgb[crop_lower_y:crop_upper_y, crop_lower_x:crop_upper_x]
mask_underlay_rgb[obj_mask > 0] = (mask_blend_alpha * np.array([0, 255, 0])) + (1 - mask_blend_alpha) * mask_underlay_rgb[obj_mask > 0]
mask_underlay_rgb = mask_underlay_rgb.astype(np.uint8)
plt.imshow(mask_underlay_rgb)
# plt.imshow(obj_mask, cmap='gray', alpha=0.5)
# plt.imshow(mask_underlay_rgb, alpha=0.5)
#%%
# save the figure
plt.imsave(dir_for_visualizations / "obj_mask_viz.png", mask_underlay_rgb)
# %%

theia_image_idx = 350
theia_underlay_rgb = cv2.imread(str(list_of_rgb_images[theia_image_idx]))
theia_underlay_rgb = cv2.cvtColor(theia_underlay_rgb, cv2.COLOR_BGR2RGB)
plt.imshow(theia_underlay_rgb)

#%%
theia_output_size = (int(original_height // patch_size), int(original_width // patch_size))
theia_transform = T.Compose([
    # T.ToPILImage(),
    # T.Resize(theia_resize, interpolation=T.InterpolationMode.BICUBIC),
    T.ToImage(),
])
color_image_for_theia = theia_transform(theia_underlay_rgb).to(device)
#%%
with torch.inference_mode():
    patch_features = theia_model.forward_feature(color_image_for_theia, do_resize=False, interpolate_pos_encoding=True)
    # patch_features = einops.rearrange(patch_features, 'b (h w) c -> b h w c', h=theia_output_size[0], w=theia_output_size[1])
#%%
pca = PCA(n_components=3)
pca_features = pca.fit_transform(patch_features[0].cpu().numpy())
# pca_features = pca.fit_transform(dinov2_output[0].cpu().numpy())
pca_features_rgb = (pca_features - np.min(pca_features)) / (np.max(pca_features) - np.min(pca_features))
pca_features_rgb = (pca_features_rgb * 255).astype(np.uint8)
pca_features_rgb = pca_features_rgb.reshape(*theia_output_size, 3)
# pca_features_rgb = pca_features_rgb.reshape(16, 16, 3)
# plt.imshow(pca_features_rgb)
#%%
theia_blend_alpha = 0.8
rescaled_pca_features_rgb = cv2.resize(pca_features_rgb, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
rescaled_pca_features_rgb = rescaled_pca_features_rgb[crop_lower_y:crop_upper_y, crop_lower_x:crop_upper_x]
rescaled_pca_features_rgb = (theia_blend_alpha * rescaled_pca_features_rgb) + (1 - theia_blend_alpha) * theia_underlay_rgb[crop_lower_y:crop_upper_y, crop_lower_x:crop_upper_x]
rescaled_pca_features_rgb = rescaled_pca_features_rgb.astype(np.uint8)
plt.imshow(rescaled_pca_features_rgb)
# plt.imshow(rescaled_pca_features_rgb, alpha=0.8)
# plt.imshow(theia_underlay_rgb[crop_lower_y:crop_upper_y, crop_lower_x:crop_upper_x], alpha=0.2)
# %%
# save the figure
plt.imsave(dir_for_visualizations / "theia_features_viz.png", rescaled_pca_features_rgb)
# %%
