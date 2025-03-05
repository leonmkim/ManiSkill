#%%
import h5py
from pathlib import Path
import numpy as np
import torch

import zarr

from mani_skill.utils.visualization.misc import images_to_video, tile_images

import cv2
import matplotlib.pyplot as plt

import sys, os
# add contact_estimation to the path
path_to_this_file = Path(os.path.abspath(__file__))
path_to_contact_estimation = path_to_this_file.parents[2] / "contact_estimation"
sys.path.append(str(path_to_contact_estimation))
from src.utils.viz_utils import normals_to_rgb_image, grasped_env_dtc_map_to_im, normals_masked_within_dtc_map, normalized_surface_normal_to_rgb

#%%
# path_to_demo = Path('/mnt/crucialSSD/datasetsSSD/fish_datasets/simulated/teleop/20250210_191246.h5')
# demo = h5py.File(path_to_demo, 'r')
# # %%
# traj = demo['traj_0']
# path_to_demo = Path('/mnt/crucialSSD/datasetsSSD/fish_datasets/simulated/teleop/20250214_072559.zarr')
path_to_demo = Path('/mnt/crucialSSD/datasetsSSD/fish_datasets/simulated/teleop/FISH/expert_demos/frankagym/FrankaInsertion-v1/413_sim_demos_left_of_4th_book_20hz_act/demos.zarr')

demo = zarr.open(path_to_demo, 'r')
dataset_name = path_to_demo.stem

#%%
episode_idx = 412
within_episode_idx = 50
episode_name = f"traj_{episode_idx}"
output_dir = path_to_demo.parent / dataset_name / episode_name

if not output_dir.exists():
    output_dir.mkdir(parents=True)
episode_start = 0
if episode_idx > 0:
    episode_start = demo.meta.episode_ends[episode_idx - 1]
episode_end = demo.meta.episode_ends[episode_idx]

rgb_image = demo.data['observation.rgb'][episode_start+within_episode_idx]
EE_dtc_map = demo.data.gt_contact['observation.EE_dtc_map'][episode_start+within_episode_idx]
env_dtc_map = demo.data.gt_contact['observation.env_dtc_map'][episode_start+within_episode_idx]
EE_normals_map = demo.data.gt_contact['observation.EE_normals_map'][episode_start+within_episode_idx]
env_normals_map = demo.data.gt_contact['observation.env_normals_map'][episode_start+within_episode_idx]
#%%

rgb_images_for_episode = demo.data['observation.rgb'][episode_start:episode_end]
# depth_images_for_episode = demo.data['observation.depth'][episode_start:episode_end]
# contact_images_for_episode = demo.data.gt_contact['observation.contact_map'][episode_start:episode_end]
# grasped_object_masks_for_episode = demo.data.gt_segmentation['observation.EE_obj_mask'][episode_start:episode_end]*255
EE_dtc_maps_for_episode = demo.data.gt_contact['observation.EE_dtc_map'][episode_start:episode_end]
env_dtc_maps_for_episode = demo.data.gt_contact['observation.env_dtc_map'][episode_start:episode_end]
EE_normals_maps_for_episode = demo.data.gt_contact['observation.EE_normals_map'][episode_start:episode_end]
env_normals_maps_for_episode = demo.data.gt_contact['observation.env_normals_map'][episode_start:episode_end]
#%%
images_to_video(
    images=rgb_images_for_episode,
    output_dir=str(output_dir),
    video_name='rgb_video',
    fps=20,
)
#%%
EE_env_dtc_map_images_for_episode = []
for (EE_dtc_map, env_dtc_map) in zip(EE_dtc_maps_for_episode, env_dtc_maps_for_episode):
    EE_env_dtc_map_image, _, _, _ = grasped_env_dtc_map_to_im(env_dtc_map[:,:,0], EE_dtc_map[:,:,0], sdf_viz_max_clip=0.3, sdf_viz_min_clip=0.0, return_BGR=False)
    EE_env_dtc_map_images_for_episode.append(EE_env_dtc_map_image)
#%%
images_to_video(
    images=EE_env_dtc_map_images_for_episode,
    output_dir=str(output_dir),
    video_name='EE_env_dtc_map_video',
    fps=20,
)
#%%
EE_normals_images_for_episode = []
for (EE_normals_map, EE_dtc_map) in zip(EE_normals_maps_for_episode, EE_dtc_maps_for_episode):
    EE_normals_masked_within_dtc_map, EE_dtc_map_mask = normals_masked_within_dtc_map(EE_normals_map, EE_dtc_map[:,:,0], dtc_threshold=0.3, adaptive_threshold=True, adaptive_threshold_buffer=0.1)
    EE_normals_viz_image = normalized_surface_normal_to_rgb(EE_normals_masked_within_dtc_map)
    EE_normals_images_for_episode.append(EE_normals_viz_image)
images_to_video(
    images=EE_normals_images_for_episode,
    output_dir=str(output_dir),
    video_name='EE_normals_video',
    fps=20,
)
#%%
env_normals_images_for_episode = []
for (env_normals_map, env_dtc_map) in zip(env_normals_maps_for_episode, env_dtc_maps_for_episode):
    env_normals_masked_within_dtc_map, env_dtc_map_mask = normals_masked_within_dtc_map(env_normals_map, env_dtc_map[:,:,0], dtc_threshold=0.3, adaptive_threshold=True, adaptive_threshold_buffer=0.1)
    env_normals_viz_image = normalized_surface_normal_to_rgb(env_normals_masked_within_dtc_map)
    env_normals_images_for_episode.append(env_normals_viz_image)
images_to_video(
    images=env_normals_images_for_episode,
    output_dir=str(output_dir),
    video_name='env_normals_video',
    fps=20,
)
#%%
images_to_video(
    images=grasped_object_masks_for_episode,
    output_dir=str(output_dir),
    video_name='grasped_obj_mask_video',
    fps=20,
)
#%%
images_to_video(
    images=rgb_images_for_episode,
    output_dir=str(output_dir),
    video_name='rgb_video',
    fps=20,
)
#%%
contact_overlay_images = []
for (rgb_frame, contact_frame) in zip(rgb_images_for_episode, contact_images_for_episode):
    contact_overlay_image = rgb_frame
    # contact_bool_mask = contact_frame == 1
    contact_pixel_coords = np.argwhere(contact_frame == 1)[:, :2]
    # flip indices to match the order of the image dimensions
    contact_pixel_coords = contact_pixel_coords[:, ::-1]
    for contact_pixel_coord in contact_pixel_coords:
        cv2.circle(contact_overlay_image, tuple(contact_pixel_coord), 2, (255, 0, 0), -1)

    # # repeat the boolean mask 3 times to match the number of channels in the RGB image
    # contact_bool_mask = np.repeat(contact_bool_mask, 3, axis=2)
    # # add false to the channel dimension to match the number of channels in the RGB image
    # contact_bool_mask[:, :, 1:3] = False
    # if np.any(contact_bool_mask):
    #     contact_overlay_image[contact_bool_mask] = 255
    contact_overlay_images.append(contact_overlay_image)

images_to_video(
    images=contact_overlay_images,
    output_dir=str(output_dir),
    video_name='contact_overlay_video',
    fps=20,
)
#%%
import matplotlib.pyplot as pltK
plt.imshow(contact_bool_mask.astype(np.uint8)*255)
# %%
path_to_zarr = Path('~/fish_leon/FISH/expert_demos/frankagym/FrankaInsertion-v1/120_240x320_all_twodim_left_to_right_annotated_start_idx_5hz_zstd7_EE_pxl_coords_expert_demos_imp_act/demos.zarr')
path_to_zarr = path_to_zarr.expanduser()
zarr_dataset = zarr.open(path_to_zarr, 'r')
# %%
plt.imshow(zarr_dataset.data['observation.EE_obj_mask'][0])

# %%
