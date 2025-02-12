#%%
import h5py
from pathlib import Path
import numpy as np
import torch

import zarr

from mani_skill.utils.visualization.misc import images_to_video, tile_images

import cv2
#%%
# path_to_demo = Path('/mnt/crucialSSD/datasetsSSD/fish_datasets/simulated/teleop/20250210_191246.h5')
# demo = h5py.File(path_to_demo, 'r')
# # %%
# traj = demo['traj_0']
path_to_demo = Path('/mnt/crucialSSD/datasetsSSD/fish_datasets/simulated/teleop/20250212_130205.zarr')
demo = zarr.open(path_to_demo, 'r')
dataset_name = path_to_demo.stem

#%%
episode_idx = 4
episode_name = f"traj_{episode_idx}"
output_dir = path_to_demo.parent / dataset_name / episode_name

if not output_dir.exists():
    output_dir.mkdir(parents=True)
episode_start = 0
if episode_idx > 0:
    episode_start = demo.meta.episode_ends[episode_idx - 1]
episode_end = demo.meta.episode_ends[episode_idx]
rgb_images_for_episode = demo.data['observation.rgb'][episode_start:episode_end]
depth_images_for_episode = demo.data['observation.depth'][episode_start:episode_end]
contact_images_for_episode = demo.data['observation.contact_map'][episode_start:episode_end]
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
import matplotlib.pyplot as plt
plt.imshow(contact_bool_mask.astype(np.uint8)*255)
# %%
path_to_zarr = Path('~/fish_leon/FISH/expert_demos/frankagym/FrankaInsertion-v1/120_240x320_all_twodim_left_to_right_annotated_start_idx_5hz_zstd7_EE_pxl_coords_expert_demos_imp_act/demos.zarr')
path_to_zarr = path_to_zarr.expanduser()
zarr_dataset = zarr.open(path_to_zarr, 'r')
# %%
