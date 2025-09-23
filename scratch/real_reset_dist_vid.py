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
import imageio
#%%
path_to_root_episode_dir = Path("/mnt/crucialSSD/datasetsSSD/fish_datasets/real/")
dataset_name_list = [
    "cook_twodim_bookends_nominal",
    "modelsys_twodim_bookends_nominal",
    "lib_twodim_bookends_nominal",
    "fowlers_twodim_bookends_nominal",
    # "german_twodim_bookends_nominal",
    "hbm_twodim_bookends_nominal",
    # "lady_twodim_bookends_nominal",
    "greece_twodim_bookends_nominal",
]
dir_for_visualizations = Path("./real_reset_dist_vid")
dir_for_visualizations.mkdir(exist_ok=True, parents=True)
# dataset_name = "cook_twodim_bookends_nominal"
fps = 2
quality = 8
with imageio.get_writer(str(dir_for_visualizations / "real_reset_dist_2fps.mp4"), fps=fps, quality=quality) as writer:
    for dataset_name in dataset_name_list:
        path_to_episode_dir = path_to_root_episode_dir / dataset_name
        
        list_of_episodes = natsort.natsorted([x for x in path_to_episode_dir.iterdir() if x.is_dir()])
        path_to_episode_dir = list_of_episodes[0]

        path_to_episode_rgb_dir = path_to_episode_dir / "color"
        list_of_rgb_images = natsort.natsorted([x for x in path_to_episode_rgb_dir.iterdir() if x.suffix == ".png"])
        # print(f"Number of RGB images: {len(list_of_rgb_images)}")

        raw_rgb_frame = cv2.imread(str(list_of_rgb_images[0]))
        original_height, original_width = raw_rgb_frame.shape[0], raw_rgb_frame.shape[1]
        raw_rgb_frame = cv2.cvtColor(raw_rgb_frame, cv2.COLOR_BGR2RGB)
        # raw_rgb_frame = raw_rgb_frame[crop_lower_y:crop_upper_y, crop_lower_x:crop_upper_x]
        writer.append_data(raw_rgb_frame)
            # frames.append(raw_rgb_frame)
            # plt.imshow(raw_rgb_frame)
            # save the figure
            # plt.imsave(dir_for_visualizations / "raw_rgb_frame.png", raw_rgb_frame)

