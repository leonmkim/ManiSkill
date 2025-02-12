#%%
import h5py
from pathlib import Path
import numpy as np
import torch

import zarr
#%%
# path_to_demo = Path('/mnt/crucialSSD/datasetsSSD/fish_datasets/simulated/teleop/20250210_191246.h5')
# demo = h5py.File(path_to_demo, 'r')
# # %%
# traj = demo['traj_0']
path_to_demo = Path('/mnt/crucialSSD/datasetsSSD/fish_datasets/simulated/teleop/20250212_124946.zarr')
demo = zarr.open(path_to_demo, 'r')

# %%
path_to_zarr = Path('~/fish_leon/FISH/expert_demos/frankagym/FrankaInsertion-v1/120_240x320_all_twodim_left_to_right_annotated_start_idx_5hz_zstd7_EE_pxl_coords_expert_demos_imp_act/demos.zarr')
path_to_zarr = path_to_zarr.expanduser()
zarr_dataset = zarr.open(path_to_zarr, 'r')
# %%
