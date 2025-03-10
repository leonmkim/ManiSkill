#%%
import zarr
import numpy as np
import os, sys
from pathlib import Path
from tqdm import tqdm
#%%
path_to_existing_zarr = Path('/mnt/crucialSSD/datasetsSSD/fish_datasets/simulated/teleop/FISH/expert_demos/frankagym/FrankaInsertion-v1/413_sim_demos_left_of_4th_book_20hz_act_sharded/413_sim_demos_left_of_4th_book_20hz_act/demos.zarr')
assert path_to_existing_zarr.exists()
# %%
zarr_root = zarr.open(str(path_to_existing_zarr), mode='w')
#%%
shard_factor = 10
# %%
dataset_shape = zarr_root.data.action.shape[1:]
dataset_chunks = zarr_root.data.action.chunks
shard_shape = (dataset_chunks[0] * shard_factor,) + dataset_chunks[1:]
dataset_compressor = zarr_root.data.action.compressor
dataset_dtype = zarr_root.data.action.dtype
#%%
# del zarr_root.data['action_sharded']
zarr_root.data.create_dataset('action_sharded', shape=(0,) + dataset_shape, shards=shard_shape, chunks=dataset_chunks, compressor=dataset_compressor, dtype=dataset_dtype, zarr_format=3)

# %%
# copy over the action data to the sharded dataset
for i in tqdm(range(zarr_root.data.action.shape[0])):
    zarr_root.data.action_sharded.append(zarr_root.data.action[i:i+1])

# %%
