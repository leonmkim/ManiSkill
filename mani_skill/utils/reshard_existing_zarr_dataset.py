#%%
import zarr
import numpy as np
import os, sys
from pathlib import Path
from tqdm import tqdm
import numcodecs
#%%
path_to_existing_zarr = Path('/mnt/crucialSSD/datasetsSSD/fish_datasets/simulated/teleop/FISH/expert_demos/frankagym/FrankaInsertion-v1/413_sim_demos_left_of_4th_book_20hz_act_sharded/413_sim_demos_left_of_4th_book_20hz_act/demos.zarr')
assert path_to_existing_zarr.exists()
# %%
original_zarr_root = zarr.open(str(path_to_existing_zarr), mode='r')
new_zarr_path = Path(str(path_to_existing_zarr).replace('.zarr', '_resharded.zarr'))
assert new_zarr_path.exists()
print(f"new zarr path: {new_zarr_path}")
new_zarr_root = zarr.open(str(new_zarr_path), mode='r', zarr_format=3)
# #%%
# new_zarr_root.create_group('data')
# #%%
# new_zarr_root.create_group('meta')
# #%%

# #%%
# shard_factor = 10
# dataset_compressor = zarr.codecs.BloscCodec(cname='zstd', clevel=3, shuffle=zarr.codecs.BloscShuffle.shuffle)
# #%%
# def recursive_copy_and_reshard_arrays(original_group, new_group, shard_factor=None, dataset_compressor=zarr.codecs.BloscCodec(cname='zstd', clevel=3, shuffle=zarr.codecs.BloscShuffle.shuffle)):
#     for key in original_group.keys():
#         if isinstance(original_group[key], zarr.Group):
#             if key not in new_group:
#                 new_group.create_group(key)
#                 # copy over attributes
#                 for attr_key in original_group[key].attrs.keys():
#                     new_group[key].attrs[attr_key] = original_group[key].attrs[attr_key]
#                 recursive_copy_and_reshard_arrays(original_group[key], new_group[key])
#         elif isinstance(original_group[key], zarr.Array):
#             if key not in new_group:
#                 dataset_shape = original_group[key].shape[1:]
#                 dataset_chunks = original_group[key].chunks
#                 if shard_factor is None:
#                     shard_shape = None
#                 else:
#                     shard_shape = (dataset_chunks[0] * shard_factor,) + dataset_chunks[1:]
#                 dataset_dtype = original_group[key].dtype
#                 new_group.create_array(key, shape=(0,) + dataset_shape, shards=shard_shape, chunks=dataset_chunks, compressor=dataset_compressor, dtype=dataset_dtype)
#                 for i in tqdm(range(original_group[key].shape[0])):
#                     new_group[key].append(original_group[key][i:i+1])
#                 # copy over attributes
#                 for attr_key in original_group[key].attrs.keys():
#                     new_group[key].attrs[attr_key] = original_group[key].attrs[attr_key]
# #%%
# recursive_copy_and_reshard_arrays(original_zarr_root['data'], new_zarr_root['data'], shard_factor=shard_factor, dataset_compressor=dataset_compressor)

# # %%
# # then recursively copy over data under 'meta' without sharding
# recursive_copy_and_reshard_arrays(original_zarr_root['meta'], new_zarr_root['meta'], shard_factor=None, dataset_compressor=dataset_compressor)

# # %%
# del new_zarr_root['metadata']
# %%
# check that the resharded zarr is correct
def recursive_check_arrays(original_group, new_group):
    for key in tqdm(original_group.keys()):
        if isinstance(original_group[key], zarr.Group):
            # also check attributes
            assert original_group[key].attrs == new_group[key].attrs
            recursive_check_arrays(original_group[key], new_group[key])
        elif isinstance(original_group[key], zarr.Array):
            # also check attributes
            assert original_group[key].attrs == new_group[key].attrs
            assert original_group[key].shape == new_group[key].shape
            # assert original_group[key].chunks == new_group[key].chunks
            assert original_group[key].dtype == new_group[key].dtype
            # assert np.allclose(original_group[key][:], new_group[key][:])
            for i in tqdm(range(original_group[key].shape[0])):
                assert np.allclose(original_group[key][i], new_group[key][i]), f"array not the same for key {key} at index {i} with original value {original_group[key][i]} and new value {new_group[key][i]}"
recursive_check_arrays(original_zarr_root['data'], new_zarr_root['data'])
recursive_check_arrays(original_zarr_root['meta'], new_zarr_root['meta'])
# %%
