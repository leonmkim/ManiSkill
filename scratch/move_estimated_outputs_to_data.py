#%%
from pathlib import Path
import zarr
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# %%

path_to_zarr = Path("/mnt/kostas-graid/datasets/extrinsic_contact_data/FISH/expert_demos/frankagym/FrankaInsertion-v1/1232_sim_w_recovery_demos_leftof4thbook_springbookends_graspedrand_noenvrand_noslotrand_20hz_act/demos.zarr")
zarr_store = zarr.open(path_to_zarr, mode='r+')
#%%
# gt_model_group_name = 'gt_segmentation'
# estimated_model_group_name = 'sam2-hiera-base-plus'
# mask_data_array_name = 'observation.EE_obj_mask'

#%%
# episode_idx = 639
# episode_start_idx = zarr_store['meta']['episode_ends'][episode_idx - 1] if episode_idx > 0 else 0
# episode_end_idx = zarr_store['meta']['episode_ends'][episode_idx]
# idx_within_episode = 120
# global_idx = episode_start_idx + idx_within_episode
# mask = zarr_store['data'][estimated_model_group_name][mask_data_array_name][global_idx]
# rgb_image = zarr_store['data']['observation.rgb'][global_idx]
# #%%
# plt.imshow(rgb_image)
# plt.imshow(mask, cmap='gray', alpha=0.5)
#%%
gt_model_group_name = 'gt_contact'
# estimated_model_group_name = 'contact_model_175604_2_epoch_9'
estimated_model_group_name = 'contact_model_197406_2_epoch_8'

contact_map_data_array_name = 'observation.contact_map'
EE_dtc_data_array_name = 'observation.EE_dtc_map'
env_dtc_data_array_name = 'observation.env_dtc_map'
EE_normals_data_array_name = 'observation.EE_normals_map'
env_normals_data_array_name = 'observation.env_normals_map'
data_array_names_list = [
    contact_map_data_array_name,
    EE_dtc_data_array_name,
    env_dtc_data_array_name,
    EE_normals_data_array_name,
    env_normals_data_array_name
]

# %%
episode_lengths = np.diff(np.hstack([np.array([0]), zarr_store['meta']['episode_ends'][:]]))
# %%
for i, episode_length in enumerate(episode_lengths):
    for data_array_name in data_array_names_list:
        # if data_array_name in zarr_store['episode_data'][f'episode_{i}'][estimated_model_group_name]:
        episode_data_array_length = zarr_store['episode_data'][f'episode_{i}'][estimated_model_group_name][data_array_name].shape[0]
        assert episode_data_array_length == episode_length, f"Episode {i} length {episode_length} does not match mask length {episode_data_array_length}."
print("All episode lengths and data array lengths match.")
# %%
for data_array_name in data_array_names_list:
    assert episode_lengths.sum() == zarr_store['data'][gt_model_group_name][data_array_name].shape[0], "Total episode length does not match total mask length."
#%%
# move all episode data to a new group under data
if estimated_model_group_name not in zarr_store['data']:
    zarr_store.create_group('data/' + estimated_model_group_name)
    print(f"Created group {estimated_model_group_name} in zarr dataset.")

for data_array_name in data_array_names_list:
    if data_array_name not in zarr_store['data'][estimated_model_group_name]:
        original_data_array_shape = zarr_store['episode_data']['episode_0'][estimated_model_group_name][data_array_name].shape[1:]
        original_data_array_dtype = zarr_store['episode_data']['episode_0'][estimated_model_group_name][data_array_name].dtype
        original_data_array_compressor = zarr_store['episode_data']['episode_0'][estimated_model_group_name][data_array_name].compressors[0]
        zarr_store.create_array('data/' + estimated_model_group_name + '/' + data_array_name, shape=(0, *original_data_array_shape), chunks=(1, *original_data_array_shape), dtype=original_data_array_dtype, compressor=original_data_array_compressor)

# %%
for data_array_name in data_array_names_list:
    for i in tqdm(range(len(episode_lengths))):
        if data_array_name in zarr_store['episode_data'][f'episode_{i}'][estimated_model_group_name]:
            # check if the array has already been added to the data group
            nominal_episode_ends = zarr_store['meta']['episode_ends'][i]
            current_data_array_length = zarr_store['data'][estimated_model_group_name][data_array_name].shape[0]
            if current_data_array_length < nominal_episode_ends:
                episode_data_array = zarr_store['episode_data'][f'episode_{i}'][estimated_model_group_name][data_array_name][:]
                zarr_store['data'][estimated_model_group_name][data_array_name].append(episode_data_array)
            else:
                print(f"Skipping episode {i} for data array {data_array_name} as it has already been added to the data group.")
            del zarr_store['episode_data'][f'episode_{i}'][estimated_model_group_name][data_array_name]

    assert zarr_store['data'][estimated_model_group_name][data_array_name].shape[0] == zarr_store['data'][gt_model_group_name][data_array_name].shape[0], "Total mask length does not match total ground truth mask length."

# %%
