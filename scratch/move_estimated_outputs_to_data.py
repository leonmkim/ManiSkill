#%%
from pathlib import Path
import zarr
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
#%%
# path_to_file = Path(__file__).parents[2] / 'episodes_with_missing_data.txt'
# # read the txt file with episode idxs with missing data
# with open(path_to_file, 'r') as f:
#     episode_indices_missing_data = [int(line.strip()) for line in f.readlines()]
#     print(f"Found {len(episode_indices_missing_data)} episodes with missing data.")
# #%%
# # extract unique indices from the list
# unique_episode_indices_missing_data = list(set(episode_indices_missing_data))
# print(f"Found {len(unique_episode_indices_missing_data)} unique episodes with missing data.")
#%%
# # convert this list to a slurm sbatch array string
# # for contiguous indices, use the format 0-10
# # for non-contiguous indices, use the format 0,1,2,3,
# slurm_array_string = ''
# if len(unique_episode_indices_missing_data) == 0:
#     print("No episodes with missing data found.")
# else:
#     unique_episode_indices_missing_data.sort()
#     start_idx = unique_episode_indices_missing_data[0]
#     end_idx = start_idx
#     for idx in unique_episode_indices_missing_data[1:]:
#         if idx == end_idx + 1:
#             end_idx = idx
#         else:
#             if start_idx == end_idx:
#                 slurm_array_string += f"{start_idx},"
#             else:
#                 slurm_array_string += f"{start_idx}-{end_idx},"
#             start_idx = idx
#             end_idx = start_idx
#     if start_idx == end_idx:
#         slurm_array_string += f"{start_idx}"
#     else:
#         slurm_array_string += f"{start_idx}-{end_idx}"
# %%

path_to_zarr = Path("/mnt/kostas-graid/datasets/extrinsic_contact_data/FISH/expert_demos/frankagym/FrankaInsertion-v1/1232_sim_w_recovery_demos_leftof4thbook_springbookends_graspedrand_noenvrand_noslotrand_20hz_act/demos.zarr")
# path_to_zarr = Path("/mnt/crucialSSD/datasetsSSD/fish_datasets/simulated/teleop/FISH/expert_demos/frankagym/FrankaInsertion-v1/2_demo_test/demos.zarr")
zarr_store = zarr.open(path_to_zarr, mode='r+')
#%%
episode_lengths = np.diff(np.hstack([np.array([0]), zarr_store['meta']['episode_ends'][:]]))
max_demo_length = episode_lengths.max()
#%%
zarr_store['meta'].attrs['max_demo_length'] = max_demo_length
#%%
# gt_model_group_name = 'gt_segmentation'
# estimated_model_group_name = 'sam2-hiera-base-plus'
# mask_data_array_name = 'observation.EE_obj_mask'
#%%
estimated_model_group_name = 'theia-base-patch16-224-cdiv_15x20'
x_norm_patchtokens_array_name = 'observation.x_norm_patchtokens'
# x_norm_clstoken_array_name = 'observation.x_norm_clstoken'
# x_norm_regtokens_array_name = 'observation.x_norm_regtokens'
data_array_names_list = [
    # mask_data_array_name,
    x_norm_patchtokens_array_name,
    # x_norm_clstoken_array_name,
    # x_norm_regtokens_array_name
]

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
# gt_model_group_name = 'gt_contact'
# estimated_model_group_name = 'contact_model_175604_2_epoch_9'
# # estimated_model_group_name = 'contact_model_197406_2_epoch_8'

# contact_map_data_array_name = 'observation.contact_map'
# EE_dtc_data_array_name = 'observation.EE_dtc_map'
# env_dtc_data_array_name = 'observation.env_dtc_map'
# EE_normals_data_array_name = 'observation.EE_normals_map'
# env_normals_data_array_name = 'observation.env_normals_map'
# data_array_names_list = [
#     contact_map_data_array_name,
#     EE_dtc_data_array_name,
#     env_dtc_data_array_name,
#     EE_normals_data_array_name,
#     env_normals_data_array_name
# ]

# %%
episode_lengths = np.diff(np.hstack([np.array([0]), zarr_store['meta']['episode_ends'][:]]))
# %%
episode_indices_missing_data = []
all_episodes_complete = False
for episode_idx, episode_length in enumerate(episode_lengths):
    episode_first_data_array_length = None
    for data_array_name in data_array_names_list:
        # if data_array_name in zarr_store['episode_data'][f'episode_{i}'][estimated_model_group_name]:
        try:
            episode_data_array_length = zarr_store['episode_data'][f'episode_{episode_idx}'][estimated_model_group_name][data_array_name].shape[0]
        except:
            print(f"Episode {episode_idx} does not have data array {data_array_name}.")
            episode_indices_missing_data.append(episode_idx)
            break
        if episode_first_data_array_length is None:
            episode_first_data_array_length = episode_data_array_length
        if episode_data_array_length != episode_first_data_array_length:
            print(f"Episode {episode_idx} data array {data_array_name} length {episode_data_array_length} does not match first data array length {episode_first_data_array_length}.")
            break

        # assert episode_data_array_length == episode_first_data_array_length, f"Episode {episode_idx} data array {data_array_name} length {episode_data_array_length} does not match first data array length {episode_first_data_array_length}."
        # assert episode_data_array_length == episode_length, f"Episode {episode_idx} length {episode_length} does not match mask length {episode_data_array_length}."
    if episode_length != episode_first_data_array_length:
        episode_indices_missing_data.append(episode_idx)
        print(f"Episode {episode_idx} has missing data: length {episode_length}, first data array length {episode_first_data_array_length}.")
if len(episode_indices_missing_data) > 0:
    print(f"Found {len(episode_indices_missing_data)} episodes with missing data: {episode_indices_missing_data}")
    # output list of indices of episodes with missing data to a text file
    with open('episodes_with_missing_data.txt', 'w') as f:
        for episode_idx in episode_indices_missing_data:
            f.write(f"{episode_idx}\n")
else:
    print("All episodes have complete data for all data arrays.")
    all_episodes_complete = True
if not all_episodes_complete:
    print("Not all episodes have complete data for all data arrays. Exiting.")
    exit(1)
# %%
# move all episode data to a new group under data
if estimated_model_group_name not in zarr_store['data']:
    zarr_store.create_group('data/' + estimated_model_group_name)
    print(f"Created group {estimated_model_group_name} in zarr dataset.")

for data_array_name in data_array_names_list:
    if data_array_name not in zarr_store['data'][estimated_model_group_name]:
        original_data_array_shape = zarr_store['episode_data']['episode_0'][estimated_model_group_name][data_array_name].shape[1:]
        original_data_array_dtype = zarr_store['episode_data']['episode_0'][estimated_model_group_name][data_array_name].dtype
        original_data_array_compressors = zarr_store['episode_data']['episode_0'][estimated_model_group_name][data_array_name].compressors
        zarr_store.create_array('data/' + estimated_model_group_name + '/' + data_array_name, shape=(0, *original_data_array_shape), chunks=(1, *original_data_array_shape), dtype=original_data_array_dtype, compressors=original_data_array_compressors)

# %%
for data_array_name in data_array_names_list:
    # for episode_idx in tqdm(range(len(episode_lengths))):
    for episode_idx, episode_length in tqdm(enumerate(episode_lengths)):
        if data_array_name in zarr_store['episode_data'][f'episode_{episode_idx}'][estimated_model_group_name]:
            # check if the array has already been added to the data group
            nominal_episode_ends = zarr_store['meta']['episode_ends'][episode_idx]
            current_data_array_length = zarr_store['data'][estimated_model_group_name][data_array_name].shape[0]
            if current_data_array_length < nominal_episode_ends:
                episode_data_array = zarr_store['episode_data'][f'episode_{episode_idx}'][estimated_model_group_name][data_array_name][:]
                zarr_store['data'][estimated_model_group_name][data_array_name].append(episode_data_array)
            else:
                print(f"Skipping episode {episode_idx} for data array {data_array_name} as it has already been added to the data group.")
            del zarr_store['episode_data'][f'episode_{episode_idx}'][estimated_model_group_name][data_array_name]

    assert zarr_store['data'][estimated_model_group_name][data_array_name].shape[0] == zarr_store['meta']['episode_ends'][-1], "Total mask length does not match total ground truth mask length."

# %%
