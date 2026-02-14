#%%
from pathlib import Path
import zarr
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# %%
path_to_zarr = Path("/mnt/kostas-graid/datasets/extrinsic_contact_data/FISH/expert_demos/frankagym/FrankaInsertion-v1/1232_sim_w_recovery_demos_leftof4thbook_springbookends_graspedrand_noenvrand_noslotrand_20hz_act/demos.zarr")
# path_to_zarr = Path("/mnt/kostas-graid/datasets/extrinsic_contact_data/FISH/expert_demos/frankagym/FrankaInsertion-v1/638_sim_nominal_demos_peginsertion_20hz_act/demos.zarr")
# path_to_zarr = Path("/mnt/kostas-graid/datasets/extrinsic_contact_data/FISH/expert_demos/frankagym/FrankaInsertion-v1/269_sim_recovery_demos_peginsertion_20hz_act/demos.zarr")
# path_to_zarr = Path("/mnt/kostas-graid/datasets/extrinsic_contact_data/FISH/expert_demos/frankagym/FrankaInsertion-v1/907_sim_all_demos_peginsertion_20hz_act/demos.zarr")
# path_to_zarr = Path("/mnt/crucialSSD/datasetsSSD/fish_datasets/simulated/teleop/FISH/expert_demos/frankagym/FrankaInsertion-v1/2_demo_test/demos.zarr")
# path_to_zarr = Path("/mnt/12_tb_hdd/fish_contact_backup/FISH/expert_demos/frankagym/FrankaInsertion-v1/907_sim_all_demos_peginsertion_20hz_act/demos.zarr")
assert path_to_zarr.exists(), f"could not find zarr at {path_to_zarr}"
zarr_store = zarr.open(path_to_zarr, mode='r+')
#%%
# idx= 50
# max_contact_prob = 0.05
# rerender_group = 'rerendered_-30.0_deg_rotation'
# # rerender_group = 'rerendered_-60.0_deg_rotation'
# # episode_0_estimated_masks = zarr_store['episode_data']['episode_0']['rerendered_-30.0_deg_rotation']['sam2-hiera-base-plus']['observation.EE_obj_mask'][idx]
# estimated_contact_map = zarr_store['episode_data']['episode_0'][rerender_group]['contact_model_175604_2_epoch_9']['observation.contact_map'][idx]
# estimated_contact_map = (np.clip(estimated_contact_map, a_min=0.0, a_max=max_contact_prob)*(255/max_contact_prob)).astype(np.uint8)
# episode_0_rgb = zarr_store['data'][rerender_group]['observation.rgb'][idx]
# #%%
# plt.imshow(episode_0_rgb, alpha=0.5)
# # plt.imshow(episode_0_estimated_masks,alpha=0.5)
# plt.imshow(estimated_contact_map, alpha=0.5, cmap='jet')

#%%
def format_slurm_array_string(episode_indices):
    episode_indices.sort()
    slurm_array_string = ''
    start_idx = episode_indices[0]
    end_idx = start_idx
    for idx in episode_indices[1:]:
        if idx == end_idx + 1:
            end_idx = idx
        else:
            if start_idx == end_idx:
                slurm_array_string += f"{start_idx},"
            else:
                slurm_array_string += f"{start_idx}-{end_idx},"
            start_idx = idx
            end_idx = start_idx
    if start_idx == end_idx:
        slurm_array_string += f"{start_idx}"
    else:
        slurm_array_string += f"{start_idx}-{end_idx}"
    return slurm_array_string
#%%
dry_run=False
cam_extrinsic_rotation_angle_deg = float(-60)
list_of_arrays_to_check = list()
# list_of_arrays_to_check.append(
#     dict(
#         root_path='',
#         array_names=[
#             'observation.rgb',
#             'observation.depth',
#             'observation.EE_pixel_coord',
#         ]
#     )
# )
# list_of_arrays_to_check.append(
#     dict(
#         root_path='gt_contact',
#         array_names=[
#             # 'observation.env_dtc_map',
#             # 'observation.env_normals_map',
#             # 'observation.EE_dtc_map',
#             # 'observation.EE_normals_map',
#             'observation.contact_map'
#         ]
#     )
# )
list_of_arrays_to_check.append(
    dict(
        root_path='contact_model_175604_2_epoch_9',
        array_names=[
            'observation.env_dtc_map',
            'observation.env_normals_map',
            'observation.EE_dtc_map',
            'observation.EE_normals_map',
            'observation.contact_map'
        ]
    )
)
list_of_arrays_to_check.append(
    dict(
        root_path='contact_model_197406_2_epoch_8',
        array_names=[
            'observation.env_dtc_map',
            'observation.env_normals_map',
            'observation.EE_dtc_map',
            'observation.EE_normals_map',
            'observation.contact_map'
        ]
    )
)
# list_of_arrays_to_check.append(
#     dict(
#         root_path='gt_segmentation',
#         array_names=[
#             'observation.EE_obj_mask',
#             'observation.segmentation',
#         ]
#     )
# )
list_of_arrays_to_check.append(
    dict(
        root_path='sam2-hiera-base-plus',
        array_names=[
            'observation.EE_obj_mask',
        ]
    )
)
#%%
episode_lengths = np.diff(np.hstack([np.array([0]), zarr_store['meta']['episode_ends'][:]]))
# %%
# episode_indices_missing_data = []
episode_indices_missing_data = dict()
all_episodes_complete = False
for episode_idx, episode_length in enumerate(episode_lengths):
    # traverse the list of arrays to check and verify that each array exists for the current episode and has the correct length
    episode_path = f'episode_data/episode_{episode_idx}'
    if cam_extrinsic_rotation_angle_deg != 0:
        episode_path += f"/rerendered_{cam_extrinsic_rotation_angle_deg}_deg_rotation"
        
    for array_info in list_of_arrays_to_check:
        root_path = array_info['root_path']
        for array_name in array_info['array_names']:
            array_path = array_name if root_path == '' else root_path + '/' + array_name

            # if array_path not in zarr_store['episode_data'][f'episode_{episode_idx}']:
            if array_path not in zarr_store[episode_path]:
                print(f"Episode {episode_idx} is missing array {array_name} in group {root_path}.")
                # episode_indices_missing_data.append(episode_idx)
                if episode_idx not in episode_indices_missing_data:
                    episode_indices_missing_data[episode_idx] = []
                episode_indices_missing_data[episode_idx].append(array_path)
            else:
                episode_data_array_length = zarr_store[episode_path][array_path].shape[0]
                if episode_data_array_length != episode_length:
                    print(f"Episode {episode_idx} array {array_name} in group {root_path} has length {episode_data_array_length} but expected length is {episode_length}.")
                    # episode_indices_missing_data.append(episode_idx)
                    if episode_idx not in episode_indices_missing_data:
                        episode_indices_missing_data[episode_idx] = []
                    episode_indices_missing_data[episode_idx].append(array_path)
#%%
if len(episode_indices_missing_data) > 0:
    print(f"Found {len(episode_indices_missing_data)} episodes with missing data: {episode_indices_missing_data}")
    # reformat into categories of missing data
    missing_base_group_episodes = list()
    base_group_arrays = [
            'observation.rgb',
            'observation.depth',
            'observation.EE_pixel_coord',
            'gt_segmentation/observation.EE_obj_mask',
            'gt_segmentation/observation.segmentation',
            ]
    missing_contact_group_episodes = list()
    contact_group_arrays = [
            'gt_contact/observation.env_dtc_map',
            'gt_contact/observation.env_normals_map',
            'gt_contact/observation.EE_dtc_map',
            'gt_contact/observation.EE_normals_map',
            'gt_contact/observation.contact_map'
            ]
    for episode_idx, missing_arrays in episode_indices_missing_data.items():
        for missing_array in missing_arrays:
            if missing_array in base_group_arrays:
                missing_base_group_episodes.append(episode_idx)
            elif missing_array in contact_group_arrays:
                missing_contact_group_episodes.append(episode_idx)

    # extract unique episode indices for each category
    missing_base_group_episodes = list(set(missing_base_group_episodes))
    missing_contact_group_episodes = list(set(missing_contact_group_episodes))
    print(f"Found {len(missing_base_group_episodes)} episodes with missing data in base group: {missing_base_group_episodes}")
    # format into slurm compatible array string
    
    if len(missing_base_group_episodes) > 0:
        missing_base_group_slurm_array_string = format_slurm_array_string(missing_base_group_episodes)
        missing_base_group_filename = 'episodes_with_missing_base_group_data'
        if cam_extrinsic_rotation_angle_deg != 0:
            missing_base_group_filename += f"_rerendered_{cam_extrinsic_rotation_angle_deg}_deg_rotation"
        path_to_missing_base_group_file = missing_base_group_filename + '.txt'
        with open(path_to_missing_base_group_file, 'w') as f:
            f.write(missing_base_group_slurm_array_string)
    if len(missing_contact_group_episodes) > 0:
        missing_contact_group_slurm_array_string = format_slurm_array_string(missing_contact_group_episodes)
        # save these slurm array strings to text files
        missing_contact_group_filename = 'episodes_with_missing_contact_group_data'
        if cam_extrinsic_rotation_angle_deg != 0:
            missing_contact_group_filename += f"_rerendered_{cam_extrinsic_rotation_angle_deg}_deg_rotation"
        path_to_missing_contact_group_file = missing_contact_group_filename + '.txt'
        with open(path_to_missing_contact_group_file, 'w') as f:
            f.write(missing_contact_group_slurm_array_string)
else:
    print("All episodes have complete data for all data arrays.")
    all_episodes_complete = True
if not all_episodes_complete:
    print("Not all episodes have complete data for all data arrays. Exiting.")
    exit(1)
# %%
# # move all episode data to a new group under data
zarr_store_data = zarr_store['data']
zarr_store_meta = zarr_store['meta']

first_episode_path = f'episode_data/episode_0'
if cam_extrinsic_rotation_angle_deg != 0:
    first_episode_path += f"/rerendered_{cam_extrinsic_rotation_angle_deg}_deg_rotation"
    if f'rerendered_{cam_extrinsic_rotation_angle_deg}_deg_rotation' not in zarr_store_data:
        zarr_store_data.create_group(f'rerendered_{cam_extrinsic_rotation_angle_deg}_deg_rotation')
    zarr_store_data = zarr_store_data[f'rerendered_{cam_extrinsic_rotation_angle_deg}_deg_rotation']
    
    if f'rerendered_{cam_extrinsic_rotation_angle_deg}_deg_rotation' not in zarr_store_meta:
        zarr_store_meta.create_group(f'rerendered_{cam_extrinsic_rotation_angle_deg}_deg_rotation')
    zarr_store_meta = zarr_store_meta[f'rerendered_{cam_extrinsic_rotation_angle_deg}_deg_rotation']
    if 'episode_cam_tf_world' not in zarr_store_meta:
        zarr_store_meta.create_array('episode_cam_tf_world', shape=(0, 4, 4), chunks=(1, 4, 4), dtype=np.float32, overwrite=True)

for array_info in list_of_arrays_to_check:
    root_path = array_info['root_path']
    for array_name in array_info['array_names']:
        array_path = array_name if root_path == '' else root_path + '/' + array_name
        # make sure all arrays are created under data group with the correct shape, dtype, and compressors
        if array_path not in zarr_store_data:
            original_data_array_shape = zarr_store[f'{first_episode_path}/{array_path}'].shape[1:]
            original_data_array_dtype = zarr_store[f'{first_episode_path}/{array_path}'].dtype
            original_data_array_compressors = zarr_store[f'{first_episode_path}/{array_path}'].compressors
            zarr_store_data.create_array(array_path, shape=(0, *original_data_array_shape), chunks=(1, *original_data_array_shape), dtype=original_data_array_dtype, compressors=original_data_array_compressors, overwrite=True)

        for episode_idx, episode_length in tqdm(enumerate(episode_lengths), desc=f"Processing episodes for array {array_path}", position=0, leave=True):
            episode_path = f'episode_data/episode_{episode_idx}'
            if cam_extrinsic_rotation_angle_deg != 0:
                episode_path += f"/rerendered_{cam_extrinsic_rotation_angle_deg}_deg_rotation"
            episode_data_array = zarr_store[episode_path][array_path][:]

            print(f"adding episode {episode_path} data for array {array_path} to data group. episode data array shape: {episode_data_array.shape}")
            if not dry_run:
                zarr_store_data[array_path].append(episode_data_array)
                del zarr_store[episode_path][array_path]

# array_path = 'episode_cam_tf_world'
# for episode_idx, episode_length in tqdm(enumerate(episode_lengths), desc=f"Processing episodes for array {array_path}", position=0, leave=True):
#     episode_path = f'episode_data/episode_{episode_idx}'
#     if cam_extrinsic_rotation_angle_deg != 0:
#         episode_path += f"/rerendered_{cam_extrinsic_rotation_angle_deg}_deg_rotation"
#     episode_data_array = zarr_store[episode_path][array_path][:]
#     print(f"adding episode {episode_path} data for array {array_path} to data group. episode data array shape: {episode_data_array.shape}")
#     if not dry_run:
#         zarr_store_meta[array_path].append(episode_data_array)
#         del zarr_store[episode_path][array_path]

# if estimated_model_group_name not in zarr_store['data']:
#     zarr_store.create_group('data/' + estimated_model_group_name)
#     print(f"Created group {estimated_model_group_name} in zarr dataset.")

# for data_array_name in data_array_names_list:
#     if data_array_name not in zarr_store['data'][estimated_model_group_name]:
#         original_data_array_shape = zarr_store['episode_data']['episode_0'][estimated_model_group_name][data_array_name].shape[1:]
#         original_data_array_dtype = zarr_store['episode_data']['episode_0'][estimated_model_group_name][data_array_name].dtype
#         original_data_array_compressors = zarr_store['episode_data']['episode_0'][estimated_model_group_name][data_array_name].compressors
#         zarr_store.create_array('data/' + estimated_model_group_name + '/' + data_array_name, shape=(0, *original_data_array_shape), chunks=(1, *original_data_array_shape), dtype=original_data_array_dtype, compressors=original_data_array_compressors)

# # %%
# for data_array_name in data_array_names_list:
#     # for episode_idx in tqdm(range(len(episode_lengths))):
#     for episode_idx, episode_length in tqdm(enumerate(episode_lengths)):
#         if data_array_name in zarr_store['episode_data'][f'episode_{episode_idx}'][estimated_model_group_name]:
#             # check if the array has already been added to the data group
#             nominal_episode_ends = zarr_store['meta']['episode_ends'][episode_idx]
#             current_data_array_length = zarr_store['data'][estimated_model_group_name][data_array_name].shape[0]
#             if current_data_array_length < nominal_episode_ends:
#                 episode_data_array = zarr_store['episode_data'][f'episode_{episode_idx}'][estimated_model_group_name][data_array_name][:]
#                 zarr_store['data'][estimated_model_group_name][data_array_name].append(episode_data_array)
#             else:
#                 print(f"Skipping episode {episode_idx} for data array {data_array_name} as it has already been added to the data group.")
#             del zarr_store['episode_data'][f'episode_{episode_idx}'][estimated_model_group_name][data_array_name]

#     assert zarr_store['data'][estimated_model_group_name][data_array_name].shape[0] == zarr_store['meta']['episode_ends'][-1], "Total mask length does not match total ground truth mask length."

# # %%
