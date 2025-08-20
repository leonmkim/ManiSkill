#%%
from deepdiff import DeepDiff
import zarr
ZARR_VERSION=int(zarr.__version__.split('.')[0])
if ZARR_VERSION < 3:
    from zarr.hierarchy import Group as ZarrGroup
    from zarr.core import Array as ZarrArray
else:
    from zarr import Group as ZarrGroup
    from zarr import Array as ZarrArray
from pathlib import Path

import numpy as np

import json
import tqdm
# import matplotlib.pyplot as plt

import shutil

# from mani_skill.utils.visualization.misc import images_to_video, tile_images

from tqdm import tqdm

import logging
logging.basicConfig(
    level=logging.INFO,)

post_process_logger = logging.getLogger("post_process_logger")
# post_process_logger.setLevel(logging.INFO)

#%%
def recursive_append_new_demo_data(base_demo, new_demo, copy_over_n_chunks_at_a_time: int = 500):
    for key in base_demo.keys():
        if isinstance(base_demo[key], ZarrArray):
            # base_demo[key].append(new_demo[key][:])
            # need to copy over n chunks at a time because the array is too large
            for i in tqdm(range(0, new_demo[key].shape[0], copy_over_n_chunks_at_a_time), desc=f"Appending {key}", unit="chunk"):
                start = i
                end = min(i + copy_over_n_chunks_at_a_time, new_demo[key].shape[0])
                base_demo[key].append(new_demo[key][start:end])
        elif isinstance(base_demo[key], ZarrGroup):
            if key == 'episode_data': # ignore episode data
                continue
            recursive_append_new_demo_data(base_demo[key], new_demo[key])

def recursive_trim_trimmed_arrays(demo_data: ZarrGroup, 
                                  new_episode_start: int, new_episode_end: int, 
                                  new_untrimmed_episode_start: int, new_untrimmed_episode_end: int, 
                                  pretrimmed: bool = True):
    # pretrimmed refers to fact that the non state arrays under actors/articulations groups have been trimmed to exclude one frame per episode (first frame for action, and last frame for observations)
    for key in demo_data.keys():
        if isinstance(demo_data[key], ZarrArray) and not key.endswith('_tmp') and not key == 'start':
            if f'{key}_tmp' not in demo_data:
                demo_data.create_array(f'{key}_tmp', shape=(0, *demo_data[key].shape[1:]), dtype=demo_data[key].dtype, chunks=(1, *demo_data[key].shape[1:]), compressors=demo_data[key].compressors)
            if pretrimmed:
                demo_data[f'{key}_tmp'].append(demo_data[key][new_episode_start:new_episode_end])
            else:
                demo_data[f'{key}_tmp'].append(demo_data[key][new_untrimmed_episode_start:new_untrimmed_episode_end])
        elif isinstance(demo_data[key], ZarrGroup):
            if key in ['actors', 'articulations', 'controller', 'arm']:
                recursive_trim_trimmed_arrays(demo_data[key], new_episode_start, new_episode_end, new_untrimmed_episode_start, new_untrimmed_episode_end, pretrimmed=False)
            else:
                recursive_trim_trimmed_arrays(demo_data[key], new_episode_start, new_episode_end, new_untrimmed_episode_start, new_untrimmed_episode_end, pretrimmed=True)

def recursive_trim_trimmed_arrays_to_new_demo(demo_data: ZarrGroup, 
                                              new_demo_data: ZarrGroup,
                                  new_episode_start: int, new_episode_end: int, 
                                  new_untrimmed_episode_start: int, new_untrimmed_episode_end: int, 
                                  pretrimmed: bool = True):
    # pretrimmed refers to fact that the non state arrays under actors/articulations groups have been trimmed to exclude one frame per episode (first frame for action, and last frame for observations)
    for key in demo_data.keys():
        if isinstance(demo_data[key], ZarrArray) and not key == 'start':
            if f'{key}' not in new_demo_data:
                new_demo_data.create_array(key, shape=(0, *demo_data[key].shape[1:]), dtype=demo_data[key].dtype, chunks=(1, *demo_data[key].shape[1:]), compressors=demo_data[key].compressors)
            if pretrimmed:
                new_demo_data[key].append(demo_data[key][new_episode_start:new_episode_end])
            else:
                new_demo_data[key].append(demo_data[key][new_untrimmed_episode_start:new_untrimmed_episode_end])
        elif isinstance(demo_data[key], ZarrGroup):
            if key not in new_demo_data:
                new_demo_data.create_group(key)
            if key in ['actors', 'articulations', 'controller', 'arm']:
                recursive_trim_trimmed_arrays_to_new_demo(demo_data[key], new_demo_data[key], new_episode_start, new_episode_end, new_untrimmed_episode_start, new_untrimmed_episode_end, pretrimmed=False)
            else:
                recursive_trim_trimmed_arrays_to_new_demo(demo_data[key], new_demo_data[key], new_episode_start, new_episode_end, new_untrimmed_episode_start, new_untrimmed_episode_end, pretrimmed=True)

# delete the non_tmp arrays
def recursive_delete_non_tmp_arrays(demo_data: ZarrGroup):
    for key in demo_data.keys():
        if isinstance(demo_data[key], ZarrArray) and not key.endswith('_tmp'):
            del demo_data[key]
        elif isinstance(demo_data[key], ZarrGroup):
            recursive_delete_non_tmp_arrays(demo_data[key])

def rename_zarr_array(demo_data: ZarrGroup, old_key: str, new_key: str, copy_over_n_chunks_at_a_time: int = 500):
    if old_key in demo_data:
        # demo_data.move(old_key, new_key)
        # need to manually rename the array by copying the data and then deleting the old array
        demo_data.create_array(new_key, shape=(0, *demo_data[old_key].shape[1:]), dtype=demo_data[old_key].dtype, chunks=demo_data[old_key].chunks, compressors=demo_data[old_key].compressors)
        # need to copy over n chunks at a time because the array is too large
        for i in tqdm(range(0, demo_data[old_key].shape[0], copy_over_n_chunks_at_a_time), desc=f"Copying {old_key} to {new_key}", unit="chunk"):
            start = i
            end = min(i + copy_over_n_chunks_at_a_time, demo_data[old_key].shape[0])
            demo_data[new_key].append(demo_data[old_key][start:end])
        assert demo_data[old_key].shape == demo_data[new_key].shape, f"demo_data[old_key].shape: {demo_data[old_key].shape} != demo_data[new_key].shape: {demo_data[new_key].shape}"
        del demo_data[old_key]
    else:
        raise KeyError(f"{old_key} not found in {demo_data.name}")

# rename the tmp arrays back to the original names
def recursive_rename_tmp_arrays(demo_data: ZarrGroup):
    for key in demo_data.keys():
        if isinstance(demo_data[key], ZarrArray) and key.endswith('_tmp'):
            # zarr.store.rename(key, key[:-4])
            # use move
            if ZARR_VERSION < 3:
                demo_data.move(key, key[:-4])
            else:
                rename_zarr_array(demo_data, key, key[:-4])
        elif isinstance(demo_data[key], ZarrGroup):
            recursive_rename_tmp_arrays(demo_data[key])
            
def all_arrays_are_tmp(demo_data: ZarrGroup):
    for key in demo_data.keys():
        if isinstance(demo_data[key], ZarrArray) and not key.endswith('_tmp'):
            return False
        elif isinstance(demo_data[key], ZarrGroup):
            if not all_arrays_are_tmp(demo_data[key]):
                return False
    return True

def trim_start_and_end_of_trajectories(demo: ZarrGroup, 
                                    meta_json: dict, 
                                    path_to_json: Path,
                                    total_action_norm_threshold: float = .005,
                                    ):
    # raise NotImplementedError("Theres a bug when using with zarr v3 that doubles the size of the arrays. and pads the original array shape with zeros")
    # found the bug, it was from the manual rename zarr array function not using the correct shape (just passed in shape of current array, hence the doubling)
    post_process_logger.info(f"trimming episodes of {path_to_json.name}...")
    num_episodes = demo['meta']['episode_ends'].shape[0]
    # if not all_arrays_are_tmp(demo['data']):
    for trajectory_idx in tqdm(range(num_episodes), desc="Trimming episodes", unit="episode"):
        episode_start = demo['meta']['episode_ends'][trajectory_idx - 1] if trajectory_idx > 0 else 0
        episode_end = demo['meta']['episode_ends'][trajectory_idx]
        untrimmed_episode_start = episode_start + trajectory_idx
        untrimmed_episode_end = episode_end + trajectory_idx + 1

        episode_length = episode_end - episode_start
        assert episode_length == meta_json['episodes'][trajectory_idx]['elapsed_steps']

        actions_for_episode = demo['data']['action'][episode_start:episode_end]
        action_total_norms = np.linalg.norm(actions_for_episode[:,0:6], axis=1, ord=2)
        threshold_condition_idx = np.argwhere(action_total_norms > total_action_norm_threshold)
        assert len(threshold_condition_idx) > 0
        new_episode_start = threshold_condition_idx[0][0] - 1 + episode_start

        gripper_actions = actions_for_episode[:,6]
        threshold_condition_idx = np.argwhere(gripper_actions > -1)
        assert len(threshold_condition_idx) > 0
        new_episode_end = threshold_condition_idx[0][0] + episode_start

        assert new_episode_start < new_episode_end
        new_episode_length = new_episode_end - new_episode_start

        start_trim = new_episode_start - episode_start
        end_trim = episode_end - new_episode_end
        post_process_logger.info(f"Trajectory {trajectory_idx}: Trimmed {start_trim} frames from the start and {end_trim} frames from the end")

        new_untrimmed_episode_start = untrimmed_episode_start + start_trim
        new_untrimmed_episode_end = untrimmed_episode_end - end_trim

        assert new_episode_length == (new_untrimmed_episode_end - new_untrimmed_episode_start) - 1

        recursive_trim_trimmed_arrays(demo['data'], new_episode_start, new_episode_end, new_untrimmed_episode_start, new_untrimmed_episode_end, pretrimmed=True)

        meta_json['episodes'][trajectory_idx]['elapsed_steps'] = int(new_episode_length)

    episode_start = 0
    for trajectory_idx in range(num_episodes):
        episode_length = meta_json['episodes'][trajectory_idx]['elapsed_steps']
        demo['meta']['episode_ends'][trajectory_idx] = episode_length + episode_start
        episode_start += episode_length

    post_process_logger.info(f"Deleting non-tmp arrays...")
    recursive_delete_non_tmp_arrays(demo['data'])
    
    assert all_arrays_are_tmp(demo['data']), "Not all arrays are tmp arrays"
    
    post_process_logger.info(f"Renaming tmp arrays...")
    recursive_rename_tmp_arrays(demo['data'])

    # change the dtype of episode elapsed_steps to int
    for episode in meta_json['episodes']:
        episode['elapsed_steps'] = int(episode['elapsed_steps'])
    # update the json file
    with open(path_to_json, 'w') as f:
        json.dump(meta_json, f, indent=4)

def recursive_copy_meta_data(base_demo, new_demo):
    for key in base_demo.keys():
        if isinstance(base_demo[key], ZarrArray):
            # if key == 'ep_ids':
            #     new_demo.create_array(key, shape=base_demo[key].shape, dtype='|S256', chunks=base_demo[key].chunks, compressors=base_demo[key].compressors)
            #     for i, episode_id in enumerate(base_demo['ep_ids'][:]):
            #         episode_id_string = episode_id.decode('utf-8')
            #         assert episode_id_string.startswith('traj_')
            #         episode_id_string = f"{episode_id_string}"
            #         new_demo['ep_ids'][i:i+1] = np.array([episode_id_string.encode('utf-8')])
            # else:    
            new_demo.create_array(key, shape=base_demo[key].shape, dtype=base_demo[key].dtype, chunks=base_demo[key].chunks, compressors=base_demo[key].compressors)
            new_demo[key][...] = base_demo[key][:]
        elif isinstance(base_demo[key], ZarrGroup):
            new_demo.create_group(key)
            recursive_copy_meta_data(base_demo[key], new_demo[key])

def trim_start_and_end_of_trajectories_in_new_dataset(demo: ZarrGroup, 
                                    meta_json: dict, 
                                    path_to_json: Path,
                                    total_action_norm_threshold: float = .005,
                                    ):
    demo_path = demo.store.root
    new_demo_path = demo_path.with_name(f"{demo_path.stem}_trimmed{demo_path.suffix}")
    new_demo = zarr.storage.LocalStore(new_demo_path)
    new_demo = zarr.group(store=new_demo)
    new_demo.create_group("data")
    new_demo.create_group("meta")

    # raise NotImplementedError("Theres a bug when using with zarr v3 that doubles the size of the arrays. and pads the original array shape with zeros")
    # found the bug, it was from the manual rename zarr array function not using the correct shape (just passed in shape of current array, hence the doubling)
    post_process_logger.info(f"trimming episodes of {path_to_json.name}...")
    num_episodes = demo['meta']['episode_ends'].shape[0]
    # if not all_arrays_are_tmp(demo['data']):
    for trajectory_idx in tqdm(range(num_episodes), desc="Trimming episodes", unit="episode"):
        episode_start = demo['meta']['episode_ends'][trajectory_idx - 1] if trajectory_idx > 0 else 0
        episode_end = demo['meta']['episode_ends'][trajectory_idx]
        untrimmed_episode_start = episode_start + trajectory_idx
        untrimmed_episode_end = episode_end + trajectory_idx + 1

        episode_length = episode_end - episode_start
        assert episode_length == meta_json['episodes'][trajectory_idx]['elapsed_steps'], f"episode_length: {episode_length} != meta_json['episodes'][trajectory_idx]['elapsed_steps']: {meta_json['episodes'][trajectory_idx]['elapsed_steps']}"

        actions_for_episode = demo['data']['action'][episode_start:episode_end]
        action_total_norms = np.linalg.norm(actions_for_episode[:,0:6], axis=1, ord=2)
        threshold_condition_idx = np.argwhere(action_total_norms > total_action_norm_threshold)
        assert len(threshold_condition_idx) > 0, f"did not find any action norms greater than {total_action_norm_threshold} for trajectory_idx: {trajectory_idx}"
        new_episode_start = threshold_condition_idx[0][0] - 1 + episode_start
        if 'start' in demo['data']:
            start_signal_for_episode = demo['data']['start'][episode_start:episode_end]
            start_signal_threshold_idx = np.argwhere(start_signal_for_episode)
            if len(start_signal_threshold_idx) > 0:
                # new_episode_start = start_signal_threshold_idx[0][0] + episode_start
                # find the first threshold_condition_idx that is equal or greater than the start_signal_threshold_idx
                threshold_condition_idx_after_start_idx = np.argwhere(threshold_condition_idx >= start_signal_threshold_idx[0][0])
                assert len(threshold_condition_idx_after_start_idx) > 0, f"did not find any threshold_condition_idx after start_signal_threshold_idx for trajectory_idx: {trajectory_idx}"
                new_episode_start = threshold_condition_idx[threshold_condition_idx_after_start_idx[0][0]][0] - 1 + episode_start

        gripper_actions = actions_for_episode[:,6]
        threshold_condition_idx = np.argwhere(gripper_actions > -1)
        assert len(threshold_condition_idx) > 0, f"did not find any gripper actions greater than -1 for trajectory_idx: {trajectory_idx}"

        new_episode_end = threshold_condition_idx[0][0] + episode_start

        assert new_episode_start < new_episode_end
        new_episode_length = new_episode_end - new_episode_start

        start_trim = new_episode_start - episode_start
        end_trim = episode_end - new_episode_end

        new_untrimmed_episode_start = untrimmed_episode_start + start_trim
        new_untrimmed_episode_end = untrimmed_episode_end - end_trim

        assert new_episode_length == (new_untrimmed_episode_end - new_untrimmed_episode_start) - 1

        post_process_logger.info(f"Trajectory {trajectory_idx}: trimming {start_trim} frames from the start and {end_trim} frames from the end")
        recursive_trim_trimmed_arrays_to_new_demo(demo['data'], new_demo['data'], new_episode_start, new_episode_end, new_untrimmed_episode_start, new_untrimmed_episode_end, pretrimmed=True)

        meta_json['episodes'][trajectory_idx]['elapsed_steps'] = int(new_episode_length)

    # copy the meta data
    recursive_copy_meta_data(demo['meta'], new_demo['meta'])

    episode_start = 0
    for trajectory_idx in range(num_episodes):
        episode_length = meta_json['episodes'][trajectory_idx]['elapsed_steps']
        new_demo['meta']['episode_ends'][trajectory_idx] = episode_length + episode_start
        episode_start += episode_length

    # change the dtype of episode elapsed_steps to int
    for episode in meta_json['episodes']:
        episode['elapsed_steps'] = int(episode['elapsed_steps'])
    # update the json file
    with open(path_to_json.with_name(f"{path_to_json.stem}_trimmed.json"), 'w') as f:
        json.dump(meta_json, f, indent=4)

def recursive_assert_structure(base_demo, new_demo):
    base_demo_keys = set(base_demo.keys())
    new_demo_keys = set(new_demo.keys())
    # remove episode_data
    base_demo_keys.discard('episode_data')
    new_demo_keys.discard('episode_data')
    assert base_demo_keys == new_demo_keys, f"Keys mismatch: {base_demo_keys} vs {new_demo_keys}"
    for key in base_demo.keys():
        if isinstance(base_demo[key], ZarrArray):
            assert base_demo[key].shape[1:] == new_demo[key].shape[1:], f"Shape mismatch for {key}: {base_demo[key].shape[1:]} vs {new_demo[key].shape[1:]}"
            assert base_demo[key].dtype == new_demo[key].dtype, f"Type mismatch for {key}: {base_demo[key].dtype} vs {new_demo[key].dtype}"
        elif isinstance(base_demo[key], ZarrGroup):
            if key == 'episode_data': # ignore episode data
                continue
            recursive_assert_structure(base_demo[key], new_demo[key])

def merge_demos_into_base_demo(base_demo_path: Path, demos_to_add_to_base_paths: list, delete_merged_demos: bool = False, create_copy_of_base_demo: bool = True):
    # before merging, make a copy of the base demo
    base_demo_copy_path = base_demo_path.with_name(f"{base_demo_path.stem}_copy{base_demo_path.suffix}")
    if not base_demo_copy_path.exists() and create_copy_of_base_demo:
        post_process_logger.info(f"Copying base demo {base_demo_path.name} to {base_demo_copy_path.name}...")
        shutil.copytree(base_demo_path, base_demo_copy_path)
    # also copy the json file
    base_meta_json_path = base_demo_path.with_suffix('.json')
    base_meta_json_copy_path = base_meta_json_path.with_name(f"{base_meta_json_path.stem}_copy{base_meta_json_path.suffix}")
    if not base_meta_json_copy_path.exists():
        post_process_logger.info(f"Copying base demo json {base_meta_json_path.name} to {base_meta_json_copy_path.name}...")
        shutil.copy(base_meta_json_path, base_meta_json_copy_path)

    post_process_logger.info(f"Merging {len(demos_to_add_to_base_paths)} demo datasets into base demo {base_demo_path.name}...")
    base_demo = zarr.open(base_demo_path, mode='r+')
    if 'ep_ids' in base_demo['meta']:
        # delete the ep_ids
        print(f"deleting ep_ids of dataset {base_demo_path} as its not used")
        del base_demo['meta']['ep_ids']
    with open(base_meta_json_path, 'r') as f:
        base_meta_json = json.load(f)
    for new_demo_path in demos_to_add_to_base_paths:
        new_demo = zarr.open(new_demo_path, mode='r+')
        if 'ep_ids' in new_demo['meta']:
            print(f"deleting ep_ids of dataset {new_demo_path} as its not used")
            del new_demo['meta']['ep_ids']

        new_meta_json_path = new_demo_path.with_suffix('.json')
        with open(new_meta_json_path, 'r') as f:
            new_meta_json = json.load(f)

        if new_meta_json['max_demo_length'] > base_meta_json['max_demo_length']:
            base_meta_json['max_demo_length'] = new_meta_json['max_demo_length']

        difference = DeepDiff(base_meta_json, new_meta_json)
        values_changed_prefixes = [
            "root['episodes']",
            "root['commit_info']",
            "root['max_demo_length']",
        ]
        if 'values_changed' in difference:
            # only values that should have changed are "root['episodes']..."
            assert all(any(key.startswith(prefix) for prefix in values_changed_prefixes) for key in difference['values_changed'].keys())
        if 'iterable_item_added' in difference:
            # only values that should have changed are "root['episodes']..."
            assert all([key.startswith("root['episodes']") for key in difference['iterable_item_added'].keys()])

        recursive_assert_structure(base_demo, new_demo)
        recursive_assert_structure(new_demo, base_demo)

        # before merging, change some of the metadata of the new demo
        # last episode end of the base demo
        last_episode_end_of_base_demo = base_demo['meta']['episode_ends'][-1]
        base_demo_num_episodes = base_demo['meta']['episode_ends'].shape[0]

        # check if new_demo has already been modified
        new_demo_already_modified = False
        first_episode_elapsed_steps = new_meta_json['episodes'][0]['elapsed_steps']
        if first_episode_elapsed_steps != new_demo['meta']['episode_ends'][0]:
            new_demo_already_modified = True
            post_process_logger.info(f"New demo {new_demo_path.name} has already been modified, skipping updating...")

        # first update episode_ends of new demo
        if not new_demo_already_modified:
            new_demo['meta']['episode_ends'][...] += last_episode_end_of_base_demo

            # also update the ep_ids of the new demo
            # for i, episode_id in enumerate(new_demo['meta']['ep_ids'][:]):
            #     episode_id_string = episode_id.decode('utf-8')
            #     assert episode_id_string.startswith('traj_')
            #     current_episode_id = int(episode_id_string.split('_')[-1])
            #     new_episode_id = f'traj_{current_episode_id + base_demo_num_episodes}'
            #     new_demo['meta']['ep_ids'][i:i+1] = [new_episode_id.encode('utf-8')]
        
        for episode_dict in new_meta_json['episodes']:
            episode_dict['episode_id'] += base_demo_num_episodes
        
        # merge the two datasets by recursively appending the new demo data to the base demo data
        recursive_append_new_demo_data(base_demo, new_demo)
        
        # update the meta json
        base_meta_json['episodes'] += new_meta_json['episodes']
        
        # update the json file
        with open(base_meta_json_path, 'w') as f:
            json.dump(base_meta_json, f, indent=4)

        # then delete the new demo zarr and json
        if delete_merged_demos:
            shutil.rmtree(new_demo_path)

            new_meta_json_path.unlink()
        
def correct_faulty_trimming(demo_data: ZarrGroup,
                            to_remove_start_idx: int = 0,
                            to_remove_end_idx: int = 0,
                            env_state_to_remove_start_idx: int = 0,
                            env_state_to_remove_end_idx: int = 0,
                            env_state_array: bool = False,
                            ):
    # pretrimmed refers to fact that the non state arrays under actors/articulations groups have been trimmed to exclude one frame per episode (first frame for action, and last frame for observations)
    assert to_remove_start_idx < to_remove_end_idx, f"to_remove_start_idx: {to_remove_start_idx} >= to_remove_end_idx: {to_remove_end_idx}"
    assert env_state_to_remove_start_idx < env_state_to_remove_end_idx, f"env_state_to_remove_start_idx: {env_state_to_remove_start_idx} >= env_state_to_remove_end_idx: {env_state_to_remove_end_idx}"
    for key in demo_data.keys():
        if isinstance(demo_data[key], ZarrArray) and not key.endswith('_tmp'):
            if f'{key}_tmp' not in demo_data:
                demo_data.create_array(f'{key}_tmp', shape=(0, *demo_data[key].shape[1:]), dtype=demo_data[key].dtype, chunks=(1, *demo_data[key].shape[1:]), compressors=demo_data[key].compressors)
                if not env_state_array:
                    if to_remove_start_idx > 0:
                        demo_data[f'{key}_tmp'].append(demo_data[key][0:to_remove_start_idx])
                    if to_remove_end_idx < demo_data[key].shape[0]:
                        demo_data[f'{key}_tmp'].append(demo_data[key][to_remove_end_idx:])
                    assert demo_data[f'{key}_tmp'].shape[0] == demo_data[key].shape[0] - (to_remove_end_idx - to_remove_start_idx), f"demo_data[f'{key}_tmp'].shape[0]: {demo_data[f'{key}_tmp'].shape[0]} != demo_data[key].shape[0] - (to_remove_end_idx - to_remove_start_idx): {demo_data[key].shape[0] - (to_remove_end_idx - to_remove_start_idx)}"
                else:
                    if env_state_to_remove_start_idx > 0:
                        demo_data[f'{key}_tmp'].append(demo_data[key][0:env_state_to_remove_start_idx])
                    if env_state_to_remove_end_idx < demo_data[key].shape[0]:
                        demo_data[f'{key}_tmp'].append(demo_data[key][env_state_to_remove_end_idx:])
                    assert demo_data[f'{key}_tmp'].shape[0] == demo_data[key].shape[0] - (env_state_to_remove_end_idx - env_state_to_remove_start_idx), f"demo_data[f'{key}_tmp'].shape[0]: {demo_data[f'{key}_tmp'].shape[0]} != demo_data[key].shape[0] - (to_remove_end_idx - to_remove_start_idx): {demo_data[key].shape[0] - (env_state_to_remove_end_idx - env_state_to_remove_start_idx)}"
        elif isinstance(demo_data[key], ZarrGroup):
            if key in ['actors', 'articulations', 'controller', 'arm']:
                correct_faulty_trimming(demo_data[key], to_remove_start_idx, to_remove_end_idx, env_state_to_remove_start_idx, env_state_to_remove_end_idx, env_state_array=True)
            else:
                correct_faulty_trimming(demo_data[key], to_remove_start_idx, to_remove_end_idx, env_state_to_remove_start_idx, env_state_to_remove_end_idx, env_state_array=False)
    
# def remove_episodes_from_dataset(demo: ZarrGroup, 
#                             meta_json: dict, 
#                             episodes_to_remove: list,
#                             ):        

# # traverse the tree and print any attrs of groups or arrays
# def traverse_tree(node, indent=0):
#     if isinstance(node, ZarrGroup):
#         print(f"{' '*indent}{node.name} with attrs: {list(node.attrs.items())}")
#         for key in node.keys():
#             traverse_tree(node[key], indent+2)
#     elif isinstance(node, ZarrArray):
#         print(f"{' '*indent}{node.name} with attrs: {list(node.attrs.items())}")

# traverse_tree(demo)
#%%
# # # base_demo_path = Path('/mnt/crucialSSD/datasetsSSD/fish_datasets/simulated/teleop/2_sim_recovery_demos_peginsertion_20hz_act/demos.zarr')
# # base_demo_path = Path('/mnt/crucialSSD/datasetsSSD/fish_datasets/simulated/teleop/20250818_111025.zarr')
# # base_demo_path = Path('/mnt/crucialSSD/datasetsSSD/fish_datasets/simulated/teleop/3_sim_nominal_demos_peginsertion_20hz_act/demos.zarr')
# # base_demo_path = Path('/mnt/crucialSSD/datasetsSSD/fish_datasets/simulated/teleop/2_sim_recovery_demos_peginsertion_20hz_act/demos.zarr')
# base_demo_path = Path('/mnt/crucialSSD/datasetsSSD/fish_datasets/simulated/teleop/5_sim_all_demos_peginsertion_20hz_act/demos.zarr')

# zarr_store = zarr.open(base_demo_path, mode='r')

# from mani_skill.utils.visualization import images_to_video
# episode_idx = 2
# episode_start = zarr_store['meta']['episode_ends'][episode_idx - 1] if episode_idx > 0 else 0
# episode_end = zarr_store['meta']['episode_ends'][episode_idx]
# images = zarr_store['data']['observation.rgb'][episode_start:episode_end]
# # images = zarr_store['episode_data'][f'episode_{episode_idx}']['sam2-hiera-base-plus']['observation.EE_obj_mask'][:]
# # images *= 255
# # images = images.astype(np.uint8)
# #%%
# # wrenches = zarr_store['data']['observation.end_effector_external_wrench_in_world'][episode_start:episode_end]
# # import matplotlib.pyplot as plt
# # plt.figure(figsize=(10, 5))
# # plt.plot(wrenches[:, 0], label='Force X')
# # plt.plot(wrenches[:, 1], label='Force Y')
# # plt.plot(wrenches[:, 2], label='Force Z')
# # plt.xlabel('Time Step')
# # plt.ylabel('Force (N)')
# # plt.title('End Effector External Wrench')
# # plt.legend()
# # plt.grid()
# # plt.show()
# #%%

# images_to_video(
#     images=images,
#     output_dir='./',
#     video_name=f'episode_{episode_idx}_video',
#     fps=20,
# )
#%%
# assert base_demo_path.exists()
# # # # # base_demo_num_episodes = 10
# base_demo = zarr.open(base_demo_path, 'r')
# episode_idx = 99
# episode_start_idx = 0 if episode_idx == 0 else base_demo['meta']['episode_ends'][episode_idx - 1]
# episode_end_idx = base_demo['meta']['episode_ends'][episode_idx]
# gripper_actions = base_demo['data']['action'][episode_start_idx:episode_end_idx][:,6]

# start_signal_for_episode = base_demo['data']['start'][episode_start_idx:episode_end_idx]
# start_signal_threshold_idx = np.argwhere(start_signal_for_episode)

# rgb_frames = base_demo['data']['observation.rgb'][episode_start_idx:episode_end_idx]

# images_to_video(
#     images=rgb_frames,
#     output_dir='./',
#     video_name=f'episode_{episode_idx}_video',
#     fps=20,
#     )
# episode_start = base_demo['data']['start'][episode_start_idx:episode_end_idx]
# if np.any(episode_start):
#     start_idx = np.argwhere(episode_start)[0][0]
#     color_frame_at_start = base_demo['data']['observation.rgb'][episode_start_idx:episode_end_idx][start_idx]
#     plt.imshow(color_frame_at_start)
# # for i, episode_id in enumerate(base_demo['meta']['ep_ids'][:]):
# #     episode_id_string = episode_id.decode('utf-8')
# #     assert episode_id_string.startswith('traj_')
# #     current_episode_id = int(episode_id_string.split('_')[-1])
# #     new_episode_id = f'traj_{current_episode_id + base_demo_num_episodes}'
# #     print(f"old: {episode_id_string.encode('utf-8')} new: {new_episode_id.encode('utf-8')}")
# #     # new_demo['meta']['ep_ids'][i] = new_episode_id.encode('utf-8')
#%%
# #################################################################################
# correct datasets that were mangled by bug in my code which caused each array to be doubled and padded with zeros at the first half
# #################################################################################
# path_to_demo = Path('/mnt/crucialSSD/datasetsSSD/fish_datasets/simulated/teleop/240_sim_demos_left_of_4th_book_bookends_no_env_rand_20hz_act/demos.zarr')
# demo = zarr.open(path_to_demo, mode='r+')
# path_to_demo_json = path_to_demo.with_suffix('.json')
# with open(path_to_demo_json, 'r') as f:
#     meta_json = json.load(f)

# # num_episodes = len(demo['meta']['episode_ends'][:])
# first_half_num_episodes = 151 + 1
# actual_num_episodes = demo['meta']['episode_ends'].shape[0]

# to_remove_start_idx = demo['meta']['episode_ends'][151]
# to_remove_end_idx = demo['meta']['episode_ends'][-1]
# env_state_to_remove_start_idx = demo['meta']['episode_ends'][151] + first_half_num_episodes
# env_state_to_remove_end_idx = demo['meta']['episode_ends'][-1] + actual_num_episodes

# print(f"to_remove_start_idx: {to_remove_start_idx} to_remove_end_idx: {to_remove_end_idx} env_state_to_remove_start_idx: {env_state_to_remove_start_idx} env_state_to_remove_end_idx: {env_state_to_remove_end_idx}")
# print(f"new length will be {demo['data']['observation.rgb'].shape[0] - (to_remove_end_idx - to_remove_start_idx)}")
# #%%
# correct_faulty_trimming(demo['data'], to_remove_start_idx, to_remove_end_idx, env_state_to_remove_start_idx, env_state_to_remove_end_idx)
# recursive_delete_non_tmp_arrays(demo['data'])
# assert all_arrays_are_tmp(demo['data']), "Not all arrays are tmp arrays"
# recursive_rename_tmp_arrays(demo['data'])

#%%

# demo = zarr.open('/mnt/crucialSSD/datasetsSSD/fish_datasets/simulated/teleop/206_sim_demos_leftof4thbook_springbookends_nograspedrand_noenvrand_slotrand_20hz_act/demos.zarr', mode='r')

#%%
# demo_json = demo.store.root.with_suffix('.json')
# with open(demo_json, 'r') as f:
#     meta_json = json.load(f)
# # # create video for first episode
# episode_idx = 2
# num_episodes = demo['meta']['episode_ends'].shape[0]

# episode_start = demo['meta']['episode_ends'][episode_idx - 1] if episode_idx > 0 else 0
# env_state_episode_start = episode_start + episode_idx
# # episode_start += demo['meta']['episode_ends'][-1]
# # env_state_episode_start += demo['meta']['episode_ends'][-1] + num_episodes
# episode_end = demo['meta']['episode_ends'][episode_idx]
# env_state_episode_end = episode_end + episode_idx + 1
# # episode_end += demo['meta']['episode_ends'][-1]
# # env_state_episode_end += demo['meta']['episode_ends'][-1] + num_episodes

# rgb_frames = demo['data']['observation.rgb'][episode_start:episode_end]

# print(f"episode_start: {episode_start} episode_end: {episode_end} env_state_episode_start: {env_state_episode_start} env_state_episode_end: {env_state_episode_end}")
# print(f"episode_length: {episode_end - episode_start} env_state_episode_length: {env_state_episode_end - env_state_episode_start}")

# # from matplotlib import pyplot as plt
# # plt.imshow(demo['data']['observation.rgb'][27630])
# # create a video from the rgb frames
# images_to_video(
#     images=rgb_frames,
#     output_dir='./',
#     video_name=f'episode_{episode_idx}_video',
#     fps=20,
# )
# #%%
# demo_2 = zarr.open('/mnt/crucialSSD/datasetsSSD/fish_datasets/simulated/teleop/206_sim_demos_leftof4thbook_springbookends_nograspedrand_noenvrand_slotrand_20hz_act/demos.zarr', mode='r')
# demo_2_json = demo_2.store.root.with_suffix('.json')
# with open(demo_2_json, 'r') as f:
#     meta_json_2 = json.load(f)

# difference = DeepDiff(meta_json, meta_json_2)
#%%
#%%
# #################################################################################
# trim each dataset using thresholds on velocity and gripper action
# #################################################################################

dataset_name = 'sim_all_demos_peginsertion_20hz_act'
# dataset_name = 'sim_recovery_demos_peginsertion_20hz_act'
# dataset_root_dir = Path('/mnt/crucialSSD/datasetsSSD/fish_datasets/simulated/teleop')
dataset_root_dir = Path('/mnt/kostas-graid/datasets/extrinsic_contact_data/FISH/expert_demos/frankagym/FrankaInsertion-v1')
# demos_to_trim = [
#     Path('/mnt/crucialSSD/datasetsSSD/fish_datasets/simulated/teleop/20250818_112731.zarr'),
# ]
demos_to_trim = list()
for demo_path in demos_to_trim:
    assert demo_path.exists()
    demo_path = demo_path.expanduser()

#%%
for path_to_demo in demos_to_trim:
    path_to_demo = path_to_demo.expanduser()
    demo = zarr.open(path_to_demo, mode='r+')

    path_to_json = path_to_demo.with_suffix('.json')
    with open(path_to_json, 'r') as f:
        meta_json = json.load(f)
    num_episodes = len(demo['meta']['episode_ends'][:])
    assert num_episodes == len(meta_json['episodes'])

    # trim_start_and_end_of_trajectories(demo, meta_json, path_to_json, total_action_norm_threshold=.005)
    trim_start_and_end_of_trajectories_in_new_dataset(demo, meta_json, path_to_json, total_action_norm_threshold=.005)

#%%
# ###########################
# merge datasets together
# ###########################
# base_demo_path = Path('/mnt/crucialSSD/datasetsSSD/fish_datasets/simulated/teleop/206_sim_demos_leftof4thbook_springbookends_nograspedrand_noenvrand_slotrand_20hz_act/demos.zarr')
# base_demo_path = Path('/mnt/crucialSSD/datasetsSSD/fish_datasets/simulated/teleop/20250428_175948_trimmed.zarr')
# base_demo_path = Path('/mnt/crucialSSD/datasetsSSD/fish_datasets/simulated/teleop/20250818_112731_trimmed.zarr')

# base_demo_path = Path('/mnt/crucialSSD/datasetsSSD/fish_datasets/simulated/teleop/20250818_112731_trimmed.zarr')
# base_demo_path = Path('/mnt/crucialSSD/datasetsSSD/fish_datasets/simulated/teleop/3_sim_nominal_demos_peginsertion_20hz_act/demos.zarr')
# base_demo_path = Path('/mnt/kostas-graid/datasets/extrinsic_contact_data/FISH/expert_demos/frankagym/FrankaInsertion-v1/700_sim_demos_leftof4thbook_springbookends_graspedrand_noenvrand_slotrand_20hz_act_copy/demos.zarr')
base_demo_path = Path('/mnt/kostas-graid/datasets/extrinsic_contact_data/FISH/expert_demos/frankagym/FrankaInsertion-v1/638_sim_nominal_demos_peginsertion_20hz_act/demos.zarr')

assert base_demo_path.exists()
base_demo_path = base_demo_path.expanduser()
demos_to_add_to_base_paths = [
    # Path('/mnt/kostas-graid/datasets/extrinsic_contact_data/FISH/expert_demos/frankagym/FrankaInsertion-v1/532_sim_recovery_demos_leftof4thbook_springbookends_graspedrand_noenvrand_slotrand_20hz_act/demos.zarr'),
    Path('/mnt/kostas-graid/datasets/extrinsic_contact_data/FISH/expert_demos/frankagym/FrankaInsertion-v1/269_sim_recovery_demos_peginsertion_20hz_act/demos.zarr'),
]
# demos_to_add_to_base_paths = list()

for demo_path in demos_to_add_to_base_paths:
    assert demo_path.exists()
    demo_path = demo_path.expanduser()
# #%%
# base_demo_json_path = base_demo_path.with_suffix('.json')
# with open(base_demo_json_path, 'r') as f:
#     base_meta_json = json.load(f)
# demo_to_add_to_base_path_json_path = demos_to_add_to_base_paths[0].with_suffix('.json')
# with open(demo_to_add_to_base_path_json_path, 'r') as f:
#     demo_to_add_to_base_meta_json = json.load(f)

# difference = DeepDiff(base_meta_json, demo_to_add_to_base_meta_json)
#%%
if len(demos_to_add_to_base_paths) > 0:
    merge_demos_into_base_demo(base_demo_path, demos_to_add_to_base_paths, delete_merged_demos=True, create_copy_of_base_demo=False)

# ###########################
# change the dataset name and location
# ###########################

demo = zarr.open(base_demo_path, mode='r+')
meta_json = json.load(open(base_demo_path.with_suffix('.json'), 'r'))

# add some needed meta attrs
max_demo_length = 0
for episode_dict in meta_json['episodes']:
    episode_length = episode_dict['elapsed_steps']
    if episode_length > max_demo_length:
        max_demo_length = episode_length

meta_json['max_demo_length'] = max_demo_length
demo['meta'].attrs['max_demo_length'] = max_demo_length
# update the json file
with open(base_demo_path.with_suffix('.json'), 'w') as f:
    json.dump(meta_json, f, indent=4)

# move the zarr and json file to a directory
# dataset_root_dir = Path('/mnt/crucialSSD/datasetsSSD/fish_datasets/simulated/teleop')
total_num_demos = len(demo['meta']['episode_ends'][:])
dataset_name = f'{total_num_demos}_' + dataset_name
# dataset_root_dir = base_demo_path.parent

new_dataset_dir = dataset_root_dir / dataset_name
new_dataset_dir.mkdir(parents=True, exist_ok=True)
#%%
shutil.move(base_demo_path, new_dataset_dir)
shutil.move(base_demo_path.with_suffix('.json'), new_dataset_dir)
#%%
# rename the zarr to demos.zarr
new_demo_path = new_dataset_dir / base_demo_path.name
new_demo_path.rename(new_dataset_dir / 'demos.zarr')
#%%
# rename the json file to demos.json
new_json_path = new_dataset_dir / base_demo_path.with_suffix('.json').name
new_json_path.rename(new_dataset_dir / 'demos.json')

#%%