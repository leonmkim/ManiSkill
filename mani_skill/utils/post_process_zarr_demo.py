#%%
from deepdiff import DeepDiff
import zarr
from pathlib import Path

import numpy as np

import json

import matplotlib.pyplot as plt

import shutil

from mani_skill.utils.visualization.misc import images_to_video, tile_images

#%%
def recursive_append_new_demo_data(base_demo, new_demo):
    for key in base_demo.keys():
        if isinstance(base_demo[key], zarr.core.Array):
            base_demo[key].append(new_demo[key][:])
        elif isinstance(base_demo[key], zarr.hierarchy.Group):
            recursive_append_new_demo_data(base_demo[key], new_demo[key])

def recursive_trim_trimmed_arrays(demo_data: zarr.hierarchy.Group, 
                                  new_episode_start: int, new_episode_end: int, 
                                  new_untrimmed_episode_start: int, new_untrimmed_episode_end: int, 
                                  pretrimmed: bool = True):
    for key in demo_data.keys():
        if isinstance(demo_data[key], zarr.core.Array) and not key.endswith('_tmp'):
            if f'{key}_tmp' not in demo_data:
                demo_data.create_dataset(f'{key}_tmp', shape=(0, *demo_data[key].shape[1:]), dtype=demo_data[key].dtype, chunks=(1, *demo_data[key].shape[1:]), compressor=demo_data[key].compressor)
            if pretrimmed:
                demo_data[f'{key}_tmp'].append(demo_data[key][new_episode_start:new_episode_end])
            else:
                demo_data[f'{key}_tmp'].append(demo_data[key][new_untrimmed_episode_start:new_untrimmed_episode_end])
        elif isinstance(demo_data[key], zarr.hierarchy.Group):
            if key in ['actors', 'articulations']:
                recursive_trim_trimmed_arrays(demo_data[key], new_episode_start, new_episode_end, new_untrimmed_episode_start, new_untrimmed_episode_end, pretrimmed=False)
            else:
                recursive_trim_trimmed_arrays(demo_data[key], new_episode_start, new_episode_end, new_untrimmed_episode_start, new_untrimmed_episode_end, pretrimmed=pretrimmed)

# delete the non_tmp arrays
def recursive_delete_non_tmp_arrays(demo_data: zarr.hierarchy.Group):
    for key in demo_data.keys():
        if isinstance(demo_data[key], zarr.core.Array) and not key.endswith('_tmp'):
            del demo_data[key]
        elif isinstance(demo_data[key], zarr.hierarchy.Group):
            recursive_delete_non_tmp_arrays(demo_data[key])

# rename the tmp arrays back to the original names
def recursive_rename_tmp_arrays(demo_data: zarr.hierarchy.Group):
    for key in demo_data.keys():
        if isinstance(demo_data[key], zarr.core.Array) and key.endswith('_tmp'):
            # zarr.store.rename(key, key[:-4])
            # use move
            demo_data.move(key, key[:-4])
        elif isinstance(demo_data[key], zarr.hierarchy.Group):
            recursive_rename_tmp_arrays(demo_data[key])

def trim_start_and_end_of_trajectories(demo: zarr.hierarchy.Group, meta_json: dict, 
                                     total_action_norm_threshold: float = .005,
                                     ):
    num_episodes = len(demo.meta.episode_ends)
    for trajectory_idx in range(num_episodes):
        episode_start = demo.meta.episode_ends[trajectory_idx - 1] if trajectory_idx > 0 else 0
        episode_end = demo.meta.episode_ends[trajectory_idx]
        untrimmed_episode_start = episode_start + trajectory_idx
        untrimmed_episode_end = episode_end + trajectory_idx + 1

        episode_length = episode_end - episode_start
        assert episode_length == meta_json['episodes'][trajectory_idx]['elapsed_steps']

        actions_for_episode = demo.data['action'][episode_start:episode_end]
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
        print(f"Trajectory {trajectory_idx}: Trimmed {start_trim} frames from the start and {end_trim} frames from the end")

        new_untrimmed_episode_start = untrimmed_episode_start + start_trim
        new_untrimmed_episode_end = untrimmed_episode_end - end_trim

        assert new_episode_length == (new_untrimmed_episode_end - new_untrimmed_episode_start) - 1

        recursive_trim_trimmed_arrays(demo.data, new_episode_start, new_episode_end, new_untrimmed_episode_start, new_untrimmed_episode_end, pretrimmed=True)

        meta_json['episodes'][trajectory_idx]['elapsed_steps'] = int(new_episode_length)

    episode_start = 0
    for trajectory_idx in range(num_episodes):
        episode_length = meta_json['episodes'][trajectory_idx]['elapsed_steps']
        demo.meta.episode_ends[trajectory_idx] = episode_length + episode_start
        episode_start += episode_length
        # episode_start = demo.meta.episode_ends[trajectory_idx - 1] if trajectory_idx > 0 else 0

    recursive_delete_non_tmp_arrays(demo.data)
    recursive_rename_tmp_arrays(demo.data)

    # change the dtype of episode elapsed_steps to int
    for episode in meta_json['episodes']:
        episode['elapsed_steps'] = int(episode['elapsed_steps'])
    # update the json file
    with open(path_to_json, 'w') as f:
        json.dump(meta_json, f, indent=4)

def recursive_assert_structure(base_demo, new_demo):
    assert base_demo.keys() == new_demo.keys()
    for key in base_demo.keys():
        if isinstance(base_demo[key], zarr.core.Array):
            assert base_demo[key].shape[1:] == new_demo[key].shape[1:]
            assert base_demo[key].dtype == new_demo[key].dtype
        elif isinstance(base_demo[key], zarr.hierarchy.Group):
            recursive_assert_structure(base_demo[key], new_demo[key])

def merge_demos_into_base_demo(base_demo_path: Path, demos_to_add_to_base_paths: list):
    base_demo = zarr.open(base_demo_path, 'rw+')
    base_meta_json_path = base_demo_path.with_suffix('.json')
    with open(base_meta_json_path, 'r') as f:
        base_meta_json = json.load(f)
    for new_demo_path in demos_to_add_to_base_paths:
        new_demo = zarr.open(new_demo_path, 'rw+')
        new_meta_json_path = new_demo_path.with_suffix('.json')
        with open(new_meta_json_path, 'r') as f:
            new_meta_json = json.load(f)

        difference = DeepDiff(base_meta_json, new_meta_json)
        if 'values_changed' in difference:
            # only values that should have changed are "root['episodes']..."
            assert all([key.startswith("root['episodes']") for key in difference['values_changed'].keys()])
        if 'iterable_item_added' in difference:
            # only values that should have changed are "root['episodes']..."
            assert all([key.startswith("root['episodes']") for key in difference['iterable_item_added'].keys()])

        recursive_assert_structure(base_demo, new_demo)
        recursive_assert_structure(new_demo, base_demo)

        # before merging, change some of the metadata of the new demo
        # last episode end of the base demo
        last_episode_end_of_base_demo = base_demo.meta.episode_ends[-1]
        base_demo_num_episodes = len(base_demo.meta.episode_ends)

        # first update episode_ends of new demo
        new_demo.meta.episode_ends[...] += last_episode_end_of_base_demo 

        # also update the ep_ids of the new demo
        for i, episode_id in enumerate(new_demo.meta.ep_ids):
            episode_id_string = episode_id.decode('utf-8')
            assert episode_id_string.startswith('traj_')
            current_episode_id = int(episode_id_string.split('_')[-1])
            new_episode_id = f'traj_{current_episode_id + base_demo_num_episodes}'
            new_demo.meta.ep_ids[i] = new_episode_id.encode('utf-8')
        
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
        shutil.rmtree(new_demo_path)

        new_meta_json_path.unlink()
# # traverse the tree and print any attrs of groups or arrays
# def traverse_tree(node, indent=0):
#     if isinstance(node, zarr.hierarchy.Group):
#         print(f"{' '*indent}{node.name} with attrs: {list(node.attrs.items())}")
#         for key in node.keys():
#             traverse_tree(node[key], indent+2)
#     elif isinstance(node, zarr.core.Array):
#         print(f"{' '*indent}{node.name} with attrs: {list(node.attrs.items())}")

# traverse_tree(demo)
#%%
# path_to_demo = Path('/mnt/crucialSSD/datasetsSSD/fish_datasets/simulated/teleop/20250214_072559.zarr')
# path_to_demo = Path('/mnt/crucialSSD/datasetsSSD/fish_datasets/simulated/teleop/20250214_083920.zarr')

# path_to_demo = Path('/mnt/crucialSSD/datasetsSSD/fish_datasets/simulated/teleop/20250215_123238.zarr')
# path_to_demo = Path('/mnt/crucialSSD/datasetsSSD/fish_datasets/simulated/teleop/20250215_142312.zarr')

# path_to_demo = Path('~/fish_leon/FISH/expert_demos/frankagym/FrankaInsertion-v1/120_240x320_all_twodim_left_to_right_annotated_start_idx_5hz_zstd7_EE_pxl_coords_expert_demos_imp_act/demos.zarr')
# path_to_demo = Path('~/fish_leon/FISH/expert_demos/frankagym/FrankaInsertion-v1/test_sim_small_demos_20hz_act/demos.zarr')

dataset_name = 'sim_demos_left_of_4th_book_20hz_act'

base_demo_path = Path('/mnt/crucialSSD/datasetsSSD/fish_datasets/simulated/teleop/20250214_072559.zarr')
assert base_demo_path.exists()
base_demo_path = base_demo_path.expanduser()
demos_to_add_to_base_paths = [
    Path('/mnt/crucialSSD/datasetsSSD/fish_datasets/simulated/teleop/20250214_083920.zarr'),
]
for demo_path in demos_to_add_to_base_paths:
    assert demo_path.exists()
    demo_path = demo_path.expanduser()

all_demo_paths = [base_demo_path] + demos_to_add_to_base_paths
for path_to_demo in all_demo_paths:
    path_to_demo = path_to_demo.expanduser()
    demo = zarr.open(path_to_demo, 'rw+')

    path_to_json = path_to_demo.with_suffix('.json')
    with open(path_to_json, 'r') as f:
        meta_json = json.load(f)
    num_episodes = len(demo.meta.episode_ends)
    assert num_episodes == len(meta_json['episodes'])

    trim_start_and_end_of_trajectories(demo, meta_json, total_action_norm_threshold=.005)

merge_demos_into_base_demo(base_demo_path, demos_to_add_to_base_paths)
#%%
demo = zarr.open(base_demo_path, 'rw+')
meta_json = json.load(open(base_demo_path.with_suffix('.json'), 'r'))

# add some needed meta attrs
max_demo_length = 0
for episode_dict in meta_json['episodes']:
    episode_length = episode_dict['elapsed_steps']
    if episode_length > max_demo_length:
        max_demo_length = episode_length

meta_json['max_demo_length'] = max_demo_length
demo.meta.attrs['max_demo_length'] = max_demo_length
# update the json file
with open(path_to_json, 'w') as f:
    json.dump(meta_json, f, indent=4)

# move the zarr and json file to a directory
# dataset_root_dir = Path('/mnt/crucialSSD/datasetsSSD/fish_datasets/simulated/teleop')
total_num_demos = len(demo.meta.episode_ends)
dataset_name = f'{total_num_demos}_' + dataset_name
dataset_root_dir = base_demo_path.parent

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

#%%
# # create a tmp dataset for the trimmed trajectory
# plt.imshow(demo.data['observation.depth'][275])
# # plt.legend()

# #%%
# key='observation.rgb'
# # trajectory_idx = 0

# images_to_video(
#     images=demo.data[f'{key}'][:],
#     output_dir='./',
#     video_name='merged_trimmed_video',
#     fps=20,
# )

#%%
# # create mask between new_episode_start and new_episode_end
# mask = np.ones_like(demo.data['observation.rgb'][:], dtype=bool)
# mask[episode_start:new_episode_start] = False
# mask[new_episode_end:episode_end] = False
# demo.data['observation.rgb'].vindex[mask].shape
# # del demo.data['observation.rgb'][episode_start:new_episode_start]

# plt.plot(gripper_actions)
# plt.xlim(new_episode_end - 5, new_episode_end + 5)

# plt.plot(action_total_norms)
# # plt.plot(action_translation_norms)
# # plt.plot(action_rotation_norms)
# plt.ylim(-0.01,0.01)
# plt.xlim(-2,50)
# plt.grid()
# plt.show()

# %%
