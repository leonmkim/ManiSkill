import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Union

import gymnasium as gym

import h5py
import zarr
from numcodecs import blosc
import zarr.hierarchy
blosc.set_nthreads(24)
blosc.use_threads = True

import numpy as np
import sapien.physx as physx
import torch

from mani_skill import get_commit_info
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common, gym_utils, sapien_utils
from mani_skill.utils.io_utils import dump_json
from mani_skill.utils.logging_utils import logger
from mani_skill.utils.structs.types import Array
from mani_skill.utils.visualization.misc import (
    images_to_video,
    put_info_on_image,
    tile_images,
)
from mani_skill.utils.wrappers import CPUGymWrapper

# NOTE (stao): The code for record.py is quite messy and perhaps confusing as it is trying to support both recording on CPU and GPU seamlessly
# and handle partial resets. It works but can be claned up a lot.


def parse_env_info(env: gym.Env):
    # spec can be None if not initialized from gymnasium.make
    env = env.unwrapped
    if env.spec is None:
        return None
    if hasattr(env.spec, "_kwargs"):
        # gym<=0.21
        env_kwargs = env.spec._kwargs
    else:
        # gym>=0.22
        env_kwargs = env.spec.kwargs
    return dict(
        env_id=env.spec.id,
        env_kwargs=env_kwargs,
    )


def temp_deep_print_shapes(x, prefix=""):
    if isinstance(x, dict):
        for k in x:
            temp_deep_print_shapes(x[k], prefix=prefix + "/" + k)
    else:
        print(prefix, x.shape)


def clean_trajectories(
        # h5_file: h5py.File, 
        zarr_root: zarr.hierarchy.Group,
        json_dict: dict, prune_empty_action=True
        ):
    """Clean trajectories by renaming and pruning trajectories in place.

    After cleanup, trajectory names are consecutive integers (traj_0, traj_1, ...),
    and trajectories with empty action are pruned.

    Args:
        h5_file: raw h5 file
        json_dict: raw JSON dict
        prune_empty_action: whether to prune trajectories with empty action
    """
    json_episodes = json_dict["episodes"]
    # assert len(h5_file) == len(json_episodes)
    assert len(zarr_root.meta.episode_ends) == len(json_episodes)

    # Assumes each trajectory is named "traj_{i}"
    prefix_length = len("traj_")
    # ep_ids = sorted([int(x[prefix_length:]) for x in h5_file.keys()])
    ep_ids = sorted([int(x[prefix_length:]) for x in zarr_root.meta.ep_ids])

    new_json_episodes = []
    new_ep_id = 0

    for i, ep_id in enumerate(ep_ids):
        traj_id = f"traj_{ep_id}"
        ep = json_episodes[i]
        assert ep["episode_id"] == ep_id
        new_traj_id = f"traj_{new_ep_id}"

        if prune_empty_action and ep["elapsed_steps"] == 0:
            raise NotImplementedError
            # del h5_file[traj_id]
            if i - 1 < 0:
                start = 0
            else:
                start = zarr_root.meta.episode_ends[i-1]
            end = zarr_root.meta.episode_ends[i]
            # del zarr_root.data.observation[start:end]
            # del zarr_root.data.action[start:end]
            # del zarr_root.data.reward[start:end]
            # del zarr_root.data.terminated[start:end]
            # del zarr_root.data.truncated[start:end]
            # del zarr_root.data.done[start:end]
            # if "success" in zarr_root.data:
            #     del zarr_root.data.success[start:end]
            # if "fail" in zarr_root.data:
            #     del zarr_root.data.fail[start:end]
            # del zarr_root.data.env_episode_ptr[i]
            continue

        if new_traj_id != traj_id:
            raise NotImplementedError
            ep["episode_id"] = new_ep_id
            h5_file[new_traj_id] = h5_file[traj_id]
            del h5_file[traj_id]

        new_json_episodes.append(ep)
        new_ep_id += 1

    json_dict["episodes"] = new_json_episodes

@dataclass
class Step:
    state: np.ndarray
    observation: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    done: np.ndarray
    env_episode_ptr: np.ndarray
    """points to index in above data arrays where current episode started (any data before should already be flushed)"""

    success: np.ndarray = None
    fail: np.ndarray = None


class RecordEpisodeZarr(gym.Wrapper):
    """Record trajectories or videos for episodes. You generally should always apply this wrapper last, particularly if you include
    observation wrappers which modify the returned observations. The only wrappers that may go after this one is any of the vector env
    interface wrappers that map the maniskill env to a e.g. gym vector env interface.

    Trajectory data is saved with two files, the actual data in a .h5 file via H5py and metadata in a JSON file of the same basename.

    Each JSON file contains:

    - `env_info` (Dict): task (also known as environment) information, which can be used to initialize the task
    - `env_id` (str): task id
    - `max_episode_steps` (int)
    - `env_kwargs` (Dict): keyword arguments to initialize the task. **Essential to recreate the environment.**
    - `episodes` (List[Dict]): episode information
    - `source_type` (Optional[str]): a simple category string describing what process generated the trajectory data. ManiSkill official datasets will usually write one of "human", "motionplanning", or "rl" at the moment.
    - `source_desc` (Optional[str]): a longer explanation of how the data was generated.

    The episode information (the element of `episodes`) includes:

    - `episode_id` (int): a unique id to index the episode
    - `reset_kwargs` (Dict): keyword arguments to reset the task. **Essential to reproduce the trajectory.**
    - `control_mode` (str): control mode used for the episode.
    - `elapsed_steps` (int): trajectory length
    - `info` (Dict): information at the end of the episode.

    With just the meta data, you can reproduce the task the same way it was created when the trajectories were collected as so:

    ```python
    env = gym.make(env_info["env_id"], **env_info["env_kwargs"])
    episode = env_info["episodes"][0] # picks the first
    env.reset(**episode["reset_kwargs"])
    ```

    Each HDF5 demonstration dataset consists of multiple trajectories. The key of each trajectory is `traj_{episode_id}`, e.g., `traj_0`.

    Each trajectory is an `h5py.Group`, which contains:

    - actions: [T, A], `np.float32`. `T` is the number of transitions.
    - terminated: [T], `np.bool_`. It indicates whether the task is terminated or not at each time step.
    - truncated: [T], `np.bool_`. It indicates whether the task is truncated or not at each time step.
    - env_states: [T+1, D], `np.float32`. Environment states. It can be used to set the environment to a certain state via `env.set_state_dict`. However, it may not be enough to reproduce the trajectory.
    - success (optional): [T], `np.bool_`. It indicates whether the task is successful at each time step. Included if task defines success.
    - fail (optional): [T], `np.bool_`. It indicates whether the task is in a failure state at each time step. Included if task defines failure.
    - obs (optional): [T+1, D] observations.

    Note that env_states is in a dictionary form (and observations may be as well depending on obs_mode), where it is formatted as a dictionary of lists. For example, a typical environment state looks like this:

    ```python
    env_state = env.get_state_dict()
    \"\"\"
    env_state = {
    "actors": {
        "actor_id": [...numpy_actor_state...],
        ...
    },
    "articulations": {
        "articulation_id": [...numpy_articulation_state...],
        ...
    }
    }
    \"\"\"
    ```
    In the trajectory file env_states will be the same structure but each value/leaf in the dictionary will be a sequence of states representing the state of that particular entity in the simulation over time.

    In practice it is may be more useful to use slices of the env_states data (or the observations data), which can be done with

    ```python
    import mani_skill.trajectory.utils as trajectory_utils
    env_states = trajectory_utils.dict_to_list_of_dicts(env_states)
    # now env_states[i] is the same as the data env.get_state_dict() returned at timestep i
    i = 10
    env_state_i = trajectory_utils.index_dict(env_states, i)
    # now env_state_i is the same as the data env.get_state_dict() returned at timestep i
    ```

    Args:
        env: the environment to record
        output_dir: output directory
        save_trajectory: whether to save trajectory
        trajectory_name: name of trajectory file (.h5). Use timestamp if not provided.
        save_video: whether to save video
        info_on_video: whether to write data about reward, action, and data in the info object to the video. The first video frame is generally the result
            of the first env.reset() (visualizing the first observation). Text is written on frames after that, showing the action taken to get to that
            environment state and reward.
        save_on_reset: whether to save the previous trajectory (and video of it if `save_video` is True) automatically when resetting.
            Not that for environments simulated on the GPU (to leverage fast parallel rendering) you must
            set `max_steps_per_video` to a fixed number so that every `max_steps_per_video` steps a video is saved. This is
            required as there may be partial environment resets which makes it ambiguous about how to save/cut videos.
        save_video_trigger: a function that takes the current number of elapsed environment steps and outputs a bool. If output is True, will start saving that timestep to the video.
        max_steps_per_video: how many steps can be recorded into a single video before flushing the video. If None this is not used. A internal step counter is maintained to do this.
            If the video is flushed at any point, the step counter is reset to 0.
        clean_on_close: whether to rename and prune trajectories when closed.
            See `clean_trajectories` for details.
        record_reward: whether to record the reward in the trajectory data
        record_env_state: whether to record the environment state in the trajectory data
        video_fps (int): The FPS of the video to generate if save_video is True
        render_substeps (bool): Whether to render substeps for video. This is captures an image of the environment after each physics step. This runs slower but generates more image frames
            per environment step which when coupled with a higher video FPS can yield a smoother video.
        avoid_overwriting_video (bool): If true, the wrapper will iterate over possible video names to avoid overwriting existing videos in the output directory. Useful for resuming training runs.
        source_type (Optional[str]): a word to describe the source of the actions used to record episodes (e.g. RL, motionplanning, teleoperation)
        source_desc (Optional[str]): A longer description describing how the demonstrations are collected
    """

    def __init__(
        self,
        env: BaseEnv,
        output_dir: str,
        save_trajectory: bool = True,
        trajectory_name: Optional[str] = None,
        save_video: bool = True,
        info_on_video: bool = False,
        save_on_reset: bool = True,
        save_video_trigger: Optional[Callable[[int], bool]] = None,
        max_steps_per_video: Optional[int] = None,
        clean_on_close: bool = True,
        record_reward: bool = True,
        record_env_state: bool = True,
        video_fps: int = 30,
        render_substeps: bool = False,
        avoid_overwriting_video: bool = False,
        source_type: Optional[str] = None,
        source_desc: Optional[str] = None,
        store_in_memory: bool = True,
        zarr_compression_level: int = 3,
        eval_mode: bool = False,
    ) -> None:
        super().__init__(env)
        self.current_env_seed = None
        self.eval_mode = eval_mode
        self.output_dir = Path(output_dir)
        if save_trajectory or save_video:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.video_fps = video_fps
        self._elapsed_record_steps = 0
        self._episode_id = -1
        self._video_id = -1
        self._video_steps = 0
        self._closed = False
        self._store_in_memory = store_in_memory

        self.save_video_trigger = save_video_trigger

        self._trajectory_buffer: Step = None

        self.max_steps_per_video = max_steps_per_video
        self.max_episode_steps = gym_utils.find_max_episode_steps_value(env)

        self.save_on_reset = save_on_reset
        self.save_trajectory = save_trajectory
        if self.base_env.num_envs > 1 and save_video:
            assert (
                max_steps_per_video is not None
            ), "On GPU parallelized environments, \
                there must be a given max steps per video value in order to flush videos in order \
                to avoid issues caused by partial resets. If your environment does not do partial \
                resets you may set max_steps_per_video equal to the max_episode_steps"
        self.clean_on_close = clean_on_close
        self.record_reward = record_reward
        self.record_env_state = record_env_state
        if self.save_trajectory:
            if not trajectory_name:
                trajectory_name = time.strftime("%Y%m%d_%H%M%S")

            # self._h5_file = h5py.File(self.output_dir / f"{trajectory_name}.h5", "w")
            self.filename = str(self.output_dir / f"{trajectory_name}.zarr")
            self.zarr_storage = zarr.DirectoryStore(
                self.filename
            )
            self.zarr_root = zarr.group(store=self.zarr_storage)
            self.data_group = self.zarr_root.create_group("data")
            self.meta_group = self.zarr_root.create_group("meta")

            self.zarr_compressor = blosc.Blosc(cname="zstd", clevel=zarr_compression_level, shuffle=1)

            # self._trajectory_buffer = None

            # Use a separate json to store non-array data
            self._json_path = self.filename.replace(".zarr", ".json")
            self._json_data = dict(
                env_info=parse_env_info(self.env),
                commit_info=get_commit_info(),
                episodes=[],
            )
            if self._json_data["env_info"] is not None:
                self._json_data["env_info"][
                    "max_episode_steps"
                ] = self.max_episode_steps
            if source_type is not None:
                self._json_data["source_type"] = source_type
            if source_desc is not None:
                self._json_data["source_desc"] = source_desc
        self._save_video = save_video
        self.info_on_video = info_on_video
        self.render_images = []
        self.video_nrows = int(np.sqrt(self.unwrapped.num_envs))
        self._avoid_overwriting_video = avoid_overwriting_video

        self._already_warned_about_state_dict_inconsistency = False

        # check if wrapped env is already wrapped by a CPU gym wrapper
        cur_env = self.env
        self.cpu_wrapped_env = False
        while cur_env is not None:
            if isinstance(cur_env, CPUGymWrapper):
                self.cpu_wrapped_env = True
                break
            if hasattr(cur_env, "env"):
                cur_env = cur_env.env
            else:
                break

        self.render_substeps = render_substeps
        if self.render_substeps:
            _original_after_simulation_step = self.base_env._after_simulation_step

            def wrapped_after_simulation_step():
                _original_after_simulation_step()
                if self.save_video:
                    if self.base_env.gpu_sim_enabled:
                        self.base_env.scene._gpu_fetch_all()
                    self.render_images.append(self.capture_image())

            self.base_env._after_simulation_step = wrapped_after_simulation_step

    @property
    def num_envs(self):
        return self.base_env.num_envs

    @property
    def base_env(self) -> BaseEnv:
        return self.env.unwrapped

    @property
    def save_video(self):
        if not self._save_video:
            return False
        if self.save_video_trigger is not None:
            return self.save_video_trigger(self._elapsed_record_steps)
        else:
            return self._save_video

    def capture_image(self, infos=None):
        img = self.env.render()
        img = common.to_numpy(img)
        if len(img.shape) == 3:
            img = img[None]
        if infos is not None:
            for i in range(len(img)):
                info_item = {
                    k: v if np.size(v) == 1 else v[i] for k, v in infos.items()
                }
                img[i] = put_info_on_image(img[i], info_item)
        if len(img.shape) > 3:
            if len(img) == 1:
                img = img[0]
            else:
                img = tile_images(img, nrows=self.video_nrows)
        return img

    def create_new_trajectory_buffer(self):
        trajectory_buffer = zarr.MemoryStore()
        trajectory_root = zarr.group(store=trajectory_buffer)
        trajectory_root.create_group("data")
        return trajectory_root
    
    def create_and_initialize_new_trajectory_buffer(self, obs, action, env_state_dict=None):
        trajectory_buffer = self.create_new_trajectory_buffer()
        data_group = trajectory_buffer["data"]
        meta_group = trajectory_buffer.create_group("meta")
        
        image_obs_shape = obs['sensor_data']['base_camera']['rgb'].shape[2:4]
        data_group.create_dataset('observation.rgb', shape=(0,) + (self.num_envs,) + image_obs_shape + (3,), dtype=np.uint8, chunks=(1,) + (self.num_envs,) + image_obs_shape + (3,), overwrite=True)#, compressor=self.zarr_compressor)
        data_group.create_dataset('observation.depth', shape=(0,) + (self.num_envs,) + image_obs_shape + (1,), dtype=np.uint16, chunks=(1,) + (self.num_envs,) + image_obs_shape + (1,), overwrite=True)#, compressor=self.zarr_compressor)
        data_group.create_dataset('observation.segmentation', shape=(0,) + (self.num_envs,) + image_obs_shape + (1,), dtype=np.uint16, chunks=(1,) + (self.num_envs,) + image_obs_shape + (1,), overwrite=True)#, compressor=self.zarr_compressor)
        data_group.create_dataset('observation.contact_map', shape=(0,) + (self.num_envs,) + image_obs_shape + (1,), dtype=np.float32, chunks=(1,) + (self.num_envs,) + image_obs_shape + (1,), overwrite=True)#, compressor=self.zarr_compressor)
        
        contact_positions_shape = obs['extra']['extrinsic_contact_positions'].shape[2:]
        data_group.create_dataset('observation.contact_positions', shape=(0,) + (self.num_envs,) + contact_positions_shape, dtype=np.float32, chunks=(1,) + (self.num_envs,) + contact_positions_shape, overwrite=True)#, compressor=self.zarr_compressor)
        
        current_pose_shape = obs['extra']['end_effector_pose'].shape[2:]
        state_shape = list(current_pose_shape)
        state_shape[0] += 1 # for gripper width
        state_shape = tuple(state_shape)
        data_group.create_dataset('observation.state', shape=(0,) + (self.num_envs,) + state_shape, dtype=np.float32, chunks=(1,) + (self.num_envs,) + state_shape, overwrite=True)#, compressor=self.zarr_compressor)
        data_group.create_dataset('observation.EE_pixel_coord', shape=(0,) + (self.num_envs,) + (2,), dtype=np.float32, chunks=(1,) + (self.num_envs,) + (2,), overwrite=True)#, compressor=self.zarr_compressor)
        
        target_pose_shape = obs['agent']['controller']['arm']['target_pose'].shape[2:]
        data_group.create_dataset('observation.target_pose', shape=(0,) + (self.num_envs,) + target_pose_shape, dtype=np.float32, chunks=(1,) + (self.num_envs,) + target_pose_shape, overwrite=True)#, compressor=self.zarr_compressor)

        # the normalized action as accepted by the maniskill controller
        action_shape = action.shape[2:]
        data_group.create_dataset('action', shape=(0,) + (self.num_envs,) + action_shape, dtype=np.float32, chunks=(1,) + (self.num_envs,) + action_shape, overwrite=True)#, compressor=self.zarr_compressor)

        if self.record_reward:
            data_group.create_dataset('reward', shape=(0,) + (self.num_envs,), dtype=np.float32, chunks=(1,) + (self.num_envs,), overwrite=True)#, compressor=self.zarr_compressor)
    
        data_group.create_dataset('terminated', shape=(0,) + (self.num_envs,), dtype=bool, chunks=(1,) + (self.num_envs,), overwrite=True)#, compressor=self.zarr_compressor)
        data_group.create_dataset('truncated', shape=(0,) + (self.num_envs,), dtype=bool, chunks=(1,) + (self.num_envs,), overwrite=True)#, compressor=self.zarr_compressor)
        meta_group.create_dataset('done', shape=(0,) + (self.num_envs,), dtype=bool, chunks=(1,) + (self.num_envs,), overwrite=True)#, compressor=self.zarr_compressor)

        data_group.create_dataset('success', shape=(0,) + (self.num_envs,), dtype=bool, chunks=(1,) + (self.num_envs,), overwrite=True)#, compressor=self.zarr_compressor)
        data_group.create_dataset('fail', shape=(0,) + (self.num_envs,), dtype=bool, chunks=(1,) + (self.num_envs,), overwrite=True)#, compressor=self.zarr_compressor)

        meta_group.create_dataset('env_episode_ptr', shape=(0,), dtype=np.int32, overwrite=True)#, compressor=self.zarr_compressor)
        meta_group['env_episode_ptr'].append(np.zeros((self.num_envs), dtype=np.int32))

        meta_group.create_dataset('episode_cam_K', shape=(0,) + (self.num_envs,) + (3, 3), dtype=np.float32, chunks=(1,) + (self.num_envs,) + (3, 3), overwrite=True)#, compressor=self.zarr_compressor)
        meta_group['episode_cam_K'].append(obs['sensor_param']['base_camera']['intrinsic_cv'])

        meta_group.create_dataset('episode_cam_tf_world', shape=(0,) + (self.num_envs,) + (4, 4), dtype=np.float32, chunks=(1,) + (self.num_envs,) + (4, 4), overwrite=True)#, compressor=self.zarr_compressor)
        cam_tf_world = obs['sensor_param']['base_camera']['extrinsic_cv'] # bxNx3x4 where N is number of cameras?
        # need to add the last row for the homogeneous coordinates
        cam_tf_world = np.concatenate([cam_tf_world, np.zeros((1, self.num_envs, 1, 4), dtype=np.float32)], axis=2)
        cam_tf_world[:, :, 3, 3] = 1.0
        meta_group['episode_cam_tf_world'].append(cam_tf_world)

        if env_state_dict is not None:
            self.recursive_add_env_states_to_trajectory_buffer(data_group, env_state_dict)

        self.add_to_existing_trajectory_buffer(trajectory_buffer, obs, action)

        return trajectory_buffer
    
    def add_to_existing_trajectory_buffer(self, trajectory_buffer, obs, action, reward=None, terminated=None, truncated=None, done=None, success=None, fail=None, env_state_dict=None):
        trajectory_buffer_data_group = trajectory_buffer['data']
        trajectory_buffer_meta_group = trajectory_buffer['meta']

        trajectory_buffer_data_group['observation.rgb'].append(obs['sensor_data']['base_camera']['rgb'])
        trajectory_buffer_data_group['observation.depth'].append(obs['sensor_data']['base_camera']['depth'])
        trajectory_buffer_data_group['observation.segmentation'].append(obs['sensor_data']['base_camera']['segmentation'])
        trajectory_buffer_data_group['observation.contact_map'].append(obs['extra']['extrinsic_contact_map'])
        trajectory_buffer_data_group['observation.contact_positions'].append(obs['extra']['extrinsic_contact_positions'])

        current_pose = obs['extra']['end_effector_pose']
        gripper_width = np.sum(obs['agent']['qpos'][:,:,-2:], axis=-1, keepdims=True)
        state = np.concatenate([current_pose, gripper_width], axis=-1)
        trajectory_buffer_data_group['observation.state'].append(state)

        trajectory_buffer_data_group['observation.target_pose'].append(obs['agent']['controller']['arm']['target_pose'])
        trajectory_buffer_data_group['observation.EE_pixel_coord'].append(obs['extra']['end_effector_pixel_coordinates'])

        trajectory_buffer_data_group['action'].append(action)
        if env_state_dict is not None:
            self.recursive_add_env_states_to_trajectory_buffer(trajectory_buffer_data_group, env_state_dict)

        if self.record_reward:
            if reward is None:
                reward = np.zeros((1, self.num_envs), dtype=np.float32)
            trajectory_buffer_data_group['reward'].append(reward)
        
        if terminated is None:
            terminated = np.ones((1, self.num_envs), dtype=bool)
        trajectory_buffer_data_group['terminated'].append(terminated)
        if truncated is None:
            truncated = np.ones((1, self.num_envs), dtype=bool)
        trajectory_buffer_data_group['truncated'].append(truncated)
        
        if done is None:
            done = np.ones((1, self.num_envs), dtype=bool)
        trajectory_buffer_meta_group['done'].append(done)
        
        if 'success' in trajectory_buffer_data_group:
            if success is None:
                success = np.zeros((1, self.num_envs), dtype=bool)
            trajectory_buffer_data_group['success'].append(np.zeros((1, self.num_envs), dtype=bool))
    
        if 'fail' in trajectory_buffer_data_group:
            if fail is None:
                fail = np.zeros((1, self.num_envs), dtype=bool)
            trajectory_buffer_data_group['fail'].append(fail)

    def recursive_add_env_states_to_trajectory_buffer(self, trajectory_buffer_data_group, env_states_dict):
        # if 'env_states' not in trajectory_buffer_data_group:
        #     trajectory_buffer_data_group.create_group('env_states')
        # trajectory_buffer_data_group = trajectory_buffer_data_group['env_states']
        for k, v in env_states_dict.items():
            if isinstance(v, dict):
                if k not in trajectory_buffer_data_group:
                    trajectory_buffer_data_group.create_group(k)
                self.recursive_add_env_states_to_trajectory_buffer(trajectory_buffer_data_group[k], v)
            else:
                # create if it doesn't exist
                if k not in trajectory_buffer_data_group:
                    trajectory_buffer_data_group.create_dataset(k, shape=(0,) + (self.num_envs,) + v.shape[2:], dtype=v.dtype, chunks=(1,) + (self.num_envs,) + v.shape[2:], overwrite=True)#, compressor=self.zarr_compressor)
                trajectory_buffer_data_group[k].append(v)

    def reset(
        self,
        *args,
        seed: Optional[Union[int, List[int]]] = None,
        options: Optional[dict] = dict(),
        **kwargs,
    ):

        if self.save_on_reset:
            if self.save_video and self.num_envs == 1:
                self.flush_video()
            # if doing a full reset then we flush all trajectories including incompleted ones
            if self._trajectory_buffer is not None:
            # if we store in memory and its not empty, then we need to write it to disk
                if "env_idx" not in options:
                    self.flush_trajectory(env_idxs_to_flush=np.arange(self.num_envs), save=options.get("save_trajectory", self.save_trajectory))
                else:
                    self.flush_trajectory(
                        env_idxs_to_flush=common.to_numpy(options["env_idx"]),
                        save=options.get("save_trajectory", self.save_trajectory)
                    )
        if self.eval_mode:
            self.current_env_seed = seed

        obs, info = super().reset(*args, seed=seed, options=options, **kwargs)
        if info["reconfigure"]:
            # if we reconfigure, there is the possibility that state dictionary looks different now
            # so trajectory buffer must be wiped
            self._trajectory_buffer = None

        if self.save_trajectory:
            # adding the "batch" dimension as is done by common.batch is creating the temporal dimension such that each array is Txbx(data dims), where b is num envs
            state_dict = self.base_env.get_state_dict()
            action = common.batch(
                self.env.get_wrapper_attr("single_action_space").sample()
            )
            # check if state_dict is consistent
            if not sapien_utils.is_state_dict_consistent(state_dict):
                self.record_env_state = False
                if not self._already_warned_about_state_dict_inconsistency:
                    logger.warn(
                        f"State dictionary is not consistent, disabling recording of environment states for {self.env}"
                    )
                    self._already_warned_about_state_dict_inconsistency = True
            # first_step = Step(
            #     state=None,
            #     observation=common.to_numpy(common.batch(obs)),
            #     # note first reward/action etc. are ignored when saving trajectories to disk
            #     action=common.to_numpy(common.batch(action.repeat(self.num_envs, 0))),
            #     reward=np.zeros(
            #         (
            #             1,
            #             self.num_envs,
            #         ),
            #         dtype=float,
            #     ),
            #     # terminated and truncated are fixed to be True at the start to indicate the start of an episode.
            #     # an episode is done when one of these is True otherwise the trajectory is incomplete / a partial episode
            #     terminated=np.ones((1, self.num_envs), dtype=bool),
            #     truncated=np.ones((1, self.num_envs), dtype=bool),
            #     done=np.ones((1, self.num_envs), dtype=bool),
            #     success=np.zeros((1, self.num_envs), dtype=bool),
            #     fail=np.zeros((1, self.num_envs), dtype=bool),
            #     env_episode_ptr=np.zeros((self.num_envs,), dtype=int),
            # )
            # if self.record_env_state:
                # first_step.state = common.to_numpy(common.batch(state_dict))
            if self.record_env_state:
                state_dict = common.to_numpy(common.batch(state_dict))
            else:
                state_dict = None
            
            env_idx = np.arange(self.num_envs)
            if "env_idx" in options:
                env_idx = common.to_numpy(options["env_idx"])
            if self._trajectory_buffer is None: # create and add to the trajectory buffer
                # Initialize trajectory buffer on the first episode based on given observation (which should be generated after all wrappers)
                # self._trajectory_buffer = first_step
                self._trajectory_buffer = self.create_and_initialize_new_trajectory_buffer(common.to_numpy(common.batch(obs)), common.to_numpy(common.batch(action.repeat(self.num_envs, 0))), env_state_dict=state_dict)
            
            else: # add to the existing trajectory buffer

                # def recursive_replace(x, y):
                #     if isinstance(x, np.ndarray):
                #         x[-1, env_idx] = y[-1, env_idx]
                #     else:
                #         for k in x.keys():
                #             recursive_replace(x[k], y[k])

                # # TODO (stao): how do we store states from GPU sim of tasks with objects not in every sub-scene?
                # # Maybe we shouldn't?
                # if self.record_env_state:
                #     recursive_replace(self._trajectory_buffer.state, first_step.state)
                # recursive_replace(
                #     self._trajectory_buffer.observation, first_step.observation
                # )
                # recursive_replace(self._trajectory_buffer.action, first_step.action)
                # if self.record_reward:
                #     recursive_replace(self._trajectory_buffer.reward, first_step.reward)
                # recursive_replace(
                #     self._trajectory_buffer.terminated, first_step.terminated
                # )
                # recursive_replace(
                #     self._trajectory_buffer.truncated, first_step.truncated
                # )
                # recursive_replace(self._trajectory_buffer.done, first_step.done)
                # if self._trajectory_buffer.success is not None:
                #     recursive_replace(
                #         self._trajectory_buffer.success, first_step.success
                #     )
                # if self._trajectory_buffer.fail is not None:
                #     recursive_replace(self._trajectory_buffer.fail, first_step.fail)
                self.add_to_existing_trajectory_buffer(self._trajectory_buffer, common.to_numpy(common.batch(obs)), common.to_numpy(common.batch(action)), env_state_dict=state_dict)

        if options is not None and "env_idx" in options:
            options["env_idx"] = common.to_numpy(options["env_idx"])
        self.last_reset_kwargs = copy.deepcopy(dict(options=options, **kwargs))
        if seed is not None:
            self.last_reset_kwargs.update(seed=seed)
        return obs, info

    def step(self, action):
        if self.save_video and self._video_steps == 0:
            # save the first frame of the video here (s_0) instead of inside reset as user
            # may call env.reset(...) multiple times but we want to ignore empty trajectories
            self.render_images.append(self.capture_image())
        obs, rew, terminated, truncated, info = super().step(action)

        if self.save_trajectory:
            state_dict = self.base_env.get_state_dict()
            if self.record_env_state:
                # self._trajectory_buffer.state = common.append_dict_array(
                #     self._trajectory_buffer.state,
                #     common.to_numpy(common.batch(state_dict)),
                # )
                state_dict = common.to_numpy(common.batch(state_dict))
            else:
                state_dict = None
            # self._trajectory_buffer.observation = common.append_dict_array(
            #     self._trajectory_buffer.observation,
            #     common.to_numpy(common.batch(obs)),
            # )

            # self._trajectory_buffer.action = common.append_dict_array(
            #     self._trajectory_buffer.action,
            #     common.to_numpy(common.batch(action)),
            # )
            if self.record_reward:
            #     self._trajectory_buffer.reward = common.append_dict_array(
            #         self._trajectory_buffer.reward,
            #         common.to_numpy(common.batch(rew)),
            #     )
                rew = common.to_numpy(common.batch(rew))
            else:
                rew = None
            # self._trajectory_buffer.terminated = common.append_dict_array(
            #     self._trajectory_buffer.terminated,
            #     common.to_numpy(common.batch(terminated)),
            # )
            # self._trajectory_buffer.truncated = common.append_dict_array(
            #     self._trajectory_buffer.truncated,
            #     common.to_numpy(common.batch(truncated)),
            # )
            done = terminated | truncated
            done = common.to_numpy(common.batch(done))
            # self._trajectory_buffer.done = common.append_dict_array(
            #     self._trajectory_buffer.done,
            #     common.to_numpy(common.batch(done)),
            # )
            if "success" in info:
                # self._trajectory_buffer.success = common.append_dict_array(
                #     self._trajectory_buffer.success,
                #     common.to_numpy(common.batch(info["success"])),
                # )
                success = common.to_numpy(common.batch(info["success"]))
            else:
                # self._trajectory_buffer.success = None
                if 'success' in self._trajectory_buffer["data"]:
                    del self._trajectory_buffer["data"]['success']
                success = None

            if "fail" in info:
                # self._trajectory_buffer.fail = common.append_dict_array(
                #     self._trajectory_buffer.fail,
                #     common.to_numpy(common.batch(info["fail"])),
                # )
                fail = common.to_numpy(common.batch(info["fail"]))
            else:
                # self._trajectory_buffer.fail = None
                if 'fail' in self._trajectory_buffer["data"]:
                    del self._trajectory_buffer["data"]['fail']
                fail=None

            self.add_to_existing_trajectory_buffer(self._trajectory_buffer, common.to_numpy(common.batch(obs)), common.to_numpy(common.batch(common.batch(action))), reward=rew,
                                                   terminated=common.to_numpy(common.batch(terminated)), truncated=common.to_numpy(common.batch(truncated)), done=done, 
                                                   success=success, fail=fail, env_state_dict=state_dict)

        if self.save_video:
            self._video_steps += 1
            if self.info_on_video:
                scalar_info = gym_utils.extract_scalars_from_info(
                    common.to_numpy(info), batch_size=self.num_envs
                )
                scalar_info["reward"] = common.to_numpy(rew)
                if np.size(scalar_info["reward"]) > 1:
                    scalar_info["reward"] = [
                        float(rew) for rew in scalar_info["reward"]
                    ]
                else:
                    scalar_info["reward"] = float(scalar_info["reward"])
                image = self.capture_image(scalar_info)
            else:
                image = self.capture_image()

            self.render_images.append(image)
            if (
                self.max_steps_per_video is not None
                and self._video_steps >= self.max_steps_per_video
            ):
                self.flush_video()
        self._elapsed_record_steps += 1
        return obs, rew, terminated, truncated, info
    
    def move_zarr_array_to_new_group(self, old_group: zarr.hierarchy.Group, new_group: zarr.hierarchy.Group, key):
        new_group.create_dataset(key, data=old_group[key][:], shape=old_group[key].shape, dtype=old_group[key].dtype, chunks=old_group[key].chunks, overwrite=True)
        del old_group[key]

    def extract_individual_segmentation_masks(
            self, trajectory_data_buffer, segmentation_id_map,
    ):
        assert 'observation.segmentation' in trajectory_data_buffer, "Segmentation masks not found in trajectory data buffer"
        image_shape = trajectory_data_buffer['observation.segmentation'].shape[2:]
        
        if 'observation.EE_obj_mask' not in trajectory_data_buffer:
            trajectory_data_buffer.create_dataset('observation.EE_obj_mask', shape=(0,) + (self.num_envs,) + image_shape, dtype=np.uint8, chunks=(1,) + (self.num_envs,) + image_shape, overwrite=True)#, compressor=self.zarr_compressor)

        EE_obj_mask = (trajectory_data_buffer['observation.segmentation'][:] == segmentation_id_map[f"{self.env.grasped_book.name}_0"]).astype(np.uint8)
        trajectory_data_buffer['observation.EE_obj_mask'].append(EE_obj_mask)

    def flush_trajectory(
        self,
        verbose=False,
        ignore_empty_transition=True,
        env_idxs_to_flush=None,
        save: bool = True,
    ):
        """
        Flushes a trajectory and by default saves it to disk

        Arguments:
            verbose (bool): whether to print out information about the flushed trajectory
            ignore_empty_transition (bool): whether to ignore trajectories that did not have any actions
            env_idxs_to_flush: which environments by id to flush. If None, all environments are flushed.
            save (bool): whether to save the trajectory to disk
        """
        flush_count = 0
        if env_idxs_to_flush is None:
            env_idxs_to_flush = np.arange(0, self.num_envs)
        for env_idx in env_idxs_to_flush:
            # start_ptr = self._trajectory_buffer.env_episode_ptr[env_idx]
            start_ptr = self._trajectory_buffer.meta['env_episode_ptr'][env_idx]
            # end_ptr = len(self._trajectory_buffer.done)
            end_ptr = len(self._trajectory_buffer.meta['done'])
            if ignore_empty_transition and end_ptr - start_ptr <= 1:
                continue
            flush_count += 1
            if save:
                self._episode_id += 1
                traj_id = "traj_{}".format(self._episode_id)
                # group = self._h5_file.create_group(traj_id, track_order=True)
                if 'ep_ids' not in self.meta_group:
                    self.meta_group.create_dataset('ep_ids', shape=(0,), dtype='S256', chunks=(1,), overwrite=True)
                self.meta_group['ep_ids'].append(np.array([traj_id], dtype='S256'))

                # def recursive_add_to_h5py(
                #     group: h5py.Group, data: Union[dict, Array], key
                # ):
                #     """simple recursive data insertion for nested data structures into h5py, optimizing for visual data as well"""
                #     if isinstance(data, dict):
                #         subgrp = group.create_group(key, track_order=True)
                #         for k in data.keys():
                #             recursive_add_to_h5py(subgrp, data[k], k)
                #     else:
                #         if key == "rgb":
                #             # NOTE(jigu): It is more efficient to use gzip than png for a sequence of images.
                #             group.create_dataset(
                #                 "rgb",
                #                 data=data[start_ptr:end_ptr, env_idx],
                #                 dtype=data.dtype,
                #                 compression="gzip",
                #                 compression_opts=5,
                #             )
                #         elif key == "depth":
                #             # NOTE (stao): By default now cameras in ManiSkill return depth values of type uint16 for numpy
                #             group.create_dataset(
                #                 key,
                #                 data=data[start_ptr:end_ptr, env_idx],
                #                 dtype=data.dtype,
                #                 compression="gzip",
                #                 compression_opts=5,
                #             )
                #         elif key == "seg":
                #             group.create_dataset(
                #                 key,
                #                 data=data[start_ptr:end_ptr, env_idx],
                #                 dtype=data.dtype,
                #                 compression="gzip",
                #                 compression_opts=5,
                #             )
                #         else:
                #             group.create_dataset(
                #                 key,
                #                 data=data[start_ptr:end_ptr, env_idx],
                #                 dtype=data.dtype,
                #             )

                def recursive_copy_memory_store_to_disk(
                        data: Union[zarr.hierarchy.Group, zarr.core.Array], 
                        disk_group: zarr.hierarchy.Group, 
                        key: str,
                        env_idx: int,
                ):
                    if isinstance(data, zarr.hierarchy.Group):
                        if key not in disk_group:
                            disk_group.create_group(key)
                        for k in data.keys():
                            recursive_copy_memory_store_to_disk(data[k], disk_group[key], k, env_idx)
                    elif isinstance(data, zarr.core.Array):
                        if key not in disk_group:
                            disk_group.create_dataset(key, shape=(0,) + data.shape[2:], dtype=data.dtype, chunks=(1,) + data.shape[2:], overwrite=True, compressor=self.zarr_compressor)
                        disk_group[key].append(data[start_ptr:end_ptr, env_idx])
                    else:
                        raise ValueError(f"Data type {type(data)} not supported for recursive_copy_memory_store_to_disk")

                # # Observations need special processing
                # if isinstance(self._trajectory_buffer.observation, dict):
                #     recursive_add_to_h5py(
                #         group, self._trajectory_buffer.observation, "obs"
                #     )
                # elif isinstance(self._trajectory_buffer.observation, np.ndarray):
                #     if self.cpu_wrapped_env:
                #         group.create_dataset(
                #             "obs",
                #             data=self._trajectory_buffer.observation[start_ptr:end_ptr],
                #             dtype=self._trajectory_buffer.observation.dtype,
                #         )
                #     else:
                #         group.create_dataset(
                #             "obs",
                #             data=self._trajectory_buffer.observation[
                #                 start_ptr:end_ptr, env_idx
                #             ],
                #             dtype=self._trajectory_buffer.observation.dtype,
                #         )
                # else:
                #     raise NotImplementedError(
                #         f"RecordEpisode wrapper does not know how to handle observation data of type {type(self._trajectory_buffer.observation)}"
                #     )
                episode_info = dict(
                    episode_id=self._episode_id,
                    episode_seed=self.base_env._episode_seed[env_idx],
                    control_mode=self.base_env.control_mode,
                    elapsed_steps=end_ptr - start_ptr - 1,
                )

                segmentation_id_map = dict()
                for key, value in self.env.segmentation_id_map.items():
                    entity_name = value.name
                    segmentation_id_map[entity_name] = key

                episode_info.update(segmentation_id_map=segmentation_id_map)

                if self.num_envs == 1:
                    episode_info.update(reset_kwargs=self.last_reset_kwargs)
                else:
                    # NOTE (stao): With multiple envs in GPU simulation, reset_kwargs do not make much sense
                    episode_info.update(reset_kwargs=dict())

                # slice some data to remove the first dummy frame.
                # actions = common.index_dict_array(
                #     self._trajectory_buffer.action,
                #     (slice(start_ptr + 1, end_ptr), env_idx),
                # )

                self._trajectory_buffer.data['action'] = self._trajectory_buffer.data['action'][start_ptr + 1 : end_ptr]
                # terminated = self._trajectory_buffer.terminated[
                #     start_ptr + 1 : end_ptr, env_idx
                # ]
                self._trajectory_buffer.data['terminated'] = self._trajectory_buffer.data['terminated'][start_ptr + 1 : end_ptr]
                # truncated = self._trajectory_buffer.truncated[
                #     start_ptr + 1 : end_ptr, env_idx
                # ]
                self._trajectory_buffer.data['truncated'] = self._trajectory_buffer.data['truncated'][start_ptr + 1 : end_ptr]
                # if isinstance(self._trajectory_buffer.action, dict):
                #     recursive_add_to_h5py(group, actions, "actions")
                # else:
                #     group.create_dataset("actions", data=actions, dtype=np.float32)
                # group.create_dataset("terminated", data=terminated, dtype=bool)
                # group.create_dataset("truncated", data=truncated, dtype=bool)

                # if self._trajectory_buffer.success is not None:
                #     group.create_dataset(
                #         "success",
                #         data=self._trajectory_buffer.success[
                #             start_ptr + 1 : end_ptr, env_idx
                #         ],
                #         dtype=bool,
                #     )
                #     episode_info.update(
                #         success=self._trajectory_buffer.success[end_ptr - 1, env_idx]
                #     )
                if 'success' in self._trajectory_buffer.data:
                    self._trajectory_buffer.data['success'] = self._trajectory_buffer.data['success'][start_ptr + 1 : end_ptr]
                    episode_info.update(
                        success=self._trajectory_buffer.data['success'][end_ptr - 1]
                    )
                # if self._trajectory_buffer.fail is not None:
                #     group.create_dataset(
                #         "fail",
                #         data=self._trajectory_buffer.fail[
                #             start_ptr + 1 : end_ptr, env_idx
                #         ],
                #         dtype=bool,
                #     )
                #     episode_info.update(
                #         fail=self._trajectory_buffer.fail[end_ptr - 1, env_idx]
                #     )
                if 'fail' in self._trajectory_buffer.data:
                    self._trajectory_buffer.data['fail'] = self._trajectory_buffer.data['fail'][start_ptr + 1 : end_ptr]
                    episode_info.update(
                        fail=self._trajectory_buffer.data['fail'][end_ptr - 1, env_idx]
                    )
                # if self.record_env_state:
                #     recursive_add_to_h5py(
                #         group, self._trajectory_buffer.state, "env_states"
                #     )
                # if self.record_reward:
                #     group.create_dataset(
                #         "rewards",
                #         data=self._trajectory_buffer.reward[
                #             start_ptr + 1 : end_ptr, env_idx
                #         ],
                #         dtype=np.float32,
                #     )
                if self.record_reward and 'reward' in self._trajectory_buffer.data:
                    self._trajectory_buffer.data['reward'] = self._trajectory_buffer.data['reward'][start_ptr + 1 : end_ptr]

                # should trim the final observation/state as there is no corresponding action that follows from it.
                # get all keys that start with observation.
                observation_keys = [k for k in self._trajectory_buffer.data.keys() if k.startswith('observation.')]
                for k in observation_keys:
                    self._trajectory_buffer.data[k] = self._trajectory_buffer.data[k][start_ptr:end_ptr-1]

                # move contact features to new 'gt_contact' group
                if 'gt_contact' not in self._trajectory_buffer.data:
                    self._trajectory_buffer.data.create_group('gt_contact')
                self.move_zarr_array_to_new_group(self._trajectory_buffer.data, self._trajectory_buffer.data['gt_contact'], 'observation.contact_map')
                self.move_zarr_array_to_new_group(self._trajectory_buffer.data, self._trajectory_buffer.data['gt_contact'], 'observation.contact_positions')

                # move segmentation masks to new 'gt_segmentation' group
                if 'gt_segmentation' not in self._trajectory_buffer.data:
                    self._trajectory_buffer.data.create_group('gt_segmentation')
                self.move_zarr_array_to_new_group(self._trajectory_buffer.data, self._trajectory_buffer.data['gt_segmentation'], 'observation.segmentation')

                # post-process segmentation mask
                self.extract_individual_segmentation_masks(self._trajectory_buffer.data.gt_segmentation, segmentation_id_map)

                recursive_copy_memory_store_to_disk(self._trajectory_buffer["data"], self.zarr_root, 'data', env_idx)

                if 'episode_ends' not in self.meta_group:
                    self.meta_group.create_dataset('episode_ends', shape=(0,), dtype=np.int32, chunks=(1,), overwrite=True)
                prev_episode_end = 0
                if len(self.meta_group['episode_ends']) > 0:
                    prev_episode_end = self.meta_group['episode_ends'][-1]
                self.meta_group['episode_ends'].append(np.array([end_ptr - start_ptr - 1 + prev_episode_end], dtype=np.int32))

                if 'episode_cam_K' not in self.meta_group:
                    self.meta_group.create_dataset('episode_cam_K', shape=(0,) + (3, 3), dtype=np.float32, chunks=(1,) + (3, 3), overwrite=True)
                self.meta_group['episode_cam_K'].append(self._trajectory_buffer.meta['episode_cam_K'][env_idx])

                if 'episode_cam_tf_world' not in self.meta_group:
                    self.meta_group.create_dataset('episode_cam_tf_world', shape=(0,) + (4, 4), dtype=np.float32, chunks=(1,) + (4, 4), overwrite=True)
                self.meta_group['episode_cam_tf_world'].append(self._trajectory_buffer.meta['episode_cam_tf_world'][env_idx])

                # NOTE: done and env states are not trimmed

                self._json_data["episodes"].append(common.to_numpy(episode_info))
                dump_json(self._json_path, self._json_data, indent=2)
                if verbose:
                    if flush_count == 1:
                        print(f"Recorded episode {self._episode_id}")
                    else:
                        print(
                            f"Recorded episodes {self._episode_id - flush_count} to {self._episode_id}"
                        )

        # truncate self._trajectory_buffer down to save memory
        if flush_count > 0:
            # TODO: go back and double check logic. 
            # I'm being lazy by just setting trajectory buffer back to None so that next reset step doesn't try to write a partial episode with the remnants of the previous episode (done and env states should have one remaining element as they are never trimmed)
            
            # # self._trajectory_buffer.env_episode_ptr[env_idxs_to_flush] = (
            # #     len(self._trajectory_buffer.done) - 1
            # # )
            # self._trajectory_buffer.meta['env_episode_ptr'][env_idxs_to_flush] = len(self._trajectory_buffer.meta.done) - 1
            # # min_env_ptr = self._trajectory_buffer.env_episode_ptr.min()
            # min_env_ptr = np.min(self._trajectory_buffer.meta['env_episode_ptr'])
            # # N = len(self._trajectory_buffer.done)
            # N = len(self._trajectory_buffer.meta.done)

            # # if self.record_env_state:
            # #     self._trajectory_buffer.state = common.index_dict_array(
            # #         self._trajectory_buffer.state, slice(min_env_ptr, N)
            # #     )
            # # self._trajectory_buffer.observation = common.index_dict_array(
            # #     self._trajectory_buffer.observation, slice(min_env_ptr, N)
            # # )
            # # self._trajectory_buffer.action = common.index_dict_array(
            # #     self._trajectory_buffer.action, slice(min_env_ptr, N)
            # # )
            # # if self.record_reward:
            # #     self._trajectory_buffer.reward = common.index_dict_array(
            # #         self._trajectory_buffer.reward, slice(min_env_ptr, N)
            # #     )
            # # self._trajectory_buffer.terminated = common.index_dict_array(
            # #     self._trajectory_buffer.terminated, slice(min_env_ptr, N)
            # # )
            # # self._trajectory_buffer.truncated = common.index_dict_array(
            # #     self._trajectory_buffer.truncated, slice(min_env_ptr, N)
            # # )
            # # self._trajectory_buffer.done = common.index_dict_array(
            # #     self._trajectory_buffer.done, slice(min_env_ptr, N)
            # # )
            # # if self._trajectory_buffer.success is not None:
            # #     self._trajectory_buffer.success = common.index_dict_array(
            # #         self._trajectory_buffer.success, slice(min_env_ptr, N)
            # #     )
            # # if self._trajectory_buffer.fail is not None:
            # #     self._trajectory_buffer.fail = common.index_dict_array(
            # #         self._trajectory_buffer.fail, slice(min_env_ptr, N)
            # #     )
            # def recursive_flush_trajectory_buffer(
            #         trajectory_data_group: zarr.hierarchy.Group, 
            #         min_env_ptr: int,
            #         N: int
            # ):
            #     for k, v in trajectory_data_group.items():
            #         if isinstance(v, zarr.hierarchy.Group):
            #             recursive_flush_trajectory_buffer(v, min_env_ptr, N)
            #         else:
            #             trajectory_data_group[k] = v[min_env_ptr:N]

            # recursive_flush_trajectory_buffer(self._trajectory_buffer["data"], min_env_ptr, N)
            # # self._trajectory_buffer.env_episode_ptr -= min_env_ptr
            # self._trajectory_buffer.meta['env_episode_ptr'] -= min_env_ptr

            self._trajectory_buffer = None

    def flush_video(
        self,
        name=None,
        suffix="",
        verbose=False,
        ignore_empty_transition=True,
        save: bool = True,
    ):
        """
        Flush a video of the recorded episode(s) anb by default saves it to disk

        Arguments:
            name (str): name of the video file. If None, it will be named with the episode id.
            suffix (str): suffix to add to the video file name
            verbose (bool): whether to print out information about the flushed video
            ignore_empty_transition (bool): whether to ignore trajectories that did not have any actions
            save (bool): whether to save the video to disk
        """
        if len(self.render_images) == 0:
            return
        if ignore_empty_transition and len(self.render_images) == 1:
            return
        if save:
            self._video_id += 1
            if name is None:
                if self.eval_mode:
                    # use the current env seed as the video name
                    video_name = f"{self._video_id}_traj_{self.current_env_seed}_envseed"
                else:
                    video_name = "{}".format(self._video_id)
                if suffix:
                    video_name += "_" + suffix
                if self._avoid_overwriting_video:
                    while (
                        Path(self.output_dir)
                        / (video_name.replace(" ", "_").replace("\n", "_") + ".mp4")
                    ).exists():
                        self._video_id += 1
                        video_name = "{}".format(self._video_id)
                        if suffix:
                            video_name += "_" + suffix
            else:
                video_name = name
            images_to_video(
                self.render_images,
                str(self.output_dir),
                video_name=video_name,
                fps=self.video_fps,
                verbose=verbose,
            )
        self._video_steps = 0
        self.render_images = []

    def close(self) -> None:
        if self._closed:
            # There is some strange bug when vector envs using record wrapper are closed/deleted, this code runs twice
            return
        self._closed = True
        if self.save_trajectory:
            # Handle the last episode only when `save_on_reset=True`
            if self.save_on_reset and self._trajectory_buffer is not None:
                self.flush_trajectory(
                    ignore_empty_transition=True,
                    env_idxs_to_flush=np.arange(self.num_envs),
                )
            if self.clean_on_close:
                # TODO: implement clean_trajectories for zarr
                clean_trajectories(self.zarr_root, self._json_data)
                dump_json(self._json_path, self._json_data, indent=2)
            # self._h5_file.close()
        if self.save_video:
            if self.save_on_reset:
                self.flush_video()
        return super().close()
