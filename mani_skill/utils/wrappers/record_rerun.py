#%%
from pathlib import Path
path_to_workspace_root = Path(__file__).resolve().parents[4]

# self.franka_urdf_logger = URDFLogger(franka_urdf_path, root_path="world/robot/")
path_to_rerun = path_to_workspace_root / 'rerun'
import sys
sys.path.append(str(path_to_rerun))
from rerun_loader_urdf import URDFLogger
#%%
import copy
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import gymnasium as gym

import numpy as np
import sapien.physx as physx
import torch

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
import uuid

from dataclasses import is_dataclass, asdict

import rerun as rr
import open3d as o3d
from open3d.camera import PinholeCameraIntrinsic
from scipy.spatial.transform import Rotation as R
from rerun import Boxes3D
import distinctipy
import matplotlib.pyplot as plt
import cmasher as cmr
import cv2

# NOTE (stao): The code for record.py is quite messy and perhaps confusing as it is trying to support both recording on CPU and GPU seamlessly
# and handle partial resets. It works but can be claned up a lot.
def recursive_dataclass_to_dict(obj):
    # traverse dict and convert any dataclass to dict
    if isinstance(obj, dict):
        return {k: recursive_dataclass_to_dict(v) for k, v in obj.items()}
    elif is_dataclass(obj):
        obj_dict = asdict(obj)
        return {k: recursive_dataclass_to_dict(v) for k, v in obj_dict.items()}
    else:
        return obj

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
    env_kwargs = recursive_dataclass_to_dict(env_kwargs)
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


class RecordEpisodeRerun(gym.Wrapper):
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
    ) -> None:
        super().__init__(env)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._elapsed_record_steps = 0
        self._episode_id = -1
        self.current_path_to_rrd = None

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

    def init_new_rrd(self, rerun_name):
        # ###############################
        # rerun stuff
        # ###############################

        rrd_name = rerun_name + ".rrd"
        path_to_rrd = self.output_dir / rrd_name
        self.current_path_to_rrd = path_to_rrd
        recording_id = str(uuid.uuid4())
        rr.init("record_maniskill_rerun", recording_id=recording_id, spawn=False, strict=True)
        rr.save(path_to_rrd)

        self.sim_dt_bw_step = 1.0 / self.env.sim_config.control_freq

        # human_render_cam_params = self.env.scene.human_render_cameras['render_camera'].get_params()
        # human_render_cam_intrisic = human_render_cam_params['intrinsic_cv'][0]
        # human_render_cam_cam2world_gl = human_render_cam_params['cam2world_gl'][0][:3, :4]
        # human_render_cam_extrinsic_cv = human_render_cam_params['extrinsic_cv'][0]

        # base_camera_cam2world_gl = self.env.scene.sensors['base_camera'].get_params()['cam2world_gl'][0].cpu().numpy() # actually W_tf_C
        # use extrinsic cv not cam2world_gl. GL has weird convention in that it seems to flip the y and z axis

        base_camera_extrinsic_cv = self.env.scene.sensors['base_camera'].get_params()['extrinsic_cv'][0].cpu().numpy()
        # add the bottom 0,0,0,1 row to extrinsic_cv
        base_camera_extrinsic_cv = np.concatenate((base_camera_extrinsic_cv, np.array([[0, 0, 0, 1]])), axis=0)
        base_camera_intrinsic_cv = self.env.scene.sensors['base_camera'].get_params()['intrinsic_cv'][0].cpu().numpy()

        self.color_K = base_camera_intrinsic_cv
        self.depth_K = base_camera_intrinsic_cv
        # self.world_tf_cam = base_camera_cam2world_gl
        # self.cam_tf_world = np.linalg.inv(base_camera_cam2world_gl)
        self.cam_tf_world = base_camera_extrinsic_cv
        self.world_tf_cam = np.linalg.inv(base_camera_extrinsic_cv)

        self.world_camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=self.env.camera_width,
            height=self.env.camera_height,
            fx=self.color_K[0, 0],
            fy=self.color_K[1, 1],
            cx=self.color_K[0, 2],
            cy=self.color_K[1, 2],
        )

        rr.log(
            # "world/camera/color", 
            "world/camera", 
            rr.Pinhole(
            image_from_camera=self.world_camera_intrinsics.intrinsic_matrix,
            resolution=[self.world_camera_intrinsics.width, self.world_camera_intrinsics.height],
            camera_xyz=rr.ViewCoordinates.RDF,
            ),
            static=True,
        )
        rr.log(
            # "world/camera/color", 
            "world/camera", 
            rr.Transform3D(
            translation=self.world_tf_cam[:3, 3],
            mat3x3=self.world_tf_cam[:3, :3],
            ),
            static=True,
        )

        # self.policy_obs_color_K = self.color_K.copy()
        # self.policy_obs_color_K[:2, :] *= 0.5
        # self.policy_obs_camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        #     width=320,
        #     height=240,
        #     fx=self.policy_obs_color_K[0, 0],
        #     fy=self.policy_obs_color_K[1, 1],
        #     cx=self.policy_obs_color_K[0, 2],
        #     cy=self.policy_obs_color_K[1, 2],
        # )
        # rr.log("world/policy/obs/camera",
        #     rr.Pinhole(
        #     image_from_camera=self.policy_obs_camera_intrinsics.intrinsic_matrix,
        #     resolution=[self.policy_obs_camera_intrinsics.width, self.policy_obs_camera_intrinsics.height],
        #     camera_xyz=rr.ViewCoordinates.RDF,
        #     ),
        #     static=True,
        # )
        # rr.log("world/policy/obs/camera", 
        #     rr.Transform3D(
        #     translation=self.world_tf_cam[:3, 3],
        #     mat3x3=self.world_tf_cam[:3, :3],
        #     ),
        #     static=True,
        # )
        
        # 31 dimensional
        # first 7 vars are position, quaternion of root link, next 6 are velocities for root link, 
        # and finally, next 18 are joint positions and velocities (9 because 7 for joints and 2 for gripper)
        franka_state = self.env.agent.robot.get_state()[0].cpu().numpy()

        rr.log("world/robot", rr.Transform3D(
            translation=franka_state[:3],
            # mat3x3=np.eye(3),
            quaternion=franka_state[[4,5,6,3]], # needed to convert from [w,x,y,z] to [x,y,z,w]
            ),
            static=True,
        )

        # path_to_this_file = os.path.dirname(os.path.abspath(__file__))
        # use pathlib to get the path to the current file
        path_to_contact_estimation = path_to_workspace_root / 'contact_estimation'
        franka_urdf_path = path_to_contact_estimation / 'src/dataset/franka_meshes/panda.urdf'
        
        assert franka_urdf_path.exists()
        self.franka_urdf_logger = URDFLogger(franka_urdf_path, root_path="world/robot/")
        self.franka_urdf_logger.log()

        joint_dict = self.fill_joint_dict(franka_state[13:20])
        self.franka_urdf_logger.update_joints(joint_dict)

        self.num_masks = 1
        self.action_history_len = None
        self.action_plan_len = None

        self.mask_colors = distinctipy.get_colors(self.num_masks)

        # '''
        # # cmaps that avoid red (current) and green (target)
        # # blue: cosmic,  saphire, amethyst, arctic, freeze, ocean,
        # # blue/purple/green: lilac,
        # # blue/green: swamp,
        # # blue/red: torch,
        # # blue/yellow: eclipse,  
        # # purple/yellow: ghostlight, sepia, fall,
        # # purple/blue: gem,
        # # purple/blue/white: voltage,
        # # purple/pink: bubblegum,
        # # purple/pink/white: gothic,
        # # green/red: dusk, savanna,
        # # green/blue/purple/white: horizon,
        # # red/yellow: amber,
        # # black/white: neutral,
        # # self.action_history_cmap = cmr.sepia # or possibly amber
        # '''
        # self.action_history_cmap = cmr.bubblegum # or possibly amber
        # self.action_history_cmap_clamp = (0.4, 0.8)

        # # self.action_plan_cmap = plt.cm.cividis
        # self.action_plan_cmap = cmr.swamp
        # self.action_plan_cmap_clamp = (0.2, 0.8)

        # if self.only_log_tfs:
        #     current_gripper = self.get_abstract_gripper(
        #             color=[1.0, 0.0, 0.0],
        #             fill_mode="majorwireframe",
        #             label=None,
        #             radius=0.0015,
        #         )
        #     rr.log("world/robot/end_effector/current_pose", 
        #         current_gripper,
        #         static=True,
        #     )

        #     rr.log("world/policy/obs/robot/end_effector/current_pose",
        #            current_gripper,
        #         static=True,)

        #     target_gripper = self.get_abstract_gripper(
        #             color=[0.0, 1.0, 0.0],
        #             fill_mode="majorwireframe",
        #             label=None,
        #             radius=0.0015,
        #         )
        #     rr.log("world/robot/end_effector/target_pose",
        #         target_gripper,
        #         static=True,
        #     )

        #     for i in range(self.action_history_len):
        #         index = (float(i/(self.action_history_len-1))*(self.action_history_cmap_clamp[1]-self.action_history_cmap_clamp[0]))+self.action_history_cmap_clamp[0]
        #         color = list(self.action_history_cmap(index)[:3]) # ignore alpha
        #         action_history_gripper = self.get_abstract_gripper(
        #             color=color,
        #             fill_mode="majorwireframe",
        #             label=None,
        #             radius=0.0005,
        #         )
        #         rr.log(f"world/policy/obs/robot/action_history/{i}", 
        #             action_history_gripper,
        #             static=True,
        #         )

        #     for i in range(self.action_plan_len):
        #         index = (float(i/(self.action_plan_len-1))*(self.action_plan_cmap_clamp[1]-self.action_plan_cmap_clamp[0]))+self.action_plan_cmap_clamp[0]
        #         color = list(self.action_plan_cmap(index)[:3]) # ignore alpha
        #         action_plan_gripper = self.get_abstract_gripper(
        #             color=color, 
        #             fill_mode="majorwireframe",
        #             label=None,
        #             radius=0.0005,
        #         )
        #         rr.log(f"world/policy/action_plan/{i}", 
        #             action_plan_gripper,
        #             static=True,
        #         )

        # self.prev_end_effector_position = None

    def fill_joint_dict(self, joint_positions):
        joint_dict = dict()
        for i, joint_angle in enumerate(joint_positions):
            joint_dict[f"panda_joint{i+1}"] = joint_angle
        return joint_dict
    
    def change_output_dir(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def num_envs(self):
        return self.base_env.num_envs

    @property
    def base_env(self) -> BaseEnv:
        return self.env.unwrapped

    def log_color_image(
            self,
            color_image: np.ndarray,
            episode_timestamp: float,
            episode_step: int, 
            rerun_name: str,
            ) -> None:
        assert color_image.ndim == 3, f"Color image should be in HWC format, got {color_image.ndim}"
        assert color_image.shape[2] == 3, f"Color image should be in HWC format, got {color_image.shape}"
        assert color_image.dtype == np.uint8, f"Color image should be in uint8 format, got {color_image.dtype}"

        rr.set_time("episode_timestamp", timestamp=episode_timestamp)
        rr.set_time("episode_step", sequence=episode_step)

        rr.log(rerun_name, rr.Image(color_image))
    
    def log_depth_image(
            self,
            depth_image: np.ndarray,
            episode_timestamp: float, 
            episode_step: int,
            rerun_name: str,
            ) -> None:
        assert depth_image.dtype == np.uint16, f"Depth image should be in uint16 format, got {depth_image.dtype}"

        rr.set_time("episode_timestamp", timestamp=episode_timestamp)
        rr.set_time("episode_step", sequence=episode_step)

        rr.log(rerun_name, rr.DepthImage(depth_image, meter=1000)) # 1000 is to convert mm to m
    
    def log_colored_pointcloud(
                                self,
                                color_image: np.ndarray, 
                                depth_image: np.ndarray, 
                                episode_timestamp: float, 
                                episode_step: int,
                                camera_intrinsics: PinholeCameraIntrinsic,
                                world_tf_cam: np.ndarray,
                                base_rerun_name: str,
                                ) -> None:
        assert color_image.ndim == 3, f"Color image should be in HWC format, got {color_image.ndim}"
        assert color_image.shape[2] == 3, f"Color image should be in HWC format, got {color_image.shape}"
        assert color_image.dtype == np.uint8, f"Color image should be in uint8 format, got {color_image.dtype}"

        assert depth_image.dtype == np.uint16, f"Depth image should be in uint16 format, got {depth_image.dtype}"


        rr.set_time("episode_timestamp", timestamp=episode_timestamp)
        rr.set_time("episode_step", sequence=episode_step)

        depth_image = depth_image.astype(np.float32)

        # Convert to Open3D objects
        color_o3d = o3d.geometry.Image(color_image)
        depth_o3d = o3d.geometry.Image(depth_image)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, 
            depth_o3d, 
            depth_scale=1000.0,
            depth_trunc=3.0,
            convert_rgb_to_intensity=False
        )

        # Project colorized points using depth and camera params
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera_intrinsics)
        pcd.transform(world_tf_cam)

        rr.log(base_rerun_name + "/pointcloud", rr.Points3D(pcd.points, colors=pcd.colors))

    def log_obs(self, obs):
        color_image = obs['sensor_data']['base_camera']['rgb'][0].cpu().numpy()
        depth_image = obs['sensor_data']['base_camera']['depth'][0].cpu().numpy().astype(np.uint16)
        self.log_color_image(
            color_image,
            episode_timestamp=self._elapsed_record_steps * self.sim_dt_bw_step,
            episode_step=self._elapsed_record_steps,
            rerun_name="world/camera/color",
        )
        self.log_depth_image(
            depth_image,
            episode_timestamp=self._elapsed_record_steps * self.sim_dt_bw_step,
            episode_step=self._elapsed_record_steps,
            rerun_name="world/camera/depth",
        )
        self.log_colored_pointcloud(
            color_image,
            depth_image,
            episode_timestamp=self._elapsed_record_steps * self.sim_dt_bw_step,
            episode_step=self._elapsed_record_steps,
            camera_intrinsics=self.world_camera_intrinsics,
            world_tf_cam=self.world_tf_cam,
            base_rerun_name="world",
        )

    def reset(
        self,
        *args,
        seed: Optional[Union[int, List[int]]] = None,
        options: Optional[dict] = dict(),
        **kwargs,
    ):
        # if self.current_path_to_rrd is not None:
        #     rr.flush()
        self._episode_id += 1
        if seed is not None:
            self.current_env_seed = seed

        # then close the rrd and start a new one
        rrd_filename = f"traj_{self._episode_id}_seed_{self.current_env_seed}"
        self.init_new_rrd(rrd_filename)

        obs, info = super().reset(*args, seed=seed, options=options, **kwargs)
        self.log_obs(obs)
        # adding the "batch" dimension as is done by common.batch is creating the temporal dimension such that each array is Txbx(data dims), where b is num envs
        
        self.last_reset_kwargs = copy.deepcopy(dict(options=options, **kwargs))
        if seed is not None:
            self.last_reset_kwargs.update(seed=seed)
        return obs, info

    def step(self, action, start_signal=None):
        obs, rew, terminated, truncated, info = super().step(action)
        self._elapsed_record_steps += 1

        self.log_obs(obs)
        franka_state = self.env.agent.robot.get_state()[0].cpu().numpy()
        joint_dict = self.fill_joint_dict(franka_state[13:20])
        self.franka_urdf_logger.update_joints(joint_dict)

        return obs, rew, terminated, truncated, info
    
    def extract_individual_segmentation_masks(
            self, trajectory_data_buffer, segmentation_id_map,
    ):
        assert 'observation.segmentation' in trajectory_data_buffer, "Segmentation masks not found in trajectory data buffer"
        image_shape = trajectory_data_buffer['observation.segmentation'].shape[2:]
        
        if 'observation.EE_obj_mask' not in trajectory_data_buffer:
            trajectory_data_buffer.create_array('observation.EE_obj_mask', shape=(0,) + (self.num_envs,) + image_shape, dtype=np.uint8, chunks=(1,) + (self.num_envs,) + image_shape, overwrite=True)#, compressor=self.zarr_compressor)

        EE_obj_mask = (trajectory_data_buffer['observation.segmentation'][:] == segmentation_id_map[f"{self.env.grasped_book.name}_0"]).astype(np.uint8)
        trajectory_data_buffer['observation.EE_obj_mask'].append(EE_obj_mask)


    def close(self) -> None:
        # if self.current_path_to_rrd is not None:
        #     rr.flush()
            
        rr.disconnect()
        return super().close()
