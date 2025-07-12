#%%
from pathlib import Path
path_to_workspace_root = Path(__file__).resolve().parents[4]

# self.franka_urdf_logger = URDFLogger(franka_urdf_path, root_path="world/robot/")
path_to_rerun = path_to_workspace_root / 'rerun_utils'
import sys
sys.path.append(str(path_to_rerun))
from rerun_loader_urdf import URDFLogger
#%%
import copy
import inspect
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
from mani_skill.utils.common import apply_transform_to_poses, unroll_delta_actions, compute_action_plan_error
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

from typing import Tuple
from pytorch3d import transforms
from torch import Tensor

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

def linestrips3d_from_boxes(
        centers: np.ndarray,
        half_sizes: np.ndarray,
        quaternions: np.ndarray,
        radii: np.ndarray,
        colors: np.ndarray,
        labels: list=None,
):
    if labels is None:
        labels = [None]*len(centers)

    linestrip_points_list = []
    linestrip_radii_list = []
    linestrip_colors_list = []
    linestrip_labels_list = []
    for center, half_size, quaternion, color, radius, label in zip(centers, half_sizes, quaternions, colors, radii, labels):
        linestrip_points, linestrip_radii, linestrip_colors, linestrip_labels = linestrip3d_from_box(
            center, half_size, quaternion, radius, color, label
        )
        linestrip_points_list.extend(linestrip_points)
        linestrip_radii_list.extend(linestrip_radii)
        linestrip_colors_list.extend(linestrip_colors)
        if linestrip_labels is not None:
            linestrip_labels_list.extend(linestrip_labels)


    # linestrip_points = np.concatenate(linestrip_points_list, axis=0)
    # linestrip_colors = np.concatenate(linestrip_colors_list, axis=0)

    if len(linestrip_labels_list) == 0:
        linestrip_labels_list = None

    linestrips3d = rr.LineStrips3D(
        linestrip_points_list,
        radii=linestrip_radii_list,
        colors=linestrip_colors_list,
        labels=linestrip_labels_list
    )
    return linestrips3d

def linestrip3d_from_box(
        center: np.ndarray,
        half_sizes: np.ndarray,
        quaternion: np.ndarray,
        radius: float,
        color: np.ndarray,
        label: str=None,
    ):
    '''
    center: 3x1 np.ndarray
    half_sizes: 3x1 np.ndarray
    quaternion: 4x1 np.ndarray (x, y, z, w)
    radius: float
    color: 3x1 np.ndarray
    label: str
    '''
    # 8 corners of the box
    corners = np.array([
        [1, 1, 1],
        [-1, 1, 1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, -1],
        [1, -1, -1],
    ])
    corners = corners*half_sizes
    # apply transformation  
    corners = (R.from_quat(quaternion).as_matrix() @ corners.T).T + center

    points = []
    points.append(
        [
            corners[0],
            corners[1],
            corners[2],
            corners[3],
            corners[0],
        ]
    )
    
    points.append(
        [
            corners[4],
            corners[5],
            corners[6],
            corners[7],
            corners[4],
        ]
    )
    points.append(
        [
            corners[0],
            corners[1],
        ]
    )
    points.append(
        [
            corners[0],
            corners[4],
        ]
    )
    points.append(
        [
            corners[1],
            corners[5],
        ]
    )
    points.append(
        [
            corners[2],
            corners[6],
        ]
    )
    points.append(
        [
            corners[3],
            corners[7],
        ]
    )

    radii = np.ones(len(points))*radius
    colors = np.ones((len(points), 3))*color
    if label is not None:
        labels = [label]*len(points)
    else:
        labels = None

    '''
    points: list of 6 Nx3 lists
    radii: 6x1 np.ndarray
    colors: 6x3 np.ndarray
    labels: list of 6 str
    '''
    return points, radii, colors, labels

def signed_max_across_plan(action_plan: torch.Tensor):
    '''
    action_plan is of shape (..., A) where A is the action dimension
    '''
    action_plan_sign = torch.sign(action_plan)
    action_plan_abs = torch.abs(action_plan)
    action_plan_argmax = torch.argmax(action_plan_abs, dim=-1, keepdim=True)
    action_plan_signed_max = action_plan_abs.gather(-1, action_plan_argmax) * action_plan_sign.gather(-1, action_plan_argmax)
    return action_plan_signed_max


class GripperDimensionsHelper:
    """A helper class to provide gripper dimensions and related properties."""

    @property
    def z_bottom_of_finger_to_center(self) -> float:
        return 0.009

    @property
    def z_top_of_finger_to_bottom_of_body(self) -> float:
        return 0.007

    @property
    def gripper_half_dims(self) -> np.ndarray:
        return np.array([
            [0.032, 0.1, .033],  # x, y, z (width, depth, height) main body
            [.018/2, .027/2, .054/2],  # x, y, z (width, depth, height) finger
            [.018/2, .027/2, .054/2],  # x, y, z (width, depth, height) finger
        ])
    
    def end_effector_to_grasped_object_translation(self, grasped_object_sizes: np.ndarray) -> np.ndarray:
        end_effector_to_grasped_object_translation = np.zeros((3,), dtype=np.float32)
        # end_effector_to_grasped_object_translation[2] = (self.gripper_dimensions_helper.gripper_half_dims[1, 2] + self.gripper_dimensions_helper.z_top_of_finger_to_bottom_of_body) - grasped_object_sizes[2]/2
        end_effector_to_grasped_object_translation[2] = self.z_bottom_of_finger_to_center - 2*self.gripper_half_dims[1, 2] + self.z_top_of_finger_to_bottom_of_body + grasped_object_sizes[2]/2.0
        return end_effector_to_grasped_object_translation
    
    def end_effector_to_grasped_object_bottom_right_corner_translation(self, grasped_object_sizes: np.ndarray) -> np.ndarray:
        """
        Calculate the translation from the end effector to the bottom right corner of the grasped object.
        grasped_object_sizes: np.ndarray given as [length, width, height] in meters.
        """
        end_effector_to_grasped_object_translation = np.zeros((3,), dtype=np.float32)
        # end_effector_to_grasped_object_translation[2] = (self.gripper_dimensions_helper.gripper_half_dims[1, 2] + self.gripper_dimensions_helper.z_top_of_finger_to_bottom_of_body) - grasped_object_sizes[2]/2
        end_effector_to_grasped_object_translation[2] = self.z_bottom_of_finger_to_center - 2*self.gripper_half_dims[1, 2] + self.z_top_of_finger_to_bottom_of_body + grasped_object_sizes[2]
        end_effector_to_grasped_object_translation[1] = - grasped_object_sizes[1]/2.0
        return end_effector_to_grasped_object_translation
    
    def grasped_book_center_to_bottom_right_corner_translation(self, grasped_object_sizes: np.ndarray) -> np.ndarray:
        """
        Calculate the translation from the center of the grasped book to its bottom right corner.
        grasped_object_sizes: np.ndarray given as [length, width, height] in meters.
        """
        # in grasped book frame, x points back towards robot, y points to the right, and z points down
        grasped_book_center_to_bottom_right_corner_translation = np.zeros((3,), dtype=np.float32)
        grasped_book_center_to_bottom_right_corner_translation[2] = grasped_object_sizes[2]/2.0
        grasped_book_center_to_bottom_right_corner_translation[1] = grasped_object_sizes[1]/2.0 
        return grasped_book_center_to_bottom_right_corner_translation

def arclength_from_action_plan(
    target_pose_trajectory: torch.Tensor,
    rotation_representation: str = 'euler_angles',
    ):
    '''
    compute the arclength of the action plan in both translation and rotation

    target_pose_trajectory: BxHxN
    rotation_representation: 'euler_angles', 'axis_angle' or 'quaternion'
    '''
    assert target_pose_trajectory.ndim == 3, "target_pose_trajectory must be a 3D tensor"
    assert rotation_representation in ['euler_angles', 'axis_angle', 'quaternion'], "rotation_representation must be one of ['euler_angles', 'axis_angle', 'quaternion']"
    error_dict = dict()
    error_dict['translation'] = dict()
    error_dict['rotation'] = dict()

    if rotation_representation == 'euler_angles':
        assert target_pose_trajectory.shape[-1] == 6, "target_pose_trajectory must have 6 dimensions for euler angles"
        target_orientation_trajectory = transforms.matrix_to_quaternion(transforms.euler_angles_to_matrix(target_pose_trajectory[..., 3:6], convention='XYZ'))
    elif rotation_representation == 'axis_angle':
        assert target_pose_trajectory.shape[-1] == 6, "target_pose_trajectory must have 6 dimensions for axis angle"
        target_orientation_trajectory = transforms.axis_angle_to_quaternion(target_pose_trajectory[..., 3:6])
    elif rotation_representation == 'quaternion':
        assert target_pose_trajectory.shape[-1] == 7, "target_pose_trajectory must have 7 dimensions for quaternion (x, y, z, w)"
        target_orientation_trajectory = target_pose_trajectory[..., 3:7]

    target_position_trajectory = target_pose_trajectory[..., :3]
    # compute the arclength of the translation
    target_displacement_trajectory = torch.linalg.norm(target_position_trajectory[:, 1:, :] - target_position_trajectory[:, :-1, :], dim=-1, ord=2)
    error_dict['translation']['arclength'] = target_displacement_trajectory.sum(dim=-1) # B
    error_dict['translation']['mean'] = torch.mean(target_displacement_trajectory, dim=-1)
    error_dict['translation']['max'] = torch.max(target_displacement_trajectory, dim=-1).values
    
    # compute the arclength of the rotation
    target_rotation_displacement_trajectory = torch.linalg.norm(transforms.quaternion_to_axis_angle(transforms.quaternion_multiply(transforms.quaternion_invert(target_orientation_trajectory[:, :-1, :]), target_orientation_trajectory[:, 1:, :])), dim=-1, ord=2)
    target_rotation_displacement_trajectory = torch.rad2deg(target_rotation_displacement_trajectory) # convert to degrees
    error_dict['rotation']['arclength'] = target_rotation_displacement_trajectory.sum(dim=-1) # B
    error_dict['rotation']['mean'] = torch.mean(target_rotation_displacement_trajectory, dim=-1)
    error_dict['rotation']['max'] = torch.max(target_rotation_displacement_trajectory, dim=-1).values
    return error_dict

def log_action_plan_error(
                            rerun_recording:rr.RecordingStream,
                            predicted_action_plan:Tensor, 
                            ground_truth_action_plan:Tensor, 
                            current_target_pose:Tensor,
                            time_between_steps:float,
                            timestamp:float,
                            step:int,
                            rotation_representation:str='euler_angles', 
                            rerun_name_prefix:str='world/',
                            ):
    '''
    predicted_action_plan: BxHxN
    ground_truth_action_plan: BxHxN
    current_target_pose: BxN
    assumes that both are expressed in the same frame
    assumes quaternions are with real/scalar part first
    '''
    basis_vectors = ['x', 'y', 'z']
    rerun_recording.set_time("episode_timestamp", timestamp=timestamp)
    rerun_recording.set_time("episode_step", sequence=step)

    # append the current target pose to the predicted action plan
    ground_truth_action_plan_with_current_target_pose = torch.cat(
        [current_target_pose.unsqueeze(1), ground_truth_action_plan[..., :6]], dim=1
    )  # Bx(H+1)xN
    assert ground_truth_action_plan_with_current_target_pose.ndim == 3, "ground_truth_action_plan_with_current_target_pose must be a 3D tensor"
    assert ground_truth_action_plan_with_current_target_pose.shape[-1] == 6, "ground_truth_action_plan_with_current_target_pose must have 6 dimensions (x, y, z, roll, pitch, yaw) for euler angles or 7 for quaternion"
    error_dict = arclength_from_action_plan(
        ground_truth_action_plan_with_current_target_pose,
        rotation_representation=rotation_representation,
    )
    horizon_period = ground_truth_action_plan_with_current_target_pose.shape[1] * time_between_steps

    ground_truth_to_predicted_translation_errors_vector_in_world, ground_truth_to_predicted_rotation_axis_angle_in_world = compute_action_plan_error(predicted_action_plan, ground_truth_action_plan, rotation_representation=rotation_representation)
        
    ground_truth_to_predicted_rotation_angle_errors = torch.linalg.norm(ground_truth_to_predicted_rotation_axis_angle_in_world, dim=-1, ord=2, keepdim=True)

    rotation_error_rmse = torch.sqrt(torch.mean(ground_truth_to_predicted_rotation_angle_errors**2)).item()
    rerun_recording.log(
        rerun_name_prefix + "rotation_error/rmse",
        rr.Scalars(
            scalars=[rotation_error_rmse],
        )
    )
    rotation_velocity_error_rmse = rotation_error_rmse / horizon_period
    rerun_recording.log(
        rerun_name_prefix + "rotation_error_velocity/rmse",
        rr.Scalars(
            scalars=[rotation_velocity_error_rmse],
        )
    )
    rerun_recording.log(
        rerun_name_prefix + "rotation_error_arclength_normalized/rmse",
        rr.Scalars(
            scalars=[rotation_error_rmse/error_dict['rotation']['arclength'].item()],
        )
    )
    
    rotation_error_mean = torch.mean(ground_truth_to_predicted_rotation_angle_errors).item()
    rerun_recording.log(
        rerun_name_prefix + "rotation_error/mean",
        rr.Scalars(
            scalars=[rotation_error_mean],
        )
    )
    rotation_velocity_error_mean = rotation_error_mean / horizon_period
    rerun_recording.log(
        rerun_name_prefix + "rotation_error_velocity/mean",
        rr.Scalars(
            scalars=[rotation_velocity_error_mean],
        )
    )
    rerun_recording.log(
        rerun_name_prefix + "rotation_error_arclength_normalized/mean",
        rr.Scalars(
            scalars=[rotation_error_mean/error_dict['rotation']['arclength'].item()],
        )
    )

    rotation_error_max = torch.max(ground_truth_to_predicted_rotation_angle_errors).item()
    rerun_recording.log(
        rerun_name_prefix + "rotation_error/max",
        rr.Scalars(
            scalars=[rotation_error_max],
        )
    )
    rotation_velocity_error_max = rotation_error_max / horizon_period
    rerun_recording.log(
        rerun_name_prefix + "rotation_error_velocity/max",
        rr.Scalars(
            scalars=[rotation_velocity_error_max],
        )
    )

    rerun_recording.log(
        rerun_name_prefix + "rotation_error_arclength_normalized/max",
        rr.Scalars(
            scalars=[rotation_error_max/ error_dict['rotation']['arclength'].item()],
        )
    )

    for i, basis_vector in enumerate(basis_vectors):
        ground_truth_to_predicted_rotation_axis_angle_in_world_basis = ground_truth_to_predicted_rotation_axis_angle_in_world[..., i]
        ground_truth_to_predicted_rotation_axis_angle_in_world_basis_mean = torch.mean(ground_truth_to_predicted_rotation_axis_angle_in_world_basis).item()
        rerun_recording.log(
            rerun_name_prefix + f"rotation_error/{basis_vector}/mean",
            rr.Scalars(
                scalars=[ground_truth_to_predicted_rotation_axis_angle_in_world_basis_mean],
            )
        )
        ground_truth_to_predicted_rotation_axis_angle_in_world_basis_mean_velocity = ground_truth_to_predicted_rotation_axis_angle_in_world_basis_mean / horizon_period
        rerun_recording.log(
            rerun_name_prefix + f"rotation_error_velocity/{basis_vector}/mean",
            rr.Scalars(
                scalars=[ground_truth_to_predicted_rotation_axis_angle_in_world_basis_mean_velocity],
            )
        )
        rerun_recording.log(
            rerun_name_prefix + f"rotation_error_arclength_normalized/{basis_vector}/mean",
            rr.Scalars(
                scalars=[ground_truth_to_predicted_rotation_axis_angle_in_world_basis_mean / error_dict['rotation']['arclength'].item()],
            )
        )
        ground_truth_to_predicted_rotation_axis_angle_in_world_basis_mean_signed_max = signed_max_across_plan(ground_truth_to_predicted_rotation_axis_angle_in_world_basis).item()
        rerun_recording.log(
            rerun_name_prefix + f"rotation_error/{basis_vector}/max",
            rr.Scalars(
                scalars=[ground_truth_to_predicted_rotation_axis_angle_in_world_basis_mean_signed_max],
            )
        )
        ground_truth_to_predicted_rotation_axis_angle_in_world_basis_max_velocity = ground_truth_to_predicted_rotation_axis_angle_in_world_basis_mean_signed_max / horizon_period
        rerun_recording.log(
            rerun_name_prefix + f"rotation_error_velocity/{basis_vector}/max",
            rr.Scalars(
                scalars=[ground_truth_to_predicted_rotation_axis_angle_in_world_basis_max_velocity],
            )
        )
        rerun_recording.log(
            rerun_name_prefix + f"rotation_error_arclength_normalized/{basis_vector}/max",
            rr.Scalars(
                scalars=[ground_truth_to_predicted_rotation_axis_angle_in_world_basis_mean_signed_max/ error_dict['rotation']['arclength'].item()],
            )
        )
    
    ground_truth_to_predicted_translation_errors = torch.linalg.norm(ground_truth_to_predicted_translation_errors_vector_in_world, dim=-1, ord=2, keepdim=True)
    ground_truth_to_predicted_translation_errors_rmse = torch.sqrt(torch.mean(ground_truth_to_predicted_translation_errors**2)).item()
    rerun_recording.log(
        rerun_name_prefix + "translation_error/rmse",
        rr.Scalars(
            scalars=[ground_truth_to_predicted_translation_errors_rmse],
        )
    )
    ground_truth_to_predicted_translation_errors_velocity_rmse = ground_truth_to_predicted_translation_errors_rmse / horizon_period
    rerun_recording.log(
        rerun_name_prefix + "translation_error_velocity/rmse",
        rr.Scalars(
            scalars=[ground_truth_to_predicted_translation_errors_velocity_rmse],
        )
    )
    rerun_recording.log(
        rerun_name_prefix + "translation_error_arclength_normalized/rmse",
        rr.Scalars(
            scalars=[ground_truth_to_predicted_translation_errors_rmse/ error_dict['translation']['arclength'].item()],
        )
    )

    ground_truth_to_predicted_translation_errors_mean = torch.mean(ground_truth_to_predicted_translation_errors).item()
    rerun_recording.log(
        rerun_name_prefix + "translation_error/mean",
        rr.Scalars(
            scalars=[ground_truth_to_predicted_translation_errors_mean],
        )
    )
    ground_truth_to_predicted_translation_errors_velocity_mean = ground_truth_to_predicted_translation_errors_mean / horizon_period
    rerun_recording.log(
        rerun_name_prefix + "translation_error_velocity/mean",
        rr.Scalars(
            scalars=[ground_truth_to_predicted_translation_errors_velocity_mean],
        )
    )
    rerun_recording.log(
        rerun_name_prefix + "translation_error_arclength_normalized/mean",
        rr.Scalars(
            scalars=[ground_truth_to_predicted_translation_errors_mean/ error_dict['translation']['arclength'].item()],
        )
    )
    ground_truth_to_predicted_translation_errors_max = torch.max(ground_truth_to_predicted_translation_errors).item()
    rerun_recording.log(
        rerun_name_prefix + "translation_error/max",
        rr.Scalars(
            scalars=[ground_truth_to_predicted_translation_errors_max],
        )
    )
    ground_truth_to_predicted_translation_errors_velocity_max = ground_truth_to_predicted_translation_errors_max / horizon_period
    rerun_recording.log(
        rerun_name_prefix + "translation_error_velocity/max",
        rr.Scalars(
            scalars=[ground_truth_to_predicted_translation_errors_velocity_max],
        )
    )
    rerun_recording.log(
        rerun_name_prefix + "translation_error_arclength_normalized/max",
        rr.Scalars(
            scalars=[ground_truth_to_predicted_translation_errors_max/ error_dict['translation']['arclength'].item()],
        )
    )
    for i, basis_vector in enumerate(basis_vectors):
        ground_truth_to_predicted_translation_errors_vector_in_world_basis = ground_truth_to_predicted_translation_errors_vector_in_world[..., i]
        
        ground_truth_to_predicted_translation_errors_vector_in_world_basis_mean = torch.mean(ground_truth_to_predicted_translation_errors_vector_in_world_basis).item()
        rerun_recording.log(
            rerun_name_prefix + f"translation_error/{basis_vector}/mean",
            rr.Scalars(
                scalars=[ground_truth_to_predicted_translation_errors_vector_in_world_basis_mean],
            )
        )
        ground_truth_to_predicted_translation_errors_vector_in_world_basis_mean_velocity = ground_truth_to_predicted_translation_errors_vector_in_world_basis_mean / horizon_period
        rerun_recording.log(
            rerun_name_prefix + f"translation_error_velocity/{basis_vector}/mean",
            rr.Scalars(
                scalars=[ground_truth_to_predicted_translation_errors_vector_in_world_basis_mean_velocity],
            )
        )
        rerun_recording.log(
            rerun_name_prefix + f"translation_error_arclength_normalized/{basis_vector}/mean",
            rr.Scalars(
                scalars=[ground_truth_to_predicted_translation_errors_vector_in_world_basis_mean/ error_dict['translation']['arclength'].item()],
            )
        )

        ground_truth_to_predicted_translation_errors_vector_in_world_basis_max = torch.max(ground_truth_to_predicted_translation_errors_vector_in_world_basis).item()
        rerun_recording.log(
            rerun_name_prefix + f"translation_error/{basis_vector}/max",
            rr.Scalars(
                scalars=[ground_truth_to_predicted_translation_errors_vector_in_world_basis_max],
            )
        )
        ground_truth_to_predicted_translation_errors_vector_in_world_basis_max_velocity = ground_truth_to_predicted_translation_errors_vector_in_world_basis_max / horizon_period
        rerun_recording.log(
            rerun_name_prefix + f"translation_error_velocity/{basis_vector}/max",
            rr.Scalars(
                scalars=[ground_truth_to_predicted_translation_errors_vector_in_world_basis_max_velocity],
            )
        )
        rerun_recording.log(
            rerun_name_prefix + f"translation_error_arclength_normalized/{basis_vector}/max",
            rr.Scalars(
                scalars=[ground_truth_to_predicted_translation_errors_vector_in_world_basis_max/ error_dict['translation']['arclength'].item()],
            )
        )
def log_object_velocity(
                            rerun_recording:rr.RecordingStream,
                            velocity:np.ndarray, 
                            timestamp:float,
                            step:int,
                            rerun_name_prefix:str='world/',
                            ):
    '''
    grasped_book_velocity: 6d np.ndarray (in world frame)
    assumes that both are expressed in the same frame
    '''
    basis_vectors = ['x', 'y', 'z']
    rerun_recording.set_time("episode_timestamp", timestamp=timestamp)
    rerun_recording.set_time("episode_step", sequence=step)
    
    translation_velocity = velocity[:3]
    velocity_translation_norm = np.linalg.norm(translation_velocity)
    rerun_recording.log(
        rerun_name_prefix + "translation/magnitude",
        rr.Scalars(
            scalars=[velocity_translation_norm],
        )
    )

    rotation_velocity = velocity[3:]
    velocity_rotation_norm = np.linalg.norm(rotation_velocity)
    rerun_recording.log(
        rerun_name_prefix + "rotation/magnitude",
        rr.Scalars(
            scalars=[velocity_rotation_norm],
        )
    )
    for i, basis_vector in enumerate(basis_vectors):
        translation_velocity_basis = translation_velocity[..., i]
        rerun_recording.log(
            rerun_name_prefix + f"translation/{basis_vector}",
            rr.Scalars(
                scalars=[translation_velocity_basis],
            )
        )
        rotation_velocity_basis = rotation_velocity[..., i]
        rerun_recording.log(
            rerun_name_prefix + f"rotation/{basis_vector}",
            rr.Scalars(
                scalars=[rotation_velocity_basis],
            )
        )
    

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
        policy_name: str,
        wandb_run_id: Optional[str] = None,
        only_log_tfs: bool = False, # whether to log the abstract geometries everytime or only the tfs. False allows trailing visual history of the gripper
        log_action_plan_lumped: bool = True,
        recording_id: Optional[str] = None,
        rollout_policy_name: Optional[str] = None,
        log_grasped_object: bool = False,
        only_log_action_plan: bool = False,
        log_executed_action_plan: bool = True,
        executed_action_plan_length: int = None,
    ) -> None:
        super().__init__(env)
        self.log_executed_action_plan = log_executed_action_plan
        self.executed_action_plan_length = executed_action_plan_length
        self.log_grasped_object = log_grasped_object
        self.policy_name = policy_name
        self.rollout_policy_name = rollout_policy_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.episode_step = 0
        self._episode_id = -1
        self.current_path_to_rrd = None
        self.only_log_tfs = only_log_tfs
        self.log_action_plan_lumped = log_action_plan_lumped
        self.recording_id = recording_id
        self.rerun_recording = None
        self.only_log_action_plan = only_log_action_plan

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
        
        self.gripper_dimensions_helper = GripperDimensionsHelper()

    def log_wandb_offline_metrics(self, entity: str, project: str, run_id: str, keys: list[str]=None):
        if keys is None:
            keys = ['train/train_trans_dist_error', 
            'train/train_rot_angle_error_traj_max',
            'train/val_rot_angle_error_traj_max',
            'train/train_rot_angle_error'
            'train/val_rot_angle_error'
            'train/train_trans_dist_error_traj_max',
            'train/val_trans_dist_error_traj_max',
            'train/train_trans_dist_error',
            'train/val_trans_dist_error',
            'eval/total_success_rate',
            'eval/total_timeout_rate',
            'eval/total_toppled_rate',
            ]
        rrd_name = f"{run_id}_offline_metrics.rrd"
        path_to_rrd = self.output_dir / rrd_name
        self.current_path_to_rrd = path_to_rrd
        if self.recording_id is None:
            self.recording_id = str(uuid.uuid4())
        rerun_recording = rr.RecordingStream("record_maniskill_rerun", recording_id=self.recording_id)
        rerun_recording.save(path_to_rrd)

        import wandb
        api = wandb.Api()

        run = api.run(f"{entity}/{project}/{run_id}")
        
        meta_columns = ['global_step', 'epoch']
        total_columns = meta_columns + keys
        # log the filtered dataframe row by row
        for metric in keys:
            for j,  row in enumerate(run.scan_history(keys=[metric]+meta_columns)):
                rerun_recording.set_time("train_step", sequence=row["global_step"])
                rerun_recording.set_time("train_epoch", sequence=row["epoch"])
                rerun_name_of_metric = f"world/{run_id}/{metric.split('/')[1]}"
                rerun_recording.log(
                    rerun_name_of_metric, 
                    rr.Scalars(
                        scalars=[row[metric]],
                    ),
                    )

        rerun_recording.flush()
        rerun_recording.disconnect()

    def init_new_rrd(self, rerun_name, seed, recording_id=None):
        # ###############################
        # rerun stuff
        # ###############################

        rrd_name = rerun_name + ".rrd"
        path_to_rrd = self.output_dir / rrd_name
        self.current_path_to_rrd = path_to_rrd
        if recording_id is None:
            recording_id = str(uuid.uuid4())
        self.rerun_recording = rr.RecordingStream("record_maniskill_rerun", recording_id=recording_id)
        
        # rr.init("record_maniskill_rerun", recording_id=recording_id, spawn=False, strict=True)
        self.rerun_recording.save(path_to_rrd)

        self.rerun_recording.set_time("episode_timestamp", timestamp=0)
        self.rerun_recording.set_time("episode_step", sequence=0)
        
        grasped_book_info = self.grasped_book_info
        self.rerun_recording.log(
            f"world/description",
            rr.TextDocument(
                f'''
# Sim env info
## Environment
- environment seed: {seed}

## Grasped book info
- grasped book width: {grasped_book_info['sizes'][0,1]}

## Slot info
- slot width: {self.slot_width.item()}
- slot half width: {self.slot_width.item() / 2.0}
- slot negative tolerance: {self.slot_config.negative_tolerance}
                '''.strip(),
                media_type=rr.MediaType.MARKDOWN
            ),
        )

        self.sim_dt_bw_step = 1.0 / self.env.sim_config.control_freq

        self.action_history_cmap = cmr.bubblegum # or possibly amber
        self.action_history_cmap_clamp = (0.4, 0.8)

        # self.action_plan_cmap = plt.cm.cividis
        self.action_plan_cmap_str = 'cmr.swamp'
        self.action_plan_cmap_clamp = (0.2, 0.8)

        self.action_plan_2_cmap_str = 'cmr.bubblegum' # or possibly amber
        self.action_plan_2_cmap_clamp = (0.4, 0.8)
        
        if not self.only_log_action_plan:
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

            self.rerun_recording.log(
                # "world/camera/color", 
                # "world/camera", 
                f"world/{self.policy_name}/camera", 
                rr.Pinhole(
                image_from_camera=self.world_camera_intrinsics.intrinsic_matrix,
                resolution=[self.world_camera_intrinsics.width, self.world_camera_intrinsics.height],
                camera_xyz=rr.ViewCoordinates.RDF,
                ),
                static=True,
            )
            self.rerun_recording.log(
                # "world/camera/color", 
                # "world/camera", 
                f"world/{self.policy_name}/camera", 
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

            self.rerun_recording.log(f"world/{self.policy_name}/robot", rr.Transform3D(
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
            self.franka_urdf_logger = URDFLogger(franka_urdf_path, root_path=f"world/{self.policy_name}/robot/")
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

            if self.only_log_tfs:
                current_gripper = self.get_abstract_gripper(
                        color=np.array([[1.0, 0.0, 0.0]]),
                        fill_mode="linestrips",
                        label=None,
                        radius=0.0015,
                    )
                self.rerun_recording.log(f"world/{self.policy_name}/robot/end_effector/current/gripper_pose", 
                    current_gripper,
                    static=True,
                )

                # rr.log("world/policy/obs/robot/end_effector/current_pose",
                #        current_gripper,
                #     static=True,)

                target_gripper = self.get_abstract_gripper(
                        color=np.array([[0.0, 1.0, 0.0]]),
                        fill_mode="linestrips",
                        label=None,
                        radius=0.0015,
                    )
                self.rerun_recording.log(f"world/{self.policy_name}/robot/end_effector/target/gripper_pose",
                    target_gripper,
                    static=True,
                )

                # for i in range(self.action_history_len):
                #     index = (float(i/(self.action_history_len-1))*(self.action_history_cmap_clamp[1]-self.action_history_cmap_clamp[0]))+self.action_history_cmap_clamp[0]
                #     color = list(self.action_history_cmap(index)[:3]) # ignore alpha
                #     action_history_gripper = self.get_abstract_gripper(
                #         color=color,
                #         fill_mode="majorwireframe",
                #         label=None,
                #         radius=0.0005,
                #     )
                #     rr.log(f"world/policy/obs/robot/action_history/{i}", 
                #         action_history_gripper,
                #         static=True,
                #     )

                # for i in range(self.action_plan_len):
                #     index = (float(i/(self.action_plan_len-1))*(self.action_plan_cmap_clamp[1]-self.action_plan_cmap_clamp[0]))+self.action_plan_cmap_clamp[0]
                #     color = list(self.action_plan_cmap(index)[:3]) # ignore alpha
                #     action_plan_gripper = self.get_abstract_gripper(
                #         color=color, 
                #         fill_mode="majorwireframe",
                #         label=None,
                #         radius=0.0005,
                #     )
                #     rr.log(f"world/policy/action_plan/{i}", 
                #         action_plan_gripper,
                #         static=True,
                #     )

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
            rerun_recording: rr.RecordingStream,
            ) -> None:
        assert color_image.ndim == 3, f"Color image should be in HWC format, got {color_image.ndim}"
        assert color_image.shape[2] == 3, f"Color image should be in HWC format, got {color_image.shape}"
        assert color_image.dtype == np.uint8, f"Color image should be in uint8 format, got {color_image.dtype}"
        
        rerun_recording.set_time("episode_timestamp", timestamp=episode_timestamp)
        rerun_recording.set_time("episode_step", sequence=episode_step)

        rerun_recording.log(rerun_name, rr.Image(color_image))
    
    def log_depth_image(
            self,
            depth_image: np.ndarray,
            episode_timestamp: float, 
            episode_step: int,
            rerun_name: str,
            rerun_recording: rr.RecordingStream,
            ) -> None:
        assert depth_image.dtype == np.uint16, f"Depth image should be in uint16 format, got {depth_image.dtype}"

        rerun_recording.set_time("episode_timestamp", timestamp=episode_timestamp)
        rerun_recording.set_time("episode_step", sequence=episode_step)

        rerun_recording.log(rerun_name, rr.DepthImage(depth_image, meter=1000)) # 1000 is to convert mm to m
    
    def log_colored_pointcloud(
                                self,
                                color_image: np.ndarray, 
                                depth_image: np.ndarray, 
                                episode_timestamp: float, 
                                episode_step: int,
                                camera_intrinsics: PinholeCameraIntrinsic,
                                world_tf_cam: np.ndarray,
                                base_rerun_name: str,
                                rerun_recording: rr.RecordingStream,
                                ) -> None:
        assert color_image.ndim == 3, f"Color image should be in HWC format, got {color_image.ndim}"
        assert color_image.shape[2] == 3, f"Color image should be in HWC format, got {color_image.shape}"
        assert color_image.dtype == np.uint8, f"Color image should be in uint8 format, got {color_image.dtype}"

        assert depth_image.dtype == np.uint16, f"Depth image should be in uint16 format, got {depth_image.dtype}"

        rerun_recording.set_time("episode_timestamp", timestamp=episode_timestamp)
        rerun_recording.set_time("episode_step", sequence=episode_step)

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

        rerun_recording.log(base_rerun_name + "/pointcloud", rr.Points3D(pcd.points, colors=pcd.colors))

    def log_obs(self, obs, rerun_recording: rr.RecordingStream):
        color_image = obs['sensor_data']['base_camera']['rgb'][0].cpu().numpy()
        depth_image = obs['sensor_data']['base_camera']['depth'][0].cpu().numpy().astype(np.uint16)
        self.log_color_image(
            color_image,
            episode_timestamp=self.episode_step * self.sim_dt_bw_step,
            episode_step=self.episode_step,
            rerun_name=f"world/{self.policy_name}/camera/color",
            rerun_recording=rerun_recording,
        )
        self.log_depth_image(
            depth_image,
            episode_timestamp=self.episode_step * self.sim_dt_bw_step,
            episode_step=self.episode_step,
            rerun_name=f"world/{self.policy_name}/camera/depth",
            rerun_recording=rerun_recording,
        )
        self.log_colored_pointcloud(
            color_image,
            depth_image,
            episode_timestamp=self.episode_step * self.sim_dt_bw_step,
            episode_step=self.episode_step,
            camera_intrinsics=self.world_camera_intrinsics,
            world_tf_cam=self.world_tf_cam,
            base_rerun_name=f"world/{self.policy_name}",
            rerun_recording=rerun_recording,
        )
        self.log_pose(
            obs['extra']['end_effector_pose'][0].cpu().numpy(),
            episode_timestamp=self.episode_step * self.sim_dt_bw_step,
            episode_step=self.episode_step,
            rerun_name=f"world/{self.policy_name}/robot/end_effector/current/gripper",
            color=np.array([1.0, 0.0, 0.0]), 
            radius=0.0015,
            rerun_recording=rerun_recording,
        )
        self.log_pose(
            obs['agent']['controller']['arm']['target_pose'][0].cpu().numpy(),
            episode_timestamp=self.episode_step * self.sim_dt_bw_step,
            episode_step=self.episode_step,
            rerun_name=f"world/{self.policy_name}/robot/end_effector/target/gripper",
            color=np.array([0.0, 1.0, 0.0]), 
            radius=0.0015,
            rerun_recording=rerun_recording,
        )
        if self.log_grasped_object:
            grasped_book_info = self.env.grasped_book_info
            self.log_grasped_object_rerun(
                obs['extra']['end_effector_pose'][0].cpu().numpy(),
                grasped_object_sizes= grasped_book_info['sizes'][0],
                episode_timestamp=self.episode_step * self.sim_dt_bw_step,
                episode_step=self.episode_step,
                rerun_name=f"world/{self.policy_name}/robot/end_effector/current/grasped_object",
                color=np.array([1.0, 0.0, 0.0]),
                radius=0.0015,
                rerun_recording=rerun_recording,
            )

            self.log_grasped_object_rerun(
                obs['agent']['controller']['arm']['target_pose'][0].cpu().numpy(),
                grasped_object_sizes= grasped_book_info['sizes'][0],
                episode_timestamp=self.episode_step * self.sim_dt_bw_step,
                episode_step=self.episode_step,
                rerun_name=f"world/{self.policy_name}/robot/end_effector/target/grasped_object",
                color=np.array([0.0, 1.0, 0.0]),
                radius=0.0015,
                rerun_recording=rerun_recording,
            )

    def get_abstract_gripper(
            self,
            color:np.ndarray = np.array([[0.5, 0.5, 0.5]]),
            fill_mode:str="solid", # solid, majorwireframe, densewireframe, linestrips
            label:str=None,
            radius:float=0.0015,
            world_tf_gripper:np.ndarray=None, # posquat with scalar first 
        ) -> Boxes3D:
        
        gripper_centers = np.array([
            [0, 0, -(self.gripper_dimensions_helper.gripper_half_dims[0, 2] + (self.gripper_dimensions_helper.gripper_half_dims[1,2]*2 - self.gripper_dimensions_helper.z_bottom_of_finger_to_center) - self.gripper_dimensions_helper.z_top_of_finger_to_bottom_of_body)], # main body
            [0, self.gripper_dimensions_helper.gripper_half_dims[1,1], -(self.gripper_dimensions_helper.gripper_half_dims[1,2]-self.gripper_dimensions_helper.z_bottom_of_finger_to_center)], # finger 1
            [0, -self.gripper_dimensions_helper.gripper_half_dims[1,1], -(self.gripper_dimensions_helper.gripper_half_dims[1,2]-self.gripper_dimensions_helper.z_bottom_of_finger_to_center)], # finger 2
        ])
        gripper_orientations = R.from_quat([
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
        ])

        colors = np.tile(color, (3, 1))
        num_grippers = 1

        gripper_half_dims = self.gripper_dimensions_helper.gripper_half_dims.copy()

        if world_tf_gripper is not None:
            num_grippers = world_tf_gripper.shape[0]
            assert world_tf_gripper.ndim == 2, f"world_tf_gripper should be of shape (B,7), got {world_tf_gripper.shape}"
            assert world_tf_gripper.shape[1] == 7, f"world_tf_gripper should be None or of shape (7,), got {world_tf_gripper.shape}"
            # should have (3*B)x3 gripper centers and (3*B)x4 gripper orientations
            if num_grippers > 1:
                # need to tile one of them and repeat the other
                # with tiling A,B becomes A,B,A,B while with repeat A,B becomes A,A,B,B
                # then repeat the gripper centers and orientations by B times

                gripper_centers = np.repeat(gripper_centers, num_grippers, axis=0)
                gripper_orientations = R.concatenate(np.repeat(gripper_orientations, num_grippers, axis=0))
                gripper_half_dims = np.repeat(gripper_half_dims, num_grippers, axis=0)

                world_tf_gripper = np.tile(world_tf_gripper, (3,1))

                assert gripper_centers.shape[0] == 3*num_grippers, f"gripper_centers should be of shape (3*B,3), got {gripper_centers.shape}"
                assert len(gripper_orientations) == 3*num_grippers, f"gripper_orientations should be of shape (3*B,4), got {gripper_orientations.shape}"
                assert gripper_half_dims.shape[0] == 3*num_grippers, f"gripper_dims should be of shape (3*B,3), got {gripper_half_dims.shape}"
            
            assert colors.shape[0] == gripper_centers.shape[0], f"colors should be of shape (3*B,3), got {colors.shape}"

            # transform box centers and orientations to world frame
            world_rot_gripper = R.from_quat(world_tf_gripper[:, 3:7], scalar_first=True)
            gripper_centers = world_rot_gripper.apply(gripper_centers) + world_tf_gripper[:, :3]

            # gripper_centers = (world_tf_gripper[:3, :3] @ gripper_centers.T).T + world_tf_gripper[:3, 3]

            # gripper_orientations = R.from_matrix(world_tf_gripper[:3, :3] @ gripper_orientations.as_matrix())
            gripper_orientations = world_rot_gripper*gripper_orientations

        if fill_mode == "linestrips" and not self.only_log_tfs:
            # use the linestrips to draw the gripper
            gripper = linestrips3d_from_boxes(
                centers=gripper_centers,
                half_sizes=gripper_half_dims,
                quaternions=gripper_orientations.as_quat(),
                colors=colors,
                radii=[radius]*3*num_grippers,
                # labels=[label]*3,
            )
        else:
            gripper = rr.Boxes3D(
            centers=gripper_centers,
            half_sizes=gripper_half_dims,
            quaternions=gripper_orientations.as_quat(),
            radii=[radius]*3*num_grippers,
            colors=colors,
            fill_mode=fill_mode,
            # labels=[label]*3,
        )
        return gripper
    
    def log_pose(
                self,
                pose: np.ndarray, #7
                episode_timestamp: float,
                episode_step: int,
                rerun_name: str,
                rerun_recording: rr.RecordingStream,
                color:np.ndarray = np.array([1.0, 0.0, 0.0]),
                radius: float=0.0015,
                ) -> None:
        rerun_recording.set_time("episode_timestamp", timestamp=episode_timestamp)
        rerun_recording.set_time("episode_step", sequence=episode_step)

        if self.only_log_tfs:
            
            rerun_recording.log(rerun_name, 
                rr.Transform3D(
                translation=pose[:3],
                # mat3x3=pose[:3, :3],
                quaternion=pose[[4,5,6,3]], # needed to convert from [w,x,y,z] to [x,y,z,w]
                )
            )
        else:
            abstract_gripper = self.get_abstract_gripper(
                color=color[np.newaxis, :], # add batch dimension
                fill_mode="linestrips",
                label=None,
                radius=radius,
                world_tf_gripper=pose[np.newaxis, :], # add batch dimension
            )

            rerun_recording.log(rerun_name,
                abstract_gripper,
            )

    def get_grasped_object(
            self,
            color: np.ndarray = np.array([[1.0, 0.0, 0.0]]),
            fill_mode:str="solid", # solid, majorwireframe, densewireframe, linestrips
            label: str = None,
            radius: float = 0.0015,
            world_tf_grasped_object: np.ndarray = None, # posquat with scalar first
            grasped_object_sizes: np.ndarray = None, # (3,) or (B, 3) with width, depth, height
        ) -> Boxes3D:
        grasped_object_centers = np.array([
            [0,0,0],
        ])
        grasped_object_orientations = R.from_quat([
            [0, 0, 0, 1],
        ])
        colors = np.tile(color, (1, 1)) # (1, 3) or (B, 3)
        num_grasped_objects = 1

        grasped_object_half_dims = np.array([
            [grasped_object_sizes[0]/2, grasped_object_sizes[1]/2, grasped_object_sizes[2]/2], # width, depth, height
        ])

        if world_tf_grasped_object is not None:
            num_grasped_objects = world_tf_grasped_object.shape[0]
            assert world_tf_grasped_object.ndim == 2, f"world_tf_gripper should be of shape (B,7), got {world_tf_grasped_object.shape}"
            assert world_tf_grasped_object.shape[1] == 7, f"world_tf_gripper should be None or of shape (7,), got {world_tf_grasped_object.shape}"
            # should have (3*B)x3 gripper centers and (3*B)x4 gripper orientations
            if num_grasped_objects > 1:
                # need to tile one of them and repeat the other
                # with tiling A,B becomes A,B,A,B while with repeat A,B becomes A,A,B,B
                # then repeat the gripper centers and orientations by B times

                grasped_object_centers = np.repeat(grasped_object_centers, num_grasped_objects, axis=0)
                grasped_object_orientations = R.concatenate(np.repeat(grasped_object_orientations, num_grasped_objects, axis=0))
                grasped_object_half_dims = np.repeat(grasped_object_half_dims, num_grasped_objects, axis=0)

                world_tf_grasped_object = np.tile(world_tf_grasped_object, (1,1))

                assert grasped_object_centers.shape[0] == num_grasped_objects, f"gripper_centers should be of shape (B,3), got {grasped_object_centers.shape}"
                assert len(grasped_object_orientations) == num_grasped_objects, f"gripper_orientations should be of shape (B,4), got {grasped_object_orientations.shape}"
                assert grasped_object_half_dims.shape[0] == num_grasped_objects, f"gripper_dims should be of shape (B,3), got {grasped_object_half_dims.shape}"
            
            assert colors.shape[0] == grasped_object_centers.shape[0], f"colors should be of shape (B,3), got {colors.shape}"

            # transform box centers and orientations to world frame
            world_rot_grasped_object = R.from_quat(world_tf_grasped_object[:, 3:7], scalar_first=True)
            grasped_object_centers = world_rot_grasped_object.apply(grasped_object_centers) + world_tf_grasped_object[:, :3]

            # gripper_centers = (world_tf_gripper[:3, :3] @ gripper_centers.T).T + world_tf_gripper[:3, 3]

            # gripper_orientations = R.from_matrix(world_tf_gripper[:3, :3] @ gripper_orientations.as_matrix())
            grasped_object_orientations = world_rot_grasped_object*grasped_object_orientations

        if fill_mode == "linestrips" and not self.only_log_tfs:
            # use the linestrips to draw the gripper
            grasped_object = linestrips3d_from_boxes(
                centers=grasped_object_centers,
                half_sizes=grasped_object_half_dims,
                quaternions=grasped_object_orientations.as_quat(),
                colors=colors,
                radii=[radius]*num_grasped_objects,
                # labels=[label]*3,
            )
        else:
            grasped_object = rr.Boxes3D(
            centers=grasped_object_centers,
            half_sizes=grasped_object_half_dims,
            quaternions=grasped_object_orientations.as_quat(),
            radii=[radius]*num_grasped_objects,
            colors=colors,
            fill_mode=fill_mode,
            # labels=[label]*3,
        )
        return grasped_object 
        
    def log_grasped_object_rerun(self, 
                           end_effector_pose: np.ndarray,
                           grasped_object_sizes: np.ndarray,
                           episode_timestamp: float,
                           episode_step: int,
                           rerun_name: str,
                           rerun_recording: rr.RecordingStream,
                           color: np.ndarray = np.array([1.0, 0.0, 0.0]),
                           radius: float = 0.0015,
                           ) -> None:
        rerun_recording.set_time("episode_timestamp", timestamp=episode_timestamp)
        rerun_recording.set_time("episode_step", sequence=episode_step)

        end_effector_to_grasped_object_translation = self.gripper_dimensions_helper.end_effector_to_grasped_object_translation(grasped_object_sizes)
        grasped_object_pose = end_effector_pose.copy()
        world_rot_end_effector = R.from_quat(end_effector_pose[3:7], scalar_first=True)
        grasped_object_pose[:3] += world_rot_end_effector.apply(end_effector_to_grasped_object_translation)

        if self.only_log_tfs:
            rerun_recording.log(rerun_name, 
                rr.Transform3D(
                    translation=grasped_object_pose[:3],
                    quaternion=grasped_object_pose[[4,5,6,3]], # needed to convert from [w,x,y,z] to [x,y,z,w]
                )
            )
        else:
            grasped_object = self.get_grasped_object(
                color=color[np.newaxis, :], # add batch dimension
                fill_mode="linestrips",
                label=None,
                radius=radius,
                world_tf_grasped_object=grasped_object_pose[np.newaxis, :], # add batch dimension
                grasped_object_sizes=grasped_object_sizes,
            )
            rerun_recording.log(rerun_name,
                grasped_object,
            )

    # def record_action_plan(self, action_plan: np.ndarray, obs: dict, action_frame_expression: str, input_rotation_representation: str) -> None:
    def record_action_plan(self, 
                           action_plan: np.ndarray, 
                           input_rotation_representation: str, 
                           rerun_recording:rr.RecordingStream, 
                           action_plan_cmap_str: str='cmr.swamp',
                           action_plan_cmap_clamp: Tuple[float, float]=(0.2, 0.8),
                           ) -> None:
        '''
        action_plan: T x 3+R+1 where first 3 are translation and last R are rotation and then 1 is gripper action
        should already be in world frame
        '''
        # assert action_frame_expression in ["relative", "delta"], f"action_frame_expression should be either relative or delta, got {action_frame_expression}"
        assert input_rotation_representation in ["axis_angle", "euler_angles", "quaternion"], f"input_rotation_representation should be axis_angle or euler_angles, got {input_rotation_representation}"

        action_plan_reparam = np.zeros((action_plan.shape[0], 7), dtype=np.float32)
        action_plan_reparam[:, :3] = action_plan[:, :3] # first 3 are translation
        if input_rotation_representation == "euler_angles":
            # convert euler angles to axis angle
            action_plan_reparam[:, 3:7] = R.from_euler('XYZ', action_plan[:, 3:6], degrees=False).as_quat(scalar_first=True)
        elif input_rotation_representation == "axis_angle":
            # convert axis angle to quaternion
            action_plan_reparam[:, 3:7] = R.from_rotvec(action_plan[:, 3:6]).as_quat(scalar_first=True)
        elif input_rotation_representation == "quaternion":
            # assume action_plan is already in quaternion format
            action_plan_reparam[:, 3:7] = action_plan[:, 3:7]
        action_plan = action_plan_reparam

        # end_effector_pose = obs['extra']['end_effector_pose'][0].cpu().numpy() # bx7->7
        # if action_frame_expression == "relative":
        #     if input_rotation_representation == "axis_angle":
        #         pass
        #     elif input_rotation_representation == "quaternion":
        #         # action_plan = np.zeros((self.action_plan_len, 6), dtype=np.float32)
        #         raise NotImplementedError("quaternion to axis_angle conversion not implemented")
        # elif action_frame_expression == "delta":
        #     current_target_pose = obs['agent']['controller']['arm']['target_pose'].cpu().numpy() # bx7->7
        #     target_pose_in_end_effector_frame = apply_transform_to_poses(end_effector_pose[np.newaxis, ...], current_target_pose, rotation_representation='quaternion')
        #     action_plan = unroll_delta_actions(
        #         delta_actions=torch.from_numpy(action_plan)[:, :6].unsqueeze(0), # accepts BxTx6
        #         init_pose=torch.from_numpy(target_pose_in_end_effector_frame), # accepts Bx7
        #         input_delta_rotation_representation=input_rotation_representation,
        #         output_rotation_representation='axis_angle',
        #     )[0]
        
        self.log_action_plan(
            action_plan=action_plan_reparam,
            # end_effector_pose=end_effector_pose,
            episode_timestamp=self.episode_step * self.sim_dt_bw_step,
            episode_step=self.episode_step,
            rerun_name=f"world/{self.policy_name}/action_plan",
            action_plan_cmap_str=action_plan_cmap_str,
            action_plan_cmap_clamp=action_plan_cmap_clamp,
            log_action_plan_lumped=self.log_action_plan_lumped,
            rerun_recording=rerun_recording,
        )

        # also log the action plan that is actually executed
        if self.log_executed_action_plan:
            self.log_action_plan(
                action_plan=action_plan_reparam[:self.executed_action_plan_length, :], # only log the executed part
                # end_effector_pose=end_effector_pose,
                episode_timestamp=self.episode_step * self.sim_dt_bw_step,
                episode_step=self.episode_step,
                rerun_name=f"world/{self.policy_name}/executed_action_plan",
                action_plan_cmap_str=action_plan_cmap_str,
                action_plan_cmap_clamp=action_plan_cmap_clamp,
                log_action_plan_lumped=self.log_action_plan_lumped,
                rerun_recording=rerun_recording,
            )
    
    def log_action_plan(
                        self,
                        action_plan: np.ndarray, # Tx6
                        # end_effector_pose: np.ndarray, # 7
                        episode_timestamp: float,
                        episode_step: int,
                        rerun_name: str,
                        action_plan_cmap_str: str,
                        action_plan_cmap_clamp: Tuple[float, float],
                        rerun_recording: rr.RecordingStream,
                        log_action_plan_lumped: bool=True,
                        ) -> None:
        '''
        Assumes action_plan is relative expression from the current_ end effector pose
        action_plan: T x 6+1 where first 3 are translation and last 3 are rotation (expressed in rotvec) and last 1 is gripper action
        '''
        rerun_recording.set_time("episode_timestamp", timestamp=episode_timestamp)
        rerun_recording.set_time("episode_step", sequence=episode_step)

        action_plan_length = action_plan.shape[0]

        # end_effector_orientation = R.from_quat(end_effector_pose[3:7], scalar_first=True)
        # action_plan = np.array(action_plan_msg.action_history).reshape(action_history_length, action_history_dim, order='C')    
        action_plan_orientations = R.from_quat(action_plan[:, 3:7], scalar_first=True)

        # action_plan_positions = action_plan[:, :3] + end_effector_pose[:3]
        action_plan_positions = action_plan[:, :3]
        # action_plan_orientations = action_plan_rotations*end_effector_orientation
        colors = np.array(cmr.take_cmap_colors(action_plan_cmap_str, action_plan_length, cmap_range=action_plan_cmap_clamp), dtype=np.float32)
        if not log_action_plan_lumped:
            for i, (position, orientation) in enumerate(zip(action_plan_positions, action_plan_orientations)):
                if self.only_log_tfs:
                    rerun_recording.log(rerun_name + "/gripper" + f"/{i}", rr.Transform3D(
                        translation=position,
                        mat3x3=orientation.as_matrix(),
                    ),
                    )
                else:
                    # color = self.action_plan_cmap_str(index)[:3] # ignore alpha
                    color = colors[i, :3] # ignore alpha
                    # world_tf_gripper = np.eye(4)
                    world_tf_gripper = np.zeros(7, dtype=np.float32)
                    world_tf_gripper[:3] = position
                    world_tf_gripper[3:7] = orientation.as_quat(scalar_first=True)
                    gripper = self.get_abstract_gripper(
                        color=color[np.newaxis, :], # add batch dimension
                        fill_mode="linestrips",
                        label=None,
                        radius=0.0005,
                        world_tf_gripper=world_tf_gripper
                    )
                    rerun_recording.log(f"{rerun_name}/gripper/{i}",
                        gripper,
                    )
        else:
            assert not self.only_log_tfs, "lumped action plan not supported with only_log_tfs"
            # color = self.action_plan_cmap(index)[:3] # ignore alpha

            action_plan = np.zeros((action_plan_length, 7), dtype=np.float32)
            action_plan[:, :3] = action_plan_positions
            action_plan[:, 3:7] = action_plan_orientations.as_quat(scalar_first=True)
            gripper = self.get_abstract_gripper(
                        color=colors, # add batch dimension
                        fill_mode="linestrips",
                        label=None,
                        radius=0.0005,
                        world_tf_gripper=action_plan
                    )
            rerun_recording.log(
                rerun_name + "/gripper",
                gripper,
            )

            if self.log_grasped_object:
                grasped_book_info = self.env.grasped_book_info
                end_effector_to_grasped_object_translation = self.gripper_dimensions_helper.end_effector_to_grasped_object_translation(grasped_book_info['sizes'][0])
                grasped_object_action_plan = action_plan.copy()
                world_rot_gripper = R.from_quat(grasped_object_action_plan[:, 3:7], scalar_first=True)
                grasped_object_action_plan[:, :3] += world_rot_gripper.apply(end_effector_to_grasped_object_translation)
                grasped_object = self.get_grasped_object(
                    color=colors, # add batch dimension
                    fill_mode="linestrips",
                    label=None,
                    radius=0.0005,
                    world_tf_grasped_object=grasped_object_action_plan,
                    grasped_object_sizes=grasped_book_info['sizes'][0],
                )
                rerun_recording.log(rerun_name + "/grasped_object",
                    grasped_object,
                )

        # if len(action_plan.gripper_action_trajectory) > 0:
        #     action_plan_gripper = np.array(action_plan.gripper_action_trajectory)
        
        #TODO log the arrows static but just update transforms
        # # Lx3x3 @ 3xN = LxNx3
        # basis_vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)*vector_length
        # action_plan_vectors = (action_plan_orientations.as_matrix() @ basis_vectors.T).transpose(0, 2, 1)
        # # base_vector_rgba = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]], dtype=np.float32)
        # # use amber turqoise violet
        # base_vector_rgba = np.array([[1, 191/255, 0, 1], [64/255, 224/255, 208/255, 1], [138/255, 43/255, 226/255, 1]], dtype=np.float32)

        # for i, (origin, vectors) in enumerate(zip(action_history_positions, action_plan_vectors)):
        #     current_vector_rgba = base_vector_rgba.copy()
        #     current_vector_rgba[:, 3] = (action_history_length - i)/action_history_length
        #     rr.log(rerun_name + f"/{i}", 
        #         rr.Arrows3D(
        #         origins=np.zeros((len(vectors), 3)),
        #         vectors=basis_vectors,
        #         colors=current_vector_rgba,
        #         labels=[action_plan_msg.header.seq]*3,
        #         ),
        #     static=True,
        #     )

    def reset(
        self,
        *args,
        seed: Optional[Union[int, List[int]]] = None,
        options: Optional[dict] = dict(),
        env_state_dict: Optional[dict] = None,
        action_history = None,
        generator_state = None,
        **kwargs,
    ):
        # if self.current_path_to_rrd is not None:
        #     rr.flush()
        if seed is not None:
            self.current_env_seed = seed

        # then close the rrd and start a new one
        if self.rerun_recording is not None:
            self.rerun_recording.flush()
            self.rerun_recording = None
            
        self._episode_id += 1

        self.episode_step = 0
        rrd_filename = f"traj_{self._episode_id}_seed_{self.current_env_seed}"
        self.init_new_rrd(rrd_filename, recording_id=self.recording_id, seed=self.current_env_seed)

        if 'action_history' in inspect.signature(self.env.reset).parameters:
            obs, info = self.env.reset(*args, seed=seed, options=options, action_history=action_history, generator_state=generator_state, **kwargs)
        else:
            obs, info = super().reset(*args, seed=seed, options=options, **kwargs)
        if env_state_dict is not None:
            self.env.set_state_dict(env_state_dict)
        if not self.only_log_action_plan:
            self.log_obs(obs, rerun_recording=self.rerun_recording)
        # adding the "batch" dimension as is done by common.batch is creating the temporal dimension such that each array is Txbx(data dims), where b is num envs
        
        self.last_reset_kwargs = copy.deepcopy(dict(options=options, **kwargs))
        if seed is not None:
            self.last_reset_kwargs.update(seed=seed)
        return obs, info

    def step(self, 
             action, 
             action_plan=None, 
             action_plan_rotation_representation=None, 
             start_signal=None, 
             env_state_dict=None, 
             action_plan_cmap_str='cmr.swamp', 
             action_plan_cmap_clamp=(0.2, 0.8),
             action_history=None, 
             generator_state=None,
        ):
        if action_plan is not None:
            assert action_plan_rotation_representation is not None, "action_plan_rotation_representation must be provided if action_plan is provided"
            assert action_plan.ndim == 3, f"action_plan must be of shape (B, T, 7), got {action_plan.shape}"
            self.record_action_plan(action_plan[0], action_plan_rotation_representation, rerun_recording=self.rerun_recording, action_plan_cmap_str=action_plan_cmap_str, action_plan_cmap_clamp=action_plan_cmap_clamp)

        # print(inspect.signature(super().step).parameters)
        # if 'action_plan' in inspect.signature(super().step).parameters:
        if 'action_plan' in inspect.signature(self.env.step).parameters:
            obs, rew, terminated, truncated, info = self.env.step(action, 
                                                                  action_plan=action_plan, 
                                                                  action_plan_rotation_representation=action_plan_rotation_representation, 
                                                                  action_history=action_history,
                                                                  generator_state=generator_state,
                                                                  )
        else:
            obs, rew, terminated, truncated, info = super().step(action)

        # snap to env state if provided
        if env_state_dict is not None:
            self.env.set_state_dict(env_state_dict)

        self.episode_step += 1

        if not self.only_log_action_plan:
            self.log_obs(obs, rerun_recording=self.rerun_recording)

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

    def eval_in_sim_starting_from_state(
            self,
            seed: int,
            env_state_dict: dict,
            obs_data: dict,
            policy,
            eval_length:int,
            step_within_trajectory: int,
            total_trajectory_length: int,
            init_generator_state = None,
        ):
        starting_end_effector_pose = obs_data['observation.state'][0,0,:7].cpu().numpy() # bxSx7->7
        ## start of reset
        from tqdm import tqdm
        
        obs, info = super().reset(seed=seed)
        self.env.set_state_dict(env_state_dict)
        ## end of reset

        # rerun_prefix = f"world/{self.policy_name}/step_{step_within_trajectory}"
        assert self.rollout_policy_name is not None, "rollout_policy_name must be set for eval_in_sim_starting_from_state"
        rerun_prefix = f"world/{self.rollout_policy_name}/branching_rollouts/{self.policy_name}"

        episode_trajectory_cmr_str = 'cmr.tropical'
        episode_trajectory_cmr_clamp = (0.0, 1.0)

        colors = np.array(cmr.take_cmap_colors(episode_trajectory_cmr_str, total_trajectory_length, cmap_range=episode_trajectory_cmr_clamp), dtype=np.float32)
        color = colors[step_within_trajectory, :3] # ignore alpha

        policy.reset(reset_inference_sampling_seed=False) # let the inference seed propagate across the trajectory, otherwise performance drops for some reason
        policy.generator_state = init_generator_state
        next_action_predicted, action_plan_predicted = policy.act(obs_data, obs_from_dataset=True)

        grasped_book_info = self.env.grasped_book_info
        prev_grasped_object_br_corner = None
        for i in tqdm(range(eval_length), desc="Simulating episode"):
            obs, rew, terminated, truncated, info = super().step(next_action_predicted)
            next_action_predicted, action_plan_predicted = policy.act(
                obs,
                obs_from_dataset=False,
            )
            self.log_grasped_object_rerun(
                obs['extra']['end_effector_pose'][0].cpu().numpy(),
                grasped_object_sizes= grasped_book_info['sizes'][0],
                episode_timestamp=(i+step_within_trajectory+1) * self.sim_dt_bw_step, # +1 because taken a step after the starting step from trajectory
                episode_step=(i+step_within_trajectory+1),
                rerun_name=rerun_prefix + f"/branch_trajectories/current/grasped_object/step_{step_within_trajectory}",
                color=color,
                radius=0.0015,
                rerun_recording=self.rerun_recording,
            )
            end_effector_to_grasped_object_bottom_right_corner_translation = torch.from_numpy(self.gripper_dimensions_helper.end_effector_to_grasped_object_bottom_right_corner_translation(grasped_book_info['sizes'][0]))

            grasped_object_bottom_right_corner_pose = obs['extra']['end_effector_pose'][0].clone()
            grasped_object_bottom_right_corner_pose[:3] += transforms.quaternion_apply(grasped_object_bottom_right_corner_pose[3:7],end_effector_to_grasped_object_bottom_right_corner_translation)
            if prev_grasped_object_br_corner is not None:
                # log linestrips from previous to current bottom right corner
                points = np.array([prev_grasped_object_br_corner[:3], grasped_object_bottom_right_corner_pose[:3]])
                self.rerun_recording.log(
                    rerun_prefix + f"/branch_trajectories/current/grasped_object_br_corner/step_{step_within_trajectory}",
                    rr.LineStrips3D(
                        points,
                        colors=np.array([color, color]),
                        radii=[0.0015, 0.0015],
                    ),
                )
            prev_grasped_object_br_corner = grasped_object_bottom_right_corner_pose.cpu().numpy()

            if info['success']:
                break
            elif not info['not_toppled']:
                break
        
        if info['success']: # policy succeeded
            # log the starting state as success
            self.log_grasped_object_rerun(
                end_effector_pose=starting_end_effector_pose,
                grasped_object_sizes=grasped_book_info['sizes'][0],
                episode_timestamp=step_within_trajectory * self.sim_dt_bw_step,
                episode_step=step_within_trajectory,
                rerun_name=rerun_prefix + "/main_trajectory/current/grasped_object",
                color=np.array([0.0, 1.0, 0.0]), # green
                radius=0.0015,
                rerun_recording=self.rerun_recording,
            )
        elif not info['success']: # policy failed
            if not info['not_toppled']: # policy toppled the book
                # log the starting state as toppling failure
                # also log the final state as a failure state
                self.log_grasped_object_rerun(
                    end_effector_pose=starting_end_effector_pose,
                    grasped_object_sizes=grasped_book_info['sizes'][0],
                    episode_timestamp=step_within_trajectory * self.sim_dt_bw_step,
                    episode_step=step_within_trajectory,
                    rerun_name=rerun_prefix + "/main_trajectory/current/grasped_object",
                    color=np.array([1.0, 0.0, 0.0]), # red
                    radius=0.0015,
                    rerun_recording=self.rerun_recording,
                )
            if (info['not_toppled']): # timed out
                # log the starting state as timeout failure
                # log the final state as a failure state
                self.log_grasped_object_rerun(
                    end_effector_pose=starting_end_effector_pose,
                    grasped_object_sizes=grasped_book_info['sizes'][0],
                    episode_timestamp=step_within_trajectory * self.sim_dt_bw_step,
                    episode_step=step_within_trajectory,
                    rerun_name=rerun_prefix + "/main_trajectory/current/grasped_object",
                    color=np.array([1.0, 0.5, 0.0]), # orange
                    radius=0.0015,
                    rerun_recording=self.rerun_recording,
                )

    def close(self) -> None:
        if self.rerun_recording is not None:
            self.rerun_recording.flush()
            self.rerun_recording.disconnect()
            self.rerun_recording = None
        
        # rr.disconnect()
        return super().close()

# %%
