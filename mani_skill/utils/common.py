"""
Common utilities often reused for internal code and task building for users.
"""

from collections import defaultdict
from typing import Dict, Optional, Sequence, Tuple, Union

import gymnasium as gym
import numpy as np
import sapien.physx as physx
import torch

from mani_skill.utils.structs.types import Array, Device

from pytorch3d import transforms

import einops

#%%
import sys, os
from pathlib import Path
# add contact_estimation to the path
path_to_this_file = Path(os.path.abspath(__file__))
#%%
path_to_contact_estimation = path_to_this_file.parents[3] / "contact_estimation"
sys.path.append(str(path_to_contact_estimation))
from src.dataset.gazebo_to_trimesh import generate_rays_from_camera, generate_min_distances_image, normals_to_xyz_map, get_min_grasped_obj_sdf_at_env_hits_data, get_min_env_sdf_at_grasped_obj_hits_data
#%%
# -------------------------------------------------------------------------- #
# Utilities for working with tensors, numpy arrays, and batched data
# -------------------------------------------------------------------------- #


def torch_clone_dict(data: dict) -> dict:
    """
    Recursively clones all torch tensors in a dictionary.
    If the input was a torch tensor, it will return a clone of the tensor.
    """
    if isinstance(data, torch.Tensor):
        return data.clone()

    output_dict = {}
    for key, value in data.items():
        if isinstance(value, dict):
            output_dict[key] = torch_clone_dict(value)
        elif isinstance(value, torch.Tensor):
            output_dict[key] = value.clone()
        else:
            output_dict[key] = value
    return output_dict


def _batch(array: Union[Array, Sequence]):
    if isinstance(array, (dict)):
        return {k: _batch(v) for k, v in array.items()}
    if isinstance(array, str):
        return array
    if isinstance(array, torch.Tensor):
        return array[None, :]
    if isinstance(array, np.ndarray):
        if array.shape == ():
            return array.reshape(1, 1)
        return array[None, :]
    if isinstance(array, list):
        if len(array) == 1:
            return [array]
    if (
        isinstance(array, float)
        or isinstance(array, int)
        or isinstance(array, bool)
        or isinstance(array, np.bool_)
    ):
        return np.array([[array]])
    return array


def batch(*args: Tuple[Union[Array, Sequence]]):
    """Adds one dimension in front of everything. If given a dictionary, every leaf in the dictionary
    has a new dimension. If given a tuple, returns the same tuple with each element batched"""
    x = [_batch(x) for x in args]
    if len(args) == 1:
        return x[0]
    return tuple(x)


# -------------------------------------------------------------------------- #
# Utilities for working with dictionaries
# -------------------------------------------------------------------------- #
def dict_merge(dct: dict, merge_dct: dict):
    """In place recursive merge of `merge_dct` into `dct`"""
    for k, v in merge_dct.items():
        if (
            k in dct and isinstance(dct[k], dict) and isinstance(merge_dct[k], dict)
        ):  # noqa
            dict_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]


# TODO (stao): Consolidate this function with the one above..
def merge_dicts(ds: Sequence[Dict], asarray=False):
    """Merge multiple dicts with the same keys to a single one."""
    # NOTE(jigu): To be compatible with generator, we only iterate once.
    ret = defaultdict(list)
    for d in ds:
        for k in d:
            ret[k].append(d[k])
    ret = dict(ret)
    # Sanity check (length)
    assert len(set(len(v) for v in ret.values())) == 1, "Keys are not same."
    if asarray:
        ret = {k: np.concatenate(v) for k, v in ret.items()}
    return ret


def append_dict_array(
    x1: Union[dict, Sequence, Array], x2: Union[dict, Sequence, Array]
):
    """Append `x2` in front of `x1` and returns the result. Tries to do this in place if possible.
    Assumes both `x1, x2` have the same dictionary structure if they are dictionaries.
    They may also both be lists/sequences in which case this is just appending like normal"""
    if isinstance(x1, np.ndarray):
        if len(x1.shape) > len(x2.shape):
            # if different dims, check if extra dim is just a 1 due to single env in batch mode and if so, add it to x2.
            if x1.shape[1] == 1:
                x2 = x2[:, None, :]
            elif x1.shape[0] == 1:
                x2 = x2[None, ...]
        return np.concatenate([x1, x2])
    elif isinstance(x1, list):
        return x1 + x2
    elif isinstance(x1, dict):
        for k in x1.keys():
            assert k in x2, "dct and append_dct need to have the same dictionary layout"
            x1[k] = append_dict_array(x1[k], x2[k])
    return x1


def index_dict_array(x1, idx: Union[int, slice], inplace=True):
    """Indexes every array in x1 with slice and returns result."""
    if (
        isinstance(x1, np.ndarray)
        or isinstance(x1, list)
        or isinstance(x1, torch.Tensor)
    ):
        return x1[idx]
    elif isinstance(x1, dict):
        if inplace:
            for k in x1.keys():
                x1[k] = index_dict_array(x1[k], idx, inplace=inplace)
            return x1
        else:
            out = dict()
            for k in x1.keys():
                out[k] = index_dict_array(x1[k], idx, inplace=inplace)
            return out


# TODO (stao): this code can be simplified
def to_tensor(array: Array, device: Optional[Device] = None):
    """
    Maps any given sequence to a torch tensor on the CPU/GPU. If physx gpu is not enabled then we use CPU, otherwise GPU, unless specified
    by the device argument

    Args:
        array: The data to map to a tensor
        device: The device to put the tensor on. By default this is None and to_tensor will put the device on the GPU if physx is enabled
            and CPU otherwise

    """
    if isinstance(array, (dict)):
        return {k: to_tensor(v, device=device) for k, v in array.items()}
    elif isinstance(array, torch.Tensor):
        ret = array.to(device)
    elif isinstance(array, np.ndarray):
        # TODO (stao): check of doing .to(device) is slow even if its just CPU
        if array.dtype == np.uint16:
            array = array.astype(np.int32)
        elif array.dtype == np.uint32:
            array = array.astype(np.int64)
        ret = torch.tensor(array).to(device)
    else:
        if isinstance(array, list) and isinstance(array[0], np.ndarray):
            array = np.array(array)
        ret = torch.tensor(array, device=device)
    if ret.dtype == torch.float64:
        ret = ret.to(torch.float32)
    return ret


def to_cpu_tensor(array: Array):
    """
    Maps any given sequence to a torch tensor on the CPU.
    """
    if isinstance(array, (dict)):
        return {k: to_cpu_tensor(v) for k, v in array.items()}
    if isinstance(array, np.ndarray):
        ret = torch.from_numpy(array)
        if ret.dtype == torch.float64:
            ret = ret.float()
        return ret
    elif isinstance(array, torch.Tensor):
        return array.cpu()
    else:
        return torch.tensor(array).cpu()


# TODO (stao): Clean up this code
def flatten_state_dict(
    state_dict: dict, use_torch=False, device: Optional[Device] = None
) -> Array:
    """Flatten a dictionary containing states recursively. Expects all data to be either torch or numpy

    Args:
        state_dict: a dictionary containing scalars or 1-dim vectors.
        use_torch (bool): Whether to convert the data to torch tensors.

    Raises:
        AssertionError: If a value of @state_dict is an ndarray with ndim > 2.

    Returns:
        np.ndarray | torch.Tensor: flattened states.

    Notes:
        The input is recommended to be ordered (e.g. dict).
        However, since python 3.7, dictionary order is guaranteed to be insertion order.
    """
    states = []

    for key, value in state_dict.items():
        if isinstance(value, dict):
            state = flatten_state_dict(value, use_torch=use_torch)
            if state.nelement() == 0:
                state = None
            elif use_torch:
                state = to_tensor(state, device=device)
        elif isinstance(value, (tuple, list)):
            state = None if len(value) == 0 else value
            if use_torch:
                state = to_tensor(state, device=device)
        elif isinstance(value, (bool, np.bool_, int, np.int32, np.int64)):
            # x = np.array(1) > 0 is np.bool_ instead of ndarray
            state = int(value)
            if use_torch:
                state = to_tensor(state, device=device)
        elif isinstance(value, (float, np.float32, np.float64)):
            state = np.float32(value)
            if use_torch:
                state = to_tensor(state, device=device)
        elif isinstance(value, np.ndarray):
            if value.ndim > 2:
                raise AssertionError(
                    "The dimension of {} should not be more than 2.".format(key)
                )
            state = value if value.size > 0 else None
            if use_torch:
                state = to_tensor(state, device=device)

        elif isinstance(value, torch.Tensor):
            state = value
            if len(state.shape) == 1:
                state = state[:, None]
        else:
            raise TypeError("Unsupported type: {}".format(type(value)))
        if state is not None:
            states.append(state)

    if use_torch:
        if len(states) == 0:
            return torch.empty(0, device=device)
        else:
            return torch.hstack(states)
    else:
        if len(states) == 0:
            return np.empty(0)
        else:
            return np.hstack(states)


def flatten_dict_keys(d: dict, prefix=""):
    """Flatten a dict by expanding its keys recursively."""
    out = dict()
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten_dict_keys(v, prefix + k + "/"))
        else:
            out[prefix + k] = v
    return out


def normalize_vector(x: torch.Tensor, eps=1e-6):
    """normalizes a given torch tensor x and if the norm is less than eps, set the norm to 0"""
    norm = torch.linalg.norm(x, axis=1)
    norm[norm < eps] = 1
    norm = 1 / norm
    return torch.multiply(x, norm[:, None])


def np_normalize_vector(x, eps=1e-6):
    """normalizes a given numpy array x and if the norm is less than eps, set the norm to 0"""
    x = np.asarray(x)
    assert x.ndim == 1, x.ndim
    norm = np.linalg.norm(x)
    return np.zeros_like(x) if norm < eps else (x / norm)


def np_compute_angle_between(x1: np.ndarray, x2: np.ndarray):
    """Compute angle (radian) between two numpy arrays"""
    x1, x2 = np_normalize_vector(x1), np_normalize_vector(x2)
    dot_prod = np.clip(np.dot(x1, x2), -1, 1)
    return np.arccos(dot_prod).item()


def compute_angle_between(x1: torch.Tensor, x2: torch.Tensor):
    """Compute angle (radian) between two torch tensors"""
    x1, x2 = normalize_vector(x1), normalize_vector(x2)
    dot_prod = torch.clip(torch.einsum("ij,ij->i", x1, x2), -1, 1)
    return torch.arccos(dot_prod)


# TODO (stao): verfy torch.jit.script provides actual speedups in inference times
def quat_diff_rad(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Get the difference in radians between two quaternions.

    Args:
        a: first quaternion, shape (N, 4)
        b: second quaternion, shape (N, 4)
    Returns:
        Difference in radians, shape (N,)
    """
    # Normalize the quaternions
    a = a / torch.norm(a, dim=1, keepdim=True)
    b = b / torch.norm(b, dim=1, keepdim=True)

    # Compute the dot product between the quaternions
    dot_product = torch.sum(a * b, dim=1)

    # Clamp the dot product to the range [-1, 1] to avoid numerical instability
    dot_product = torch.clamp(dot_product, -1.0, 1.0)

    # Compute the angle difference in radians
    angle_diff = 2 * torch.acos(torch.abs(dot_product))

    return angle_diff


def _unbatch(array: Union[Array, Sequence]):
    if isinstance(array, (dict)):
        return {k: _unbatch(v) for k, v in array.items()}
    if isinstance(array, str):
        return array
    if isinstance(array, torch.Tensor):
        return array.squeeze(0)
    if isinstance(array, np.ndarray):
        if array.shape == (1,):
            return array.item()
        if np.iterable(array) and array.shape[0] == 1:
            return array.squeeze(0)
    if isinstance(array, list):
        if len(array) == 1:
            return array[0]
    return array


def unbatch(*args: Tuple[Union[Array, Sequence]]):
    x = [_unbatch(x) for x in args]
    if len(args) == 1:
        return x[0]
    return tuple(x)


def _to_numpy(array: Union[Array, Sequence]) -> np.ndarray:
    if isinstance(array, (dict)):
        return {k: _to_numpy(v) for k, v in array.items()}
    if isinstance(array, torch.Tensor):
        return array.cpu().numpy()
    if (
        isinstance(array, np.ndarray)
        or isinstance(array, bool)
        or isinstance(array, str)
        or isinstance(array, float)
        or isinstance(array, int)
    ):
        return array
    else:
        return np.array(array)


def to_numpy(array: Union[Array, Sequence], dtype=None) -> np.ndarray:
    array = _to_numpy(array)
    if dtype is not None:
        return array.astype(dtype)
    return array


# -------------------------------------------------------------------------- #
# Utilities for working with quaternions
# -------------------------------------------------------------------------- #


# -------------------------------------------------------------------------- #
# Leon: Utilities for changing parameterizations of actions
# -------------------------------------------------------------------------- #

def apply_transform_to_poses(current_pose, all_target_poses_in_world, rotation_representation='axis_angle', output_rotation_representation='axis_angle', mode:str = 'subtract'):
    '''
    Args:
        current_pose: (1, 7)
        all_target_poses_in_world: (N, 7)
    Returns:
        all_target_poses_in_current_pose: (N, 3+R) where R is 3 for axis angle or 4 for quaternion or 6 for 6D representation 
    '''
    assert rotation_representation in ['axis_angle', 'quaternion', '6d'], f"rotation_representation {rotation_representation} not supported"
    assert current_pose.ndim == 2, f"current_pose shape {current_pose.shape} not supported"
    assert all_target_poses_in_world.ndim == 2, f"all_target_poses_in_world shape {all_target_poses_in_world.shape} not supported"
    assert mode in ['subtract', 'add'], f"mode {mode} not supported, must be either 'subtract' or 'add'" 
    convert_back_to_numpy = False
    if isinstance(current_pose, np.ndarray):
        current_pose = torch.from_numpy(current_pose)
        convert_back_to_numpy = True
    if isinstance(all_target_poses_in_world, np.ndarray):
        all_target_poses_in_world = torch.from_numpy(all_target_poses_in_world)
        convert_back_to_numpy = True

    if rotation_representation == 'axis_angle':
        rotation_dimension = 3
    elif rotation_representation == 'quaternion':
        rotation_dimension = 4
    elif rotation_representation == '6d':
        rotation_dimension = 6

    if mode == 'subtract':
        current_pose[:, 0:3] = -current_pose[:, 0:3]  # Invert the translation part
        current_pose[:, 3:7] = transforms.quaternion_invert(current_pose[:, 3:7])  # Invert the rotation part

    plan_target_poses_in_current_pose = torch.zeros((all_target_poses_in_world.shape[0], 3+rotation_dimension), dtype=torch.float32, device=current_pose.device)
    # plan_target_poses_in_current_pose[:, 0:3] = all_target_poses_in_world[:, 0:3] - current_pose[:, 0:3]
    plan_target_poses_in_current_pose[:, 0:3] = all_target_poses_in_world[:, 0:3] + current_pose[:, 0:3]
    # target_rotation_from_current_pose = transforms.quaternion_multiply(all_target_poses_in_world[:, 3:7], transforms.quaternion_invert(current_pose[:, 3:7]))
    target_rotation_from_current_pose = transforms.quaternion_multiply(all_target_poses_in_world[:, 3:7], current_pose[:, 3:7])
    if rotation_representation == 'axis_angle':
        plan_target_poses_in_current_pose[:, 3:] = transforms.quaternion_to_axis_angle(target_rotation_from_current_pose)
    elif rotation_representation == 'quaternion':
        plan_target_poses_in_current_pose[:, 3:] = target_rotation_from_current_pose
    elif rotation_representation == '6d':
        target_rotation_from_current_pose_matrix = transforms.quaternion_to_matrix(target_rotation_from_current_pose)
        plan_target_poses_in_current_pose[:, 3:] = transforms.matrix_to_rotation_6d(target_rotation_from_current_pose_matrix)
    if convert_back_to_numpy:
        plan_target_poses_in_current_pose = plan_target_poses_in_current_pose.cpu().numpy()
    return plan_target_poses_in_current_pose


from scipy.spatial.transform import Rotation as R

def get_plan_target_poses_in_current_pose_scipy(current_pose, all_target_poses_in_world, rotation_representation='axis_angle'):
    '''
    Args:
        current_pose: (1, 7)
        all_target_poses_in_world: (N, 7)
    Returns:
        all_target_poses_in_current_pose: (N, 3+R) where R is 3 for axis angle or 4 for quaternion or 6 for 6D representation 
    '''
    assert rotation_representation in ['axis_angle', 'quaternion',], f"rotation_representation {rotation_representation} not supported"
    assert len(current_pose.shape) == 2, f"current_pose shape {current_pose.shape} not supported"
    assert len(all_target_poses_in_world.shape) == 2, f"all_target_poses_in_world shape {all_target_poses_in_world.shape} not supported"
    assert len(current_pose.shape) == 2, f"current_pose shape {current_pose.shape} not supported"
    if rotation_representation == 'axis_angle':
        rotation_dimension = 3
    elif rotation_representation == 'quaternion':
        rotation_dimension = 4
    elif rotation_representation == '6d':
        rotation_dimension = 6

    plan_target_poses_in_current_pose = np.zeros((all_target_poses_in_world.shape[0], 3+rotation_dimension), dtype=np.float32)
    plan_target_poses_in_current_pose[:, 0:3] = all_target_poses_in_world[:, 0:3] - current_pose[:, 0:3]
    target_rotation_from_current_pose = R.from_quat(all_target_poses_in_world[:, 3:7], scalar_first=True) * R.from_quat(current_pose[:, 3:7], scalar_first=True).inv()
    if rotation_representation == 'axis_angle':
        plan_target_poses_in_current_pose[:, 3:] = target_rotation_from_current_pose.as_rotvec()
    elif rotation_representation == 'quaternion':
        plan_target_poses_in_current_pose[:, 3:] = target_rotation_from_current_pose.as_quat(scalar_first=True)
    return plan_target_poses_in_current_pose

def unroll_delta_actions(delta_actions, init_pose, input_delta_rotation_representation='axis_angle', output_rotation_representation='axis_angle', translation_frame_convention='root', rotation_frame_convention='root'):
    '''
    init_pose: (B, 7)
    delta_actions: (B, N, 6)
    input_rotation_representation: 'axis_angle' or 'quaternion'
    output_rotation_representation: 'axis_angle' or 'quaternion' or 'euler_angles'
    See https://maniskill.readthedocs.io/en/latest/user_guide/concepts/controllers.html#delta-control for frame convention details
    translation_frame_convention: 'body' or 'root'
    rotation_frame_convention: 'body' or 'root'
    Returns:
        gt_target_poses_in_current_pose: (B, N, 3+R) where R is 3 for axis angle or 4 for quaternion
    '''
    assert translation_frame_convention in ['body', 'root'], f"translation_frame_convention {translation_frame_convention} not supported"
    assert rotation_frame_convention in ['body', 'root'], f"rotation_frame_convention {rotation_frame_convention} not supported"
    assert input_delta_rotation_representation in ['axis_angle', 'quaternion', 'euler_angles'], f"input_delta_rotation_representation {input_delta_rotation_representation} not supported"
    assert output_rotation_representation in ['axis_angle', 'quaternion', 'euler_angles'], f"output_rotation_representation {output_rotation_representation} not supported"
    assert len(init_pose.shape) == 2, f"init_pose shape {init_pose.shape} not supported"
    assert len(delta_actions.shape) == 3, f"delta_actions shape {delta_actions.shape} not supported"
    assert init_pose.shape[0] == delta_actions.shape[0], f"init_pose shape {init_pose.shape} and delta_actions shape {delta_actions.shape} not match"
    assert init_pose.shape[1] == 7, f"init_pose shape {init_pose.shape} not supported"

    B, N = delta_actions.shape[:2]

    if output_rotation_representation in ['axis_angle', 'euler_angles']:
        rotation_dim = 3
    elif output_rotation_representation == 'quaternion':
        rotation_dim = 4

    gt_target_poses_in_current_pose = torch.zeros(delta_actions.shape[:2] + (3+rotation_dim,), dtype=torch.float32, device=delta_actions.device)
    if translation_frame_convention == 'root':
        gt_target_poses_in_current_pose[:,:,:3] = init_pose[:, :3].unsqueeze(1) + torch.cumsum(delta_actions[:, :, :3], dim=1)

    if input_delta_rotation_representation == 'axis_angle':
        assert delta_actions.shape[2] == 6, f"delta_actions shape {delta_actions.shape} not match with input_delta_rotation_representation {input_delta_rotation_representation}"
        gt_delta_quaternions = transforms.axis_angle_to_quaternion(delta_actions[:, :, 3:6])
    elif input_delta_rotation_representation == 'quaternion':
        assert delta_actions.shape[2] == 7, f"delta_actions shape {delta_actions.shape} not match with input_delta_rotation_representation {input_delta_rotation_representation}"
        gt_delta_quaternions = delta_actions[:, :, 3:7]
    elif input_delta_rotation_representation == 'euler_angles':
        assert delta_actions.shape[2] == 6, f"delta_actions shape {delta_actions.shape} not match with input_delta_rotation_representation {input_delta_rotation_representation}"
        gt_delta_quaternions = transforms.matrix_to_quaternion(transforms.euler_angles_to_matrix(delta_actions[:, :, 3:6], convention='XYZ'))
    
    gt_target_rotations_in_current_pose = torch.zeros((B, N+1, 4), dtype=torch.float32, device=delta_actions.device)
    gt_target_rotations_in_current_pose[:, 0] = init_pose[:, 3:7]  # initial rotation is the same as the initial pose
    for i in range(N):
        # if i == 0:
        #     gt_target_rotations_in_current_pose[:, i] = transforms.quaternion_multiply(gt_delta_quaternions[:, i], init_pose[:, 3:7])
        #     # gt_target_poses_in_current_pose[:, i, :3] = transforms.quaternion_apply(init_pose[:, 3:7], delta_actions[:, i, :3]) + init_pose[:, :3]
        # else:
        if rotation_frame_convention == 'root':
            gt_target_rotations_in_current_pose[:, i+1] = transforms.quaternion_multiply(gt_delta_quaternions[:, i], gt_target_rotations_in_current_pose[:, i])
        elif rotation_frame_convention == 'body':
            gt_target_rotations_in_current_pose[:, i+1] = transforms.quaternion_multiply(gt_target_rotations_in_current_pose[:, i], gt_delta_quaternions[:, i])
        if translation_frame_convention == 'body':
            gt_target_poses_in_current_pose[:, i, :3] = transforms.quaternion_apply(gt_target_rotations_in_current_pose[:, i], delta_actions[:, i, :3]) + gt_target_poses_in_current_pose[:, i, :3]
    gt_target_rotations_in_current_pose = gt_target_rotations_in_current_pose[:, 1:]  # remove the initial rotation

    if output_rotation_representation == 'axis_angle':
        gt_target_poses_in_current_pose[:, :, 3:6] = transforms.quaternion_to_axis_angle(gt_target_rotations_in_current_pose)
    elif output_rotation_representation == 'quaternion':
        gt_target_poses_in_current_pose[:, :, 3:7] = gt_target_rotations_in_current_pose
    elif output_rotation_representation == 'euler_angles':
        gt_target_orientations_in_current_pose = transforms.quaternion_to_matrix(gt_target_rotations_in_current_pose)
        gt_target_poses_in_current_pose[:, :, 3:6] = transforms.matrix_to_euler_angles(gt_target_orientations_in_current_pose, convention='XYZ')
    return gt_target_poses_in_current_pose

def get_delta_actions_from_plan_target_poses(plan_target_poses, gripper_actions=None, input_rotation_representation='quaternion', output_rotation_representation='euler_angles', translation_frame_convention='root', rotation_frame_convention='root'):
    '''
    Args:
        NOTE: if frame convention is root, then poses MUST be expressed in the root frame!
        plan_target_poses: (N, 3+R) where R is 3 for axis angle or 4 for quaternion or 6 for 6D representation. Must be plan target poses expressed in a common frame
        gripper_actions: (N-1)
    Returns:
        delta_actions_from_plan_target_poses: (N-1, 4+R) where the last element is the gripper action
    '''
    assert translation_frame_convention in ['body', 'root'], f"translation_frame_convention {translation_frame_convention} not supported"
    assert rotation_frame_convention in ['body', 'root'], f"rotation_frame_convention {rotation_frame_convention} not supported"
    assert input_rotation_representation in ['axis_angle', 'quaternion', '6d', 'euler_angles'], f"rotation_representation {input_rotation_representation} not supported"
    assert output_rotation_representation in ['axis_angle', 'quaternion', '6d', 'euler_angles'], f"rotation_representation {output_rotation_representation} not supported"
    if gripper_actions is not None:
        assert gripper_actions.shape[0] == plan_target_poses.shape[0]-1, f"gripper_actions shape {gripper_actions.shape} and plan_target_poses shape {plan_target_poses.shape} not match"

    if translation_frame_convention == 'body':
        raise NotImplementedError("translation_frame_convention 'body' is not implemented yet")
    if rotation_frame_convention == 'body':
        raise NotImplementedError("rotation_frame_convention 'body' is not implemented yet")
    
    if input_rotation_representation in ['axis_angle', 'euler_angles']:
        assert plan_target_poses.shape[1] == 6, f"plan_target_poses shape {plan_target_poses.shape} not supported for axis_angle representation"
    elif input_rotation_representation == 'quaternion':
        assert plan_target_poses.shape[1] == 7, f"plan_target_poses shape {plan_target_poses.shape} not supported for quaternion representation"
    elif input_rotation_representation == '6d':
        assert plan_target_poses.shape[1] == 9, f"plan_target_poses shape {plan_target_poses.shape} not supported for 6d representation"
    
    if output_rotation_representation in ['axis_angle', 'euler_angles']:
        output_rotation_dim = 3
    elif output_rotation_representation == 'quaternion':
        output_rotation_dim = 4
    elif output_rotation_representation == '6d':
        output_rotation_dim = 6
    
    delta_actions_from_plan_target_poses = torch.zeros((plan_target_poses.shape[0]-1, 3+output_rotation_dim+(1 if gripper_actions is not None else 0)), dtype=torch.float32, device=plan_target_poses.device)
    
    delta_actions_from_plan_target_poses[:, :3] = torch.diff(plan_target_poses[:, :3], dim=0)
    
    if input_rotation_representation == 'axis_angle':
        current_plan_target_quaternions = transforms.axis_angle_to_quaternion(plan_target_poses[:-1, 3:])
        next_plan_target_quaternions = transforms.axis_angle_to_quaternion(plan_target_poses[1:, 3:])
    elif input_rotation_representation == 'euler_angles':
        current_plan_target_quaternions = transforms.matrix_to_quaternion(transforms.euler_angles_to_matrix(plan_target_poses[:-1, 3:6], convention='XYZ'))
        next_plan_target_quaternions = transforms.matrix_to_quaternion(transforms.euler_angles_to_matrix(plan_target_poses[1:, 3:6], convention='XYZ'))
    elif input_rotation_representation == 'quaternion':
        current_plan_target_quaternions = plan_target_poses[:-1, 3:7]
        next_plan_target_quaternions = plan_target_poses[1:, 3:7]
    elif input_rotation_representation == '6d':
        current_plan_target_quaternions = transforms.matrix_to_quaternion(transforms.rotation_6d_to_matrix(plan_target_poses[:-1, 3:]))
        next_plan_target_quaternions = transforms.matrix_to_quaternion(transforms.rotation_6d_to_matrix(plan_target_poses[1:, 3:]))
    
    delta_action_rotation = transforms.quaternion_multiply(next_plan_target_quaternions, transforms.quaternion_invert(current_plan_target_quaternions))

    if output_rotation_representation == 'axis_angle':
        delta_actions_from_plan_target_poses[:, 3:6] = transforms.quaternion_to_axis_angle(delta_action_rotation)
    elif output_rotation_representation == 'euler_angles':
        delta_actions_from_plan_target_poses[:, 3:6] = transforms.matrix_to_euler_angles(transforms.quaternion_to_matrix(delta_action_rotation), convention='XYZ')
    elif output_rotation_representation == 'quaternion':
        delta_actions_from_plan_target_poses[:, 3:7] = delta_action_rotation
    elif output_rotation_representation == '6d':
        delta_actions_from_plan_target_poses[:, 3:9] = transforms.matrix_to_rotation_6d(transforms.quaternion_to_matrix(delta_action_rotation))

    if gripper_actions is not None:
        delta_actions_from_plan_target_poses[:, -1] = gripper_actions # the last element is the gripper action
    return delta_actions_from_plan_target_poses

def change_rotation_representation(poses, gripper_actions=None, input_rotation_representation='axis_angle', output_rotation_representation='axis_angle'):
    '''
    Args:
        poses: (N, 3+R) where R is 3 for axis angle or 4 for quaternion or 6 for 6D representation
        gripper_actions: (N), if provided, will be appended to the end of the output poses
        input_rotation_representation: 'axis_angle' or 'quaternion' or '6d'
        output_rotation_representation: 'axis_angle' or 'quaternion' or '6d'
    '''
    assert input_rotation_representation in ['axis_angle', 'quaternion', '6d', 'euler_angles'], f"input_rotation_representation {input_rotation_representation} not supported"
    assert output_rotation_representation in ['axis_angle', 'quaternion', '6d', 'euler_angles'], f"output_rotation_representation {output_rotation_representation} not supported"
    assert poses.ndim == 2, f"poses shape {poses.shape} not supported"
    if gripper_actions is not None:
        assert gripper_actions.ndim == 1, f"gripper_actions shape {gripper_actions.shape} not supported"
        assert gripper_actions.shape[0] == poses.shape[0], f"gripper_actions shape {gripper_actions.shape} and poses shape {poses.shape} not match"
    
    convert_back_to_numpy = False
    if isinstance(poses, np.ndarray):
        poses = torch.from_numpy(poses)
        convert_back_to_numpy = True
    if isinstance(gripper_actions, np.ndarray):
        gripper_actions = torch.from_numpy(gripper_actions)
        convert_back_to_numpy = True

    if input_rotation_representation == 'axis_angle':
        assert poses.shape[1] == 6, f"poses shape {poses.shape} not supported for axis_angle representation"
    elif input_rotation_representation == 'quaternion':
        assert poses.shape[1] == 7, f"poses shape {poses.shape} not supported for quaternion representation"
    elif input_rotation_representation == '6d':
        assert poses.shape[1] == 9, f"poses shape {poses.shape} not supported for 6d representation"
    elif input_rotation_representation == 'euler_angles':
        assert poses.shape[1] == 6, f"poses shape {poses.shape} not supported for euler_angles representation"

    if output_rotation_representation == 'axis_angle':
        output_rotation_dim = 3
    elif output_rotation_representation == 'quaternion':
        output_rotation_dim = 4
    elif output_rotation_representation == '6d':
        output_rotation_dim = 6
    elif output_rotation_representation == 'euler_angles':
        output_rotation_dim = 3

    total_dim = 3 + output_rotation_dim + (1 if gripper_actions is not None else 0)
    output_poses = torch.zeros((poses.shape[0], total_dim), dtype=torch.float32, device=poses.device)
    output_poses[:, :3] = poses[:, :3]  # copy the translation part
    if input_rotation_representation == 'axis_angle':
        current_quaternions = transforms.axis_angle_to_quaternion(poses[:, 3:6])
    elif input_rotation_representation == 'quaternion':
        current_quaternions = poses[:, 3:7]
    elif input_rotation_representation == '6d':
        current_rot_matrices = transforms.rotation_6d_to_matrix(poses[:, 3:9])
        current_quaternions = transforms.matrix_to_quaternion(current_rot_matrices)
    elif input_rotation_representation == 'euler_angles':
        current_rot_matrices = transforms.euler_angles_to_matrix(poses[:, 3:6], convention='XYZ')
        current_quaternions = transforms.matrix_to_quaternion(current_rot_matrices)
    assert current_quaternions.shape[1] == 4, f"current_quaternions shape {current_quaternions.shape} not supported"

    if output_rotation_representation == 'axis_angle':
        output_poses[:, 3:6] = transforms.quaternion_to_axis_angle(current_quaternions)
    elif output_rotation_representation == 'quaternion':
        output_poses[:, 3:7] = current_quaternions
    elif output_rotation_representation == '6d':
        output_poses[:, 3:9] = transforms.matrix_to_rotation_6d(transforms.quaternion_to_matrix(current_quaternions))
    elif output_rotation_representation == 'euler_angles':
        output_poses[:, 3:6] = transforms.matrix_to_euler_angles(transforms.quaternion_to_matrix(current_quaternions), convention='XYZ')
    if gripper_actions is not None:
        output_poses[:, -1] = gripper_actions
    if convert_back_to_numpy:
        output_poses = output_poses.cpu().numpy()
    return output_poses

def compute_delta_action(prev_target_pose, current_action, input_rotation_representation, output_rotation_representation='axis_angle'):
    '''
    IMPORTANT NOTE: assumes actions are both translation and rotation root aligned! See maniskill documentation for details. Hence we dont need to couple the rotation and translation parts of the action.
    Args:
        prev_target_pose: Bx(3+R) where R is 3 for axis angle or 4 for quaternion or 6 for 6D representation
        current_action: Bx(3+R+1) where R is 3 for axis angle or 4 for quaternion or 6 for 6D representation, +1 for gripper action
        input_rotation_representation: 'axis_angle' or 'quaternion' or '6d'
    Returns:
        delta_action: Bx(4+R) where R is 3 for axis angle or 4 for quaternion or 6 for 6D representation
    '''
    assert input_rotation_representation in ['axis_angle', 'quaternion', '6d', 'euler_angles'], f"input_rotation_representation {input_rotation_representation} not supported"
    assert output_rotation_representation in ['axis_angle', 'quaternion', '6d', 'euler_angles'], f"output_rotation_representation {output_rotation_representation} not supported"
    assert prev_target_pose.ndim == 2, f"prev_target_pose shape {prev_target_pose.shape} not supported"
    assert current_action.ndim == 2, f"current_action shape {current_action.shape} not supported"
    assert prev_target_pose.shape[0] == current_action.shape[0], f"prev_target_pose shape {prev_target_pose.shape} and current_action shape {current_action.shape} not match"
    
    B = prev_target_pose.shape[0]

    output_numpy = False
    if isinstance(prev_target_pose, np.ndarray):
        prev_target_pose = torch.from_numpy(prev_target_pose)
        output_numpy = True
    if isinstance(current_action, np.ndarray):
        current_action = torch.from_numpy(current_action)
        output_numpy = True
    
    if input_rotation_representation == 'axis_angle':
        assert prev_target_pose.shape[1] == 6, f"prev_target_pose shape {prev_target_pose.shape} not supported for axis_angle representation"
        assert current_action.shape[1] == 7, f"current_action shape {current_action.shape} not supported for axis_angle representation"
    elif input_rotation_representation == 'quaternion':
        assert prev_target_pose.shape[1] == 7, f"prev_target_pose shape {prev_target_pose.shape} not supported for quaternion representation"
        assert current_action.shape[1] == 8, f"current_action shape {current_action.shape} not supported for quaternion representation"
    elif input_rotation_representation == '6d':
        assert prev_target_pose.shape[1] == 9, f"prev_target_pose shape {prev_target_pose.shape} not supported for 6d representation"
        assert current_action.shape[1] == 10, f"current_action shape {current_action.shape} not supported for 6d representation"
    elif input_rotation_representation == 'euler_angles':
        assert prev_target_pose.shape[1] == 6, f"prev_target_pose shape {prev_target_pose.shape} not supported for euler_angles representation"
        assert current_action.shape[1] == 7, f"current_action shape {current_action.shape} not supported for euler_angles representation"

    if output_rotation_representation in ['axis_angle', 'euler_angles']:
        rotation_dim = 3
    elif output_rotation_representation == 'quaternion':
        rotation_dim = 4
    elif output_rotation_representation == '6d':
        rotation_dim = 6

    delta_action = torch.zeros((B, 3 + rotation_dim + 1,), dtype=torch.float32, device=current_action.device)
    delta_action[:, :3] = current_action[:, :3] - prev_target_pose[:, :3]
    delta_action[:, -1] = current_action[:, -1]  # gripper action
    if input_rotation_representation == 'axis_angle':
        current_quaternions = transforms.axis_angle_to_quaternion(current_action[:, 3:6])
        prev_quaternions = transforms.axis_angle_to_quaternion(prev_target_pose[:, 3:6])
    elif input_rotation_representation == 'quaternion':
        current_quaternions = current_action[:, 3:7]
        prev_quaternions = prev_target_pose[:, 3:7]
    elif input_rotation_representation == '6d':
        current_rot_matrices = transforms.rotation_6d_to_matrix(current_action[:, 3:9])
        prev_rot_matrices = transforms.rotation_6d_to_matrix(prev_target_pose[:, 3:9])
        current_quaternions = transforms.matrix_to_quaternion(current_rot_matrices)
        prev_quaternions = transforms.matrix_to_quaternion(prev_rot_matrices)
    elif input_rotation_representation == 'euler_angles':
        current_rot_matrices = transforms.euler_angles_to_matrix(current_action[:, 3:6], convention='XYZ')
        prev_rot_matrices = transforms.euler_angles_to_matrix(prev_target_pose[:, 3:6], convention='XYZ')
        current_quaternions = transforms.matrix_to_quaternion(current_rot_matrices)
        prev_quaternions = transforms.matrix_to_quaternion(prev_rot_matrices)

    delta_rotation = transforms.quaternion_multiply(current_quaternions, transforms.quaternion_invert(prev_quaternions))
    if output_rotation_representation == 'axis_angle':
        delta_action[:, 3:6] = transforms.quaternion_to_axis_angle(delta_rotation)
    elif output_rotation_representation == 'quaternion':
        delta_action[:, 3:7] = delta_rotation
    elif output_rotation_representation == '6d':
        delta_action[:, 3:9] = transforms.matrix_to_rotation_6d(transforms.quaternion_to_matrix(delta_rotation))
    elif output_rotation_representation == 'euler_angles':
        delta_action[:, 3:6] = transforms.matrix_to_euler_angles(transforms.quaternion_to_matrix(delta_rotation), convention='XYZ')

    if output_numpy:
        delta_action = delta_action.cpu().numpy()
    return delta_action

def compute_action_plan_error(predicted_action_plan:torch.Tensor, ground_truth_action_plan:torch.Tensor, rotation_representation:str='euler_angles', output_angular_error_in_degrees:bool=True):
    '''
    predicted_action_plan: BxHxN
    ground_truth_action_plan: BxHxN
    assumes that both are expressed in the same frame
    assumes quaternions are with real/scalar part first
    '''
    assert predicted_action_plan.shape == ground_truth_action_plan.shape, f"predicted_action_plan and ground_truth_action_plan must have the same shape, but got {predicted_action_plan.shape} and {ground_truth_action_plan.shape}"
    assert rotation_representation in ['euler_angles', 'quaternion', 'axis_angle'], f"rotation_representation must be one of ['euler_angles', 'quaternion', 'axis_angle'], but got {rotation_representation}"
    # remove the last dimension which is the gripper action
    predicted_action_plan = predicted_action_plan[..., :-1]
    ground_truth_action_plan = ground_truth_action_plan[..., :-1]
    if rotation_representation == 'euler_angles':
        assert predicted_action_plan.shape[-1] == 6, f"predicted_action_plan must have 6 dimensions for euler angles, but got {predicted_action_plan.shape[-1]}"
        predicted_action_world_rot_target = transforms.matrix_to_quaternion(transforms.euler_angles_to_matrix(predicted_action_plan[:, :, 3:6], convention='XYZ'))
        ground_truth_action_world_rot_target = transforms.matrix_to_quaternion(transforms.euler_angles_to_matrix(ground_truth_action_plan[:, :, 3:6], convention='XYZ'))
    elif rotation_representation == 'quaternion':
        assert predicted_action_plan.shape[-1] == 7, f"predicted_action_plan must have 7 dimensions for quaternions, but got {predicted_action_plan.shape[-1]}"
        predicted_action_world_rot_target = predicted_action_plan[:, :, 3:7]
        ground_truth_action_world_rot_target = ground_truth_action_plan[:, :, 3:7]
    elif rotation_representation == 'axis_angle':
        assert predicted_action_plan.shape[-1] == 6, f"predicted_action_plan must have 6 dimensions for axis angles, but got {predicted_action_plan.shape[-1]}"
        predicted_action_world_rot_target = transforms.axis_angle_to_quaternion(predicted_action_plan[:, :, 3:6])
        ground_truth_action_world_rot_target = transforms.axis_angle_to_quaternion(ground_truth_action_plan[:, :, 3:6])

    ground_truth_to_world_rotation = transforms.quaternion_invert(ground_truth_action_world_rot_target)
    ground_truth_to_predicted_rotation = transforms.quaternion_multiply(ground_truth_to_world_rotation, predicted_action_world_rot_target)
    ground_truth_to_predicted_rotation_axis_angle = transforms.quaternion_to_axis_angle(ground_truth_to_predicted_rotation)
    if output_angular_error_in_degrees:
        ground_truth_to_predicted_rotation_axis_angle = torch.rad2deg(ground_truth_to_predicted_rotation_axis_angle)
    ground_truth_to_predicted_rotation_axis_angle_in_world = transforms.quaternion_apply(ground_truth_to_world_rotation, ground_truth_to_predicted_rotation_axis_angle)
    # ground_truth_to_predicted_rotation_angle_errors = torch.linalg.norm(ground_truth_to_predicted_rotation_axis_angle, dim=-1, ord=2, keepdim=True)
    
    predicted_action_positions = predicted_action_plan[:, :, :3]
    ground_truth_action_positions = ground_truth_action_plan[:, :, :3]
    
    ground_truth_to_predicted_translation_errors_vector = transforms.quaternion_apply(ground_truth_to_world_rotation, predicted_action_positions) - transforms.quaternion_apply(ground_truth_to_world_rotation, ground_truth_action_positions)
    ground_truth_to_predicted_translation_errors_vector_in_world = transforms.quaternion_apply(ground_truth_action_world_rot_target, ground_truth_to_predicted_translation_errors_vector)
    # ground_truth_to_predicted_translation_errors = torch.linalg.norm(ground_truth_to_predicted_translation_errors_vector, dim=-1, ord=2, keepdim=True)

    # return ground_truth_to_predicted_translation_errors_vector_in_world, ground_truth_to_predicted_translation_errors, ground_truth_to_predicted_rotation_axis_angle_in_world, ground_truth_to_predicted_rotation_angle_errors
    return ground_truth_to_predicted_translation_errors_vector_in_world, ground_truth_to_predicted_rotation_axis_angle_in_world

def batched_position_to_pixel_coordinates(positions, camera_intrinsic, camera_extrinsic):
    '''
    positions: bxNx3
    returns: projected_points (u,v): bxNx2
    '''
    assert positions.ndim == 3, "positions must have shape bxNx3"
    assert positions.shape[-1] == 3, "positions must have shape bxNx3"
    b, N, _ = positions.shape
    positions = einops.rearrange(positions, 'b n c -> (b n) c')
    # bx4x4 @ b*Nx3 -> b*Nx3
    # contact_positions_in_cam = transform_points(contact_positions, self.base_camera_extrinsic_cv)
    positions_in_cam = torch.cat([positions, torch.ones((b*N, 1), device=positions.device)], dim=1)
    positions_in_cam = einops.rearrange(torch.bmm(camera_extrinsic, (positions_in_cam.T).unsqueeze(0)), 'b c n -> (b n) c')[..., :3]
    # project to image plane
    # bx3x3 @ b*Nx3 -> bxNx3
    projected_points = einops.rearrange(torch.bmm(camera_intrinsic, (positions_in_cam.T).unsqueeze(0)), 'b c n -> (b n) c')
    projected_points = projected_points[..., :2] / projected_points[..., 2:]
    # b*Nx2
    projected_points = einops.rearrange(projected_points, '(b n) c -> b n c', b=b, n=N)

    # filter out points outside of image plane
    projected_points = projected_points.int()

    return projected_points

def get_extrinsic_contact_data(num_envs, device, scene, max_extrinsic_contacts, object_name, robot_name, return_contact_positions=False, return_contact_forces=False):
    assert return_contact_positions or return_contact_forces, "Must return either contact positions or contact forces"
    assert num_envs == 1, "Only supports single envs for now"
    contact_data = dict()
    with torch.device(device):
        # TODO extend to multiple envs
        if return_contact_positions:
            contact_positions = torch.nan*torch.ones((1, max_extrinsic_contacts, 3), device=device)
        if return_contact_forces:
            contact_forces = torch.nan*torch.ones((1, max_extrinsic_contacts, 3), device=device)
        contacts = scene.get_contacts()
        filtered_contacts = list()
        # filter contacts to only include contacts between grasped_book
        if len(contacts) > 0:
            for contact in contacts:
                body_name_0 = contact.bodies[0].entity.name
                body_name_1 = contact.bodies[1].entity.name
                if object_name in body_name_0 or object_name in body_name_1:
                    # and not contact panda
                    if robot_name not in body_name_0 and robot_name not in body_name_1:
                        filtered_contacts.append(contact)
        contacts = filtered_contacts
        contact_idx = 0
        if len(contacts) > 0:
            for contact in contacts:
                for contact_point in contact.points:
                    if np.linalg.norm(contact_point.impulse) > 0:
                        if return_contact_forces:
                            contact_forces[0, contact_idx] = torch.from_numpy(contact_point.impulse)
                            # switch direction if grasped_book is the second body
                            # body_name_0 = contact.bodies[0].entity.name
                            body_name_1 = contact.bodies[1].entity.name
                            if object_name in body_name_1:
                                contact_forces[0, contact_idx] *= -1
                        if return_contact_positions:
                            contact_positions[0, contact_idx] = torch.from_numpy(contact_point.position)
                        contact_idx += 1
                        # torch.from_numpy(contact_point.position)
                # contact_positions.extend([torch.from_numpy(contact_point.position) for contact_point in contact.points])
        if return_contact_forces:
            contact_data['contact_forces'] = contact_forces
        if return_contact_positions:
            contact_data['contact_positions'] = contact_positions
        return contact_data

def pixel_coordinates_to_image_array_indices(pixel_coordinates, camera_height, camera_width, device, contact_forces=None):
    # TODO extend to multiple envs
    '''
    pixel_coordinates in format (u,v): bxNx2
    contact forces in format bxNx3
    returns: image_array_indices: bxHxWx1
    '''
    # filter out nan rows
    assert pixel_coordinates.ndim == 3, "pixel_coordinates must have shape bxNx2"
    assert pixel_coordinates.shape[-1] == 2, "pixel_coordinates must have shape bxNx2"
    if contact_forces is not None:
        assert contact_forces.ndim == 3, "contact_forces must have shape bxNx3"
        assert contact_forces.shape[1] == pixel_coordinates.shape[1], "contact_forces and pixel_coordinates must have same N dimension"
        assert contact_forces.shape[0] == pixel_coordinates.shape[0], "contact_forces and pixel_coordinates must have same batch dimension"
        assert contact_forces.shape[2] == 3, "contact_forces must have shape bxNx3"
    with torch.device(device):
        # swap u and v to match image coordinates
        pixel_coordinates = pixel_coordinates[..., [1, 0]]
        
        # filter out points outside of image plane
        valid_points = (pixel_coordinates[..., 0] >= 0) & (pixel_coordinates[..., 0] < camera_height) & (pixel_coordinates[..., 1] >= 0) & (pixel_coordinates[..., 1] < camera_width)
        pixel_coordinates = pixel_coordinates[valid_points]
        if contact_forces is not None:
            contact_forces = contact_forces[valid_points]
        # add index for batch dimension
        pixel_coordinates = torch.cat([torch.zeros((pixel_coordinates.shape[0], 1), dtype=torch.int), pixel_coordinates], dim=1)
    return pixel_coordinates, contact_forces

def get_extrinsic_contact_map_data(num_envs, device, scene, max_extrinsic_contacts, camera_height, camera_width, camera_intrinsic, camera_extrinsic, object_name, robot_name, return_contact_positions_map=True, return_contact_forces_map=True):
    assert return_contact_positions_map or return_contact_forces_map, "must return at least one of contact positions map or forces map"
    contact_map_dict = dict()
    with torch.device(device):
        contact_data_dict = get_extrinsic_contact_data(num_envs, device, scene, max_extrinsic_contacts, object_name, robot_name, return_contact_positions=True, return_contact_forces=return_contact_forces_map)
        assert contact_data_dict['contact_positions'].shape[-1] == 3, "contact_positions must have shape bxNx3"
        if return_contact_positions_map:
            contact_map_dict['extrinsic_contact_positions'] = contact_data_dict['contact_positions']
        if return_contact_forces_map:
            assert contact_data_dict['contact_forces'].shape[-1] == 3, "contact_forces must have shape bxNx3"
            contact_map_dict['extrinsic_contact_forces'] = contact_data_dict['contact_forces']
        b, N, _ = contact_data_dict['contact_positions'].shape
        contact_positions = contact_data_dict['contact_positions'][~torch.any(torch.isnan(contact_data_dict['contact_positions']), dim=2)].reshape(b, -1, 3)
        b, N, _ = contact_positions.shape
        if return_contact_positions_map:
            contact_map = torch.zeros((b, camera_height, camera_width, 1), dtype=torch.float32)
        if return_contact_forces_map:
            contact_forces_map = torch.zeros((b, camera_height, camera_width, 3), dtype=torch.float32)
        if N > 0:
            contact_pixel_coordinates = batched_position_to_pixel_coordinates(contact_positions, camera_intrinsic, camera_extrinsic)
            contact_forces = None
            if return_contact_forces_map:
                contact_forces = contact_data_dict['contact_forces'][~torch.any(torch.isnan(contact_data_dict['contact_forces']), dim=2)].reshape(b, -1, 3)
            contact_image_array_indices, contact_forces = pixel_coordinates_to_image_array_indices(contact_pixel_coordinates, camera_height, camera_width, device, contact_forces=contact_forces)

            if return_contact_positions_map:
                contact_map[tuple(contact_image_array_indices.T)] = 1.0
            if return_contact_forces_map:
                contact_forces_map[tuple(contact_image_array_indices.T)] = contact_forces
        if return_contact_positions_map:
            contact_map_dict['extrinsic_contact_map'] = contact_map
        if return_contact_forces_map:
            contact_map_dict['extrinsic_contact_forces_map'] = contact_forces_map
    return contact_map_dict

def get_extra_contact_features(env_mesh_list, env_mesh, EE_object_mesh_list, EE_object_mesh, tm_camera, render_dtc_maps, render_normals_maps):
    # TODO handle parallel envs
    extra_contact_features_dict = dict()

    ray_origins, ray_directions, pixels_uv = generate_rays_from_camera(tm_camera)

    env_hit_min_locations, env_hit_min_pixels_uv, env_hit_min_distances, env_hit_min_index_tri, env_hit_min_ray_directions = get_min_grasped_obj_sdf_at_env_hits_data(ray_origins, ray_directions, pixels_uv, env_mesh, EE_object_mesh_list)
    if render_dtc_maps:
        EE_obj_sdf_on_env_image, EE_obj_sdf_on_env_mask = generate_min_distances_image(env_hit_min_pixels_uv, env_hit_min_distances, tm_camera.resolution[::-1])
        EE_obj_sdf_on_env_image = EE_obj_sdf_on_env_image.astype(np.float32)[:240, :320, np.newaxis]
        # EE_obj_sdf_on_env_mask = EE_obj_sdf_on_env_mask.astype(bool)[:240, :320]
        # assert EE_obj_sdf_on_env_image.shape == image_shape + (1,)

        EE_obj_sdf_on_env_image = to_tensor(EE_obj_sdf_on_env_image).unsqueeze(0) # hack to add env/batch dimension
        extra_contact_features_dict['env_dtc_map'] = EE_obj_sdf_on_env_image

    if render_normals_maps:
        min_env_surface_normals = env_mesh.face_normals[env_hit_min_index_tri]
        env_xyz_normals_image, env_xyz_normals_image_mask = normals_to_xyz_map(min_env_surface_normals, tm_camera.resolution[::-1], env_hit_min_pixels_uv)#, fill_value=1.0/np.sqrt(3.0))
        env_xyz_normals_image = env_xyz_normals_image.astype(np.float32)[:240, :320]
        # env_xyz_normals_image_mask = env_xyz_normals_image_mask.astype(bool)[:240, :320]

        env_xyz_normals_image = to_tensor(env_xyz_normals_image).unsqueeze(0) # hack to add env/batch dimension
        extra_contact_features_dict['env_normals_map'] = env_xyz_normals_image

    EE_obj_hit_min_locations, EE_obj_hit_min_pixels_uv, EE_obj_hit_min_distances, EE_obj_hit_min_index_tri, EE_obj_hit_min_ray_directions = get_min_env_sdf_at_grasped_obj_hits_data(ray_origins, ray_directions, pixels_uv, env_mesh_list, EE_object_mesh)
    if render_dtc_maps:
        env_sdf_on_EE_obj_image, env_sdf_on_EE_obj_mask = generate_min_distances_image(EE_obj_hit_min_pixels_uv, EE_obj_hit_min_distances, tm_camera.resolution[::-1])
        env_sdf_on_EE_obj_image = env_sdf_on_EE_obj_image.astype(np.float32)[:240, :320, np.newaxis]
        # env_sdf_on_EE_obj_mask = env_sdf_on_EE_obj_mask.astype(bool)[:240, :320]

        env_sdf_on_EE_obj_image = to_tensor(env_sdf_on_EE_obj_image).unsqueeze(0) # hack to add env/batch dimension
        extra_contact_features_dict['EE_dtc_map'] = env_sdf_on_EE_obj_image
    
    if render_normals_maps:       
        min_EE_object_surface_normals = EE_object_mesh.face_normals[EE_obj_hit_min_index_tri] # these are normalized already
        EE_object_xyz_normals_image, EE_object_xyz_normals_image_mask = normals_to_xyz_map(min_EE_object_surface_normals, tm_camera.resolution[::-1], EE_obj_hit_min_pixels_uv)#, fill_value=1.0/np.sqrt(3.0))
        EE_object_xyz_normals_image = EE_object_xyz_normals_image.astype(np.float32)[:240, :320]
        # EE_object_xyz_normals_image_mask = EE_object_xyz_normals_image_mask.astype(bool)[:240, :320]
        
        EE_object_xyz_normals_image = to_tensor(EE_object_xyz_normals_image).unsqueeze(0) # hack to add env/batch dimension
        extra_contact_features_dict['EE_normals_map'] = EE_object_xyz_normals_image
    
    return extra_contact_features_dict

def convert_sapien_pose_to_transform_matrix(sapien_pose):
    position, quaternion = sapien_pose.p, sapien_pose.q
    if len(position.shape) == 2:
        position = position[0]
    if len(quaternion.shape) == 2:
        quaternion = quaternion[0]
    if isinstance(position, torch.Tensor):
        position = position.cpu().numpy()
    if isinstance(quaternion, torch.Tensor):
        quaternion = quaternion.cpu().numpy()
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = R.from_quat(quaternion, scalar_first=True).as_matrix()
    transform_matrix[:3, 3] = position
    return transform_matrix

def get_cuboid_dict(device, pose, half_dims):
    # returns the batched center position, three half-lenghts, and the rotation matrix of the cuboid
    assert pose.ndim == 2 and pose.shape[1] == 7, f"Pose must have shape (batch_size, 7) with quaternion representation but got {pose.shape}"
    assert half_dims.ndim == 2 and half_dims.shape[1] == 3, f"Half dimensions must have shape (batch_size, 3) but got {half_dims.shape}"
    assert pose.shape[0] == half_dims.shape[0], "Pose and half_dims must have the same batch size"
    with torch.device(device):
        # get the cuboid pose
        cuboid_pose = pose # bx7
        # compute the cuboid
        center_pos = cuboid_pose[:, :3] # bx3
        rotation_matrix = transforms.quaternion_to_matrix(cuboid_pose[:, 3:]) # bx3x3
    return {'center_pos': center_pos, 'half_dims': half_dims, 'rotation_matrix': rotation_matrix}


def cuboid_intersection_test(device, cuboid_A_dict, cuboid_B_dict):
    # write our own function of whether the grasped book is still in the gripper by checking if two cuboids intersect
    # cuboid one is defined as the volume in between the gripper fingers, and cuboid two is the grasped book
    # return a torch boolean tensor of shape (num_envs,)
    # following Stefan Gottschalk's implementation of OBB SAT https://gamma.cs.unc.edu/users/gottschalk/main.pdf
    with torch.device(device):
        r_circumscribed_sphere_A = torch.linalg.norm(cuboid_A_dict['half_dims'], axis=1, ord=2, keepdim=True)
        r_circumscribed_sphere_B = torch.linalg.norm(cuboid_B_dict['half_dims'], axis=1, ord=2, keepdim=True)
        W_t_AB = cuboid_B_dict['center_pos'] - cuboid_A_dict['center_pos'] # bx3
        center_distance = torch.linalg.norm(W_t_AB, axis=1, ord=2, keepdim=True)
        # if distance is greater than the sum of the two spheres, then they are not intersecting
        cuboids_are_not_intersecting = center_distance > (r_circumscribed_sphere_A + r_circumscribed_sphere_B)
        if torch.all(cuboids_are_not_intersecting): # we can break early if ALL cuboids are not intersecting
            return cuboids_are_not_intersecting.logical_not().squeeze(-1) # return a boolean tensor of shape (num_envs,)
        
        # otherwise, need to continue with a more precise check using Separating Axis Theorem (SAT)
        A_R_B = torch.bmm(cuboid_A_dict['rotation_matrix'].transpose(1, 2), cuboid_B_dict['rotation_matrix']) # bx3x3
        abs_A_R_B = torch.abs(A_R_B) # bx3x3

        A_t_AB = torch.bmm(cuboid_A_dict['rotation_matrix'].transpose(1, 2), W_t_AB.unsqueeze(-1)).squeeze(-1) # bx3
        abs_A_t_AB = torch.abs(A_t_AB) # bx3

        # check the axes of A
        for i in range(3):
            ra = cuboid_A_dict['half_dims'][:, i:i+1] # bx1
            rb = (cuboid_B_dict['half_dims']*abs_A_R_B[:, i]).sum(dim=1, keepdim=True) # bx1
            cuboids_are_not_intersecting = torch.logical_or(cuboids_are_not_intersecting, abs_A_t_AB[:, i:i+1] > (ra + rb))
            if torch.all(cuboids_are_not_intersecting):
                return cuboids_are_not_intersecting.logical_not().squeeze(-1)

        # check the axes of B
        abs_B_R_A = abs_A_R_B.transpose(1, 2) # bx3x3

        B_t_AB = torch.bmm(cuboid_B_dict['rotation_matrix'].transpose(1, 2), W_t_AB.unsqueeze(-1)).squeeze(-1) # bx3
        abs_B_t_AB = torch.abs(B_t_AB) # bx3

        for i in range(3):
            ra = cuboid_B_dict['half_dims'][:, i:i+1] # bx1
            rb = (cuboid_A_dict['half_dims']*abs_B_R_A[:, i]).sum(dim=1, keepdim=True) # bx1
            cuboids_are_not_intersecting = torch.logical_or(cuboids_are_not_intersecting, abs_B_t_AB[:, i:i+1] > (ra + rb))
            if torch.all(cuboids_are_not_intersecting):
                return cuboids_are_not_intersecting.logical_not().squeeze(-1)

        # check the axes of A x B
        for i in range(3):
            for j in range(3):
                # compute the axis as the cross product of the two rotation matrices
                axis = torch.cross(cuboid_A_dict['rotation_matrix'][:, :, i], cuboid_B_dict['rotation_matrix'][:, :, j], dim=1) # bx3
                # if the axis is zero, skip it
                axis_norm = torch.linalg.norm(axis, axis=1, ord=2, keepdim=True)
                if (axis_norm < 1e-6).all():
                    continue
                # simplified SAT check for the axis 
                ra = (cuboid_A_dict['half_dims'][:, (i+1)%3] * abs_A_R_B[:, (i+2)%3, j] + 
                        cuboid_A_dict['half_dims'][:, (i+2)%3] * abs_A_R_B[:, (i+1)%3, j])
                
                rb = (cuboid_B_dict['half_dims'][:, (j+1)%3] * abs_A_R_B[:, i, (j+2)%3] + 
                        cuboid_B_dict['half_dims'][:, (j+2)%3] * abs_A_R_B[:, i, (j+1)%3])
                
                abs_axis_t_AB = torch.abs(A_t_AB[:, (i+2)%3] * A_R_B[:, (i+1)%3, j] -
                                            A_t_AB[:, (i+1)%3] * A_R_B[:, (i+2)%3, j])
                cuboids_are_not_intersecting = torch.logical_or(cuboids_are_not_intersecting, abs_axis_t_AB > (ra + rb))
                if torch.all(cuboids_are_not_intersecting):
                    return cuboids_are_not_intersecting.logical_not().squeeze(-1)

        # if we reach here, then the cuboids are intersecting
        return torch.logical_or(cuboids_are_not_intersecting, torch.zeros_like(cuboids_are_not_intersecting, dtype=torch.bool)).logical_not().squeeze(-1) # return a boolean tensor of shape (num_envs,) where True means the cuboids are intersecting