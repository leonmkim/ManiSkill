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
    assert input_delta_rotation_representation in ['axis_angle', 'quaternion'], f"input_delta_rotation_representation {input_delta_rotation_representation} not supported"
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
    return output_poses

def compute_delta_action(prev_target_pose, current_action, input_rotation_representation, output_rotation_representation='axis_angle'):
    '''
    Args:
        prev_target_pose: Bx(3+R) where R is 3 for axis angle or 4 for quaternion or 6 for 6D representation
        current_target_pose: Bx(3+R) where R is 3 for axis angle or 4 for quaternion or 6 for 6D representation
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
    
    if input_rotation_representation == 'axis_angle':
        assert prev_target_pose.shape[1] == 6, f"prev_target_pose shape {prev_target_pose.shape} not supported for axis_angle representation"
        assert current_action.shape[1] == 7, f"current_target_pose shape {current_action.shape} not supported for axis_angle representation"
    elif input_rotation_representation == 'quaternion':
        assert prev_target_pose.shape[1] == 7, f"prev_target_pose shape {prev_target_pose.shape} not supported for quaternion representation"
        assert current_action.shape[1] == 8, f"current_target_pose shape {current_action.shape} not supported for quaternion representation"
    elif input_rotation_representation == '6d':
        assert prev_target_pose.shape[1] == 9, f"prev_target_pose shape {prev_target_pose.shape} not supported for 6d representation"
        assert current_action.shape[1] == 10, f"current_target_pose shape {current_action.shape} not supported for 6d representation"
    elif input_rotation_representation == 'euler_angles':
        assert prev_target_pose.shape[1] == 6, f"prev_target_pose shape {prev_target_pose.shape} not supported for euler_angles representation"
        assert current_action.shape[1] == 7, f"current_target_pose shape {current_action.shape} not supported for euler_angles representation"

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
    return delta_action