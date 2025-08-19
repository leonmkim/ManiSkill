#%%
from typing import Any, Dict, Union

from dataclasses import dataclass, field
import numpy as np
import sapien
import torch

from mani_skill.agents.robots.panda import Panda
from mani_skill.utils.building.actors.common import build_coordinate_frame
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder, SimpleTableSceneBuilder, get_table_primitive_mesh_list
from mani_skill.utils.structs import Actor, Pose
from mani_skill.utils.structs.types import SimConfig
from mani_skill.utils.common import batched_position_to_pixel_coordinates, get_extrinsic_contact_map_data, get_extra_contact_features, convert_sapien_pose_to_transform_matrix, get_cuboid_dict, cuboid_intersection_test

from pytorch3d import transforms

import trimesh as tm

from pathlib import Path
import sys, os
path_to_this_file = Path(os.path.abspath(__file__))
path_to_contact_estimation = path_to_this_file.parents[5] / "contact_estimation"
sys.path.append(str(path_to_contact_estimation))
from src.dataset.gazebo_to_trimesh import create_trimesh_camera
#%%
from colormath2.color_objects import sRGBColor, LabColor
from colormath2.color_conversions import convert_color
from colormath2.color_diff import delta_e_cie2000


#%%
def get_peg_primitive_mesh_list(peg_length, peg_width, peg_height, global_transform=None):
    full_sizes = [
        [peg_length, peg_width, peg_height], # peg
    ]
    poses = [
        sapien.Pose([0, 0, 0]).to_transformation_matrix(), # peg
    ]
    peg_geometries = []
    for i, (full_size, pose) in enumerate(zip(full_sizes, poses)):
        # builder.add_box_collision(pose, half_size, density=density)
        object_geometry = tm.primitives.Box(extents=full_size)
        object_geometry.apply_transform(pose)
        if global_transform is not None:
            object_geometry.apply_transform(global_transform)
        peg_geometries.append(object_geometry)

    return peg_geometries

def get_box_meshes_list(inner_radius, outer_radius, depth, center, global_transform):
    env_object_meshes_list = []
    thickness = (outer_radius - inner_radius) * 0.5
    half_center = [x * 0.5 for x in center]
    full_sizes = [
        [depth*2, (thickness - half_center[0])*2, outer_radius*2],
        [depth*2, (thickness + half_center[0])*2, outer_radius*2],
        [depth*2, outer_radius*2, (thickness - half_center[1])*2],
        [depth*2, outer_radius*2, (thickness + half_center[1])*2],
    ]
    offset = thickness + inner_radius
    poses = [
        sapien.Pose([0, offset + half_center[0], 0]).to_transformation_matrix(),
        sapien.Pose([0, -offset + half_center[0], 0]).to_transformation_matrix(),
        sapien.Pose([0, 0, offset + half_center[1]]).to_transformation_matrix(),
        sapien.Pose([0, 0, -offset + half_center[1]]).to_transformation_matrix(),
    ]
    for i, (full_size, pose) in enumerate(zip(full_sizes, poses)):
        # builder.add_box_collision(pose, half_size, density=density)
        object_geometry = tm.primitives.Box(extents=full_size)
        object_geometry.apply_transform(pose)
        if global_transform is not None:
            object_geometry.apply_transform(global_transform)
        env_object_meshes_list.append(object_geometry)

    return env_object_meshes_list

def _build_box_with_hole(
    scene: ManiSkillScene, inner_radius, outer_radius, depth, center=(0, 0), color=None
):
    builder = scene.create_actor_builder()
    thickness = (outer_radius - inner_radius) * 0.5
    # x-axis is hole direction
    half_center = [x * 0.5 for x in center]
    half_sizes = [
        [depth, thickness - half_center[0], outer_radius],
        [depth, thickness + half_center[0], outer_radius],
        [depth, outer_radius, thickness - half_center[1]],
        [depth, outer_radius, thickness + half_center[1]],
    ]
    offset = thickness + inner_radius
    poses = [
        sapien.Pose([0, offset + half_center[0], 0]),
        sapien.Pose([0, -offset + half_center[0], 0]),
        sapien.Pose([0, 0, offset + half_center[1]]),
        sapien.Pose([0, 0, -offset + half_center[1]]),
    ]

    mat = sapien.render.RenderMaterial(
        base_color=color if color is not None else sapien_utils.hex2rgba("#FFD289"), roughness=0.5, specular=0.5
    )

    for half_size, pose in zip(half_sizes, poses):
        builder.add_box_collision(pose, half_size)
        builder.add_box_visual(pose, half_size, material=mat)
    return builder

def generate_batched_colors_avoiding_target_color(target_color, batch_size, rng, lab_threshold=15.0):
    """
    Generate a batch of colors that are different from the target color.

    target_color is numpy array of RGB

    generates Bx4 RGBA array
    """
    batched_colors = []
    target_color_in_lab = convert_color(sRGBColor(*target_color[:3]), LabColor)
    for _ in range(batch_size):
        color = np.ones(4)
        lab_distance = 0.0
        # just rejection sample
        while lab_distance < lab_threshold:
            color[:3] = rng.uniform(0.0, 1.0, size=3)
            rgb_color = sRGBColor(color[0], color[1], color[2])
            # compute LAB distance to target color
            lab_distance = delta_e_cie2000(convert_color(rgb_color, LabColor), target_color_in_lab)
        batched_colors.append(color)
    return np.array(batched_colors)

@dataclass
class RobotConfig:
    """
    Configuration for the robot in the BookInsertionEnv.
    """
    # init_qpos: list = field(default_factory=lambda: [0.022516679397616424, 0.11646689505116431, -0.3625673227601117, -1.37265637618617, 0.033468631741809286, 1.4658307538809252, 0.46052758571920294,.04,.04,])
    init_qpos: list = field(default_factory=lambda: [-0.45725486, 0.18291518, 0.16500726, -2.2905693, -0.0728711, 2.4728112, -1.0869355, 0.02300941, 0.02296073])
    
    gripper_friction: float = 4.0 # default is 2.0
    gripper_patch_radius: float = 0.1 # default is 0.1
    # additive_y_randomization_bounds: Union[float, list] = 0.0

@dataclass
class BoxConfig:
    """
    Configuration for the box in the PegInsertionSideCustomEnv.
    """
    randomize_color: bool = True
    
    randomize_tolerance: bool = True
    nominal_tolerance: float = 0.003
    tolerance_randomization_bounds: list = field(default_factory=lambda: [0.003, 0.015]) # default tolerance is .003m
    
    nominal_x_position: float = 0.45
    randomize_x_position: bool = True
    x_position_delta_randomization_bounds: list = field(default_factory=lambda: [-0.05, 0.05])
    
    nominal_y_position: float = 0.25
    randomize_y_position: bool = True
    y_position_delta_randomization_bounds: list = field(default_factory=lambda: [-0.05, 0.05])
    
    nominal_yaw: float = np.pi*(10/16)
    randomize_yaw: bool = False
    yaw_delta_randomization_bounds: list = field(default_factory=lambda: [-np.pi / 8, np.pi / 8])
    # yaw_delta_randomization_bounds: list = field(default_factory=lambda: [-np.pi/16, np.pi/16]) 

    randomize_hole_center_location: bool = False
    hole_center_randomization_bounds: list = field(default_factory=lambda: [-1.0, 1.0]) # ranges from -1 to 1 where 0 is the center of the box

    def __post_init__(self):
        any_randomize = any([
            self.randomize_color,
            self.randomize_tolerance,
        ])
        if self.randomize_tolerance:
            # assert bounds for tolerance are valid > 0
            assert (
                isinstance(self.tolerance_randomization_bounds, list)
                and len(self.tolerance_randomization_bounds) == 2
                and self.tolerance_randomization_bounds[0] > 0
                and self.tolerance_randomization_bounds[1] > 0
            ), f"tolerance_randomization_bounds must be a list of two positive values, but got {self.tolerance_randomization_bounds}"
        if self.randomize_x_position:
            # assert bounds for x position are valid
            assert (
                isinstance(self.x_position_delta_randomization_bounds, list)
                and len(self.x_position_delta_randomization_bounds) == 2
                and self.x_position_delta_randomization_bounds[0] < self.x_position_delta_randomization_bounds[1]
            ), f"x_position_randomization_bounds must be a list of two values with the first one smaller than the second, but got {self.x_position_delta_randomization_bounds}"
        if self.randomize_y_position:
            # assert bounds for y position are valid
            assert (
                isinstance(self.y_position_delta_randomization_bounds, list)
                and len(self.y_position_delta_randomization_bounds) == 2
                and self.y_position_delta_randomization_bounds[0] < self.y_position_delta_randomization_bounds[1]
            ), f"y_position_randomization_bounds must be a list of two values with the first one smaller than the second, but got {self.y_position_delta_randomization_bounds}"
        if self.randomize_yaw:
            # assert bounds for yaw are valid
            assert (
                isinstance(self.yaw_delta_randomization_bounds, list)
                and len(self.yaw_delta_randomization_bounds) == 2
                and self.yaw_delta_randomization_bounds[0] < self.yaw_delta_randomization_bounds[1]
            ), f"yaw_randomization_bounds must be a list of two values with the first one smaller than the second, but got {self.yaw_delta_randomization_bounds}"

@dataclass
class PegConfig:
    """
    Configuration for the peg in the PegInsertionSideCustomEnv.
    """
    randomize_color: bool = False

    randomize_length: bool = False
    nominal_length: float = 0.105  # default peg length is .105m
    length_randomization_bounds: list = field(default_factory=lambda: [0.085, 0.125])  # default peg length is .105m

    nominal_radius: float = 0.02  # default peg radius is .02m
    randomize_radius: bool = True
    radius_randomization_bounds: list = field(default_factory=lambda: [0.015, 0.03])  # default peg radius is .02m

    def __post_init__(self):
        if self.randomize_length:
            assert (
                isinstance(self.length_randomization_bounds, list)
                and len(self.length_randomization_bounds) == 2
                and self.length_randomization_bounds[0] < self.length_randomization_bounds[1]
            ), f"length_randomization_bounds must be a list of two values with the first one smaller than the second, but got {self.length_randomization_bounds}"
        if self.randomize_radius:
            assert (
                isinstance(self.radius_randomization_bounds, list)
                and len(self.radius_randomization_bounds) == 2
                and self.radius_randomization_bounds[0] < self.radius_randomization_bounds[1]
            ), f"radius_randomization_bounds must be a list of two values with the first one smaller than the second, but got {self.radius_randomization_bounds}"


@register_env("PegInsertionSideCustom-v1", max_episode_steps=100)
class PegInsertionSideCustomEnv(BaseEnv):
    """
    **Task Description:**
    Pick up a orange-white peg and insert the orange end into the box with a hole in it.

    **Randomizations:**
    - Peg half length is randomized between 0.085 and 0.125 meters. Box half length is the same value. (during reconfiguration)
    - Peg radius/half-width is randomized between 0.015 and 0.025 meters. Box hole's radius is same value + 0.003m of clearance. (during reconfiguration)
    - Peg is laid flat on table and has it's xy position and z-axis rotation randomized
    - Box is laid flat on table and has it's xy position and z-axis rotation randomized

    **Success Conditions:**
    - The white end of the peg is within 0.015m of the center of the box (inserted mid way).
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PegInsertionSide-v1_rt.mp4"
    SUPPORTED_ROBOTS = ["panda"]
    agent: Union[Panda]
    # _clearance = 0.003

    cam_resize_factor: float = 0.5

    robot_config: RobotConfig = RobotConfig()

    box_config: BoxConfig = BoxConfig()

    peg_config: PegConfig = PegConfig()

    render_contact_map: bool = False
    render_dtc_maps: bool = False
    render_normals_maps: bool = False
    render_contact_forces_map: bool = False

    max_extrinsic_contacts: int = 50 # for padding

    def __init__(
        self,
        *args,
        robot_uids="panda",
        num_envs=1,
        reconfiguration_freq=None,
        **kwargs,
    ):
        if reconfiguration_freq is None:
            if num_envs == 1:
                reconfiguration_freq = 1
            else:
                reconfiguration_freq = 0
        for key in kwargs:
            # if key in self.__dict__:
            if key in PegInsertionSideCustomEnv.__dict__:
                setattr(self, key, kwargs[key])
                # del kwargs[key]

        urdf_config = dict(
            _materials=dict(
                gripper=dict(static_friction=self.robot_config.gripper_friction, dynamic_friction=self.robot_config.gripper_friction, restitution=0.0)
            ),
            link=dict(
                panda_leftfinger=dict(
                    material="gripper", patch_radius=self.robot_config.gripper_patch_radius, min_patch_radius=self.robot_config.gripper_patch_radius
                ),
                panda_rightfinger=dict(
                    # material="gripper", patch_radius=0.1, min_patch_radius=0.1
                    material="gripper", patch_radius=self.robot_config.gripper_patch_radius, min_patch_radius=self.robot_config.gripper_patch_radius
                ),
            ),
        )
        super().__init__(
            *args,
            robot_uids=robot_uids,
            num_envs=num_envs,
            reconfiguration_freq=reconfiguration_freq,
            urdf_config=urdf_config,
            **kwargs,
        )

    @property
    def _default_sim_config(self):
        return SimConfig()

    @property
    def _default_sensor_configs(self):
        # pose = sapien_utils.look_at([0, -0.3, 0.2], [0, 0, 0.1])
        # return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

        from mani_skill.utils.geometry.rotation_conversions import matrix_to_quaternion
        # pose = sapien_utils.look_at([0, -0.3, 0.2], [0, 0, 0.1])
        self.camera_width = 640
        self.camera_height = 480
        self.intrinsics = torch.tensor([[596.61175537,0.,323.86328125],
                                [0.,596.96472168,246.78981018],
                                [0.,0.,1.]])
        if self.cam_resize_factor != 1.0:
            self.intrinsics[:2, :3] *= self.cam_resize_factor
            self.camera_width = int(self.camera_width * self.cam_resize_factor)
            self.camera_height = int(self.camera_height * self.cam_resize_factor)
        
        world_tf_root = self.agent.robot.get_pose()

        # training contact estimator
        # cam_tf_root = torch.tensor(
        # [[1.22464680e-16, 1.00000000e+00, 0.00000000e+00, -2.02066722e-16],
        # [2.03567160e-01, -2.49297871e-17, -9.79060985e-01, -1.58629880e-03],
        # [-9.79060985e-01, 1.19900390e-16, -2.03567160e-01, 1.67259212e+00],
        # [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
        # )

        # real
        # cam_tf_root = torch.tensor([[0.04930081, 0.99874239, -0.00911423, -0.04363872],
        #                             [0.12278183, -0.01511647, -0.99231855, 0.19901163],
        #                             [-0.99120838, 0.04780304, -0.12337268, 1.74423175],
        #                             [0., 0.,0.,1.]])
        # cam_tf_root = Pose.create_from_pq(p=cam_tf_root[:3, 3], q=matrix_to_quaternion(cam_tf_root[:3, :3]))
        # root_tf_cam = cam_tf_root.inv()
        
        # print(f"world_tf_root: {world_tf_root}")
        # world_tf_cam = world_tf_root * root_tf_cam
        # correct_orientation = axis_angle_to_quaternion(torch.tensor([np.pi/2, 0, 0]))
        # correct_orientation = quaternion_multiply(correct_orientation, axis_angle_to_quaternion(torch.tensor([0, 0, np.pi/2])))
        # world_tf_cam.q = quaternion_multiply(world_tf_cam.q, correct_orientation)

        look_at = world_tf_root.raw_pose[0,:3] + torch.tensor([0.,0,0.25])
        eye = torch.tensor([1.05775+.615, 0, 0.375615])
        self.world_tf_cam = sapien_utils.look_at(eye, look_at)

        return [CameraConfig("base_camera", self.world_tf_cam, width=self.camera_width, height=self.camera_height, intrinsic=self.intrinsics, near=0.01, far=5.0)]

    @property
    def _default_human_render_camera_configs(self):
        # pose = sapien_utils.look_at([0.5, -0.5, 0.8], [0.05, -0.1, 0.4])
        # return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)
        # pose = sapien_utils.look_at([0.5, -0.5, 0.8], [0.05, -0.1, 0.4])
        world_tf_root = self.agent.robot.get_pose()
        look_at = world_tf_root.raw_pose[0,:3] + torch.tensor([0.,0,0.25])
        eye = torch.tensor([1.05775+.1, 0, 0.375615])
        pose = sapien_utils.look_at(eye, look_at)

        # return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 5.0)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        with torch.device(self.device):
            self.table_scene = SimpleTableSceneBuilder(self)
            self.table_scene.build()

            if self.peg_config.randomize_length:
                peg_half_lengths = common.to_numpy(self._batched_episode_rng.uniform(self.peg_config.length_randomization_bounds[0], self.peg_config.length_randomization_bounds[1]))
            else:
                peg_half_lengths = np.ones(self.num_envs) * self.peg_config.nominal_length
            assert peg_half_lengths.ndim == 1, "Lengths should be 1D"
            assert peg_half_lengths.shape[0] == self.num_envs, f"Lengths shape {peg_half_lengths.shape} does not match num_envs {self.num_envs}"

            if self.peg_config.randomize_radius:
                peg_radii = common.to_numpy(self._batched_episode_rng.uniform(self.peg_config.radius_randomization_bounds[0], self.peg_config.radius_randomization_bounds[1]))
            else:
                peg_radii = np.ones(self.num_envs) * self.peg_config.nominal_radius
            assert peg_radii.ndim == 1, f"Radii should be 1D but got {peg_radii.ndim} with shape {peg_radii.shape}"
            assert peg_radii.shape[0] == self.num_envs, f"Radii shape {peg_radii.shape} does not match num_envs {self.num_envs}"

            if self.box_config.randomize_hole_center_location:
                box_centers = (
                    0.5
                    * (peg_half_lengths - peg_radii)[:, None]
                    * common.to_numpy(self._batched_episode_rng.uniform(self.box_config.hole_center_randomization_bounds[0], self.box_config.hole_center_randomization_bounds[1], size=(2,)))
                )
            else:
                box_centers = np.zeros((self.num_envs, 2))

            self.box_centers = common.to_tensor(box_centers)
            assert self.box_centers.shape == (self.num_envs, 2), f"Box centers shape {self.box_centers.shape} does not match num_envs {self.num_envs}"

            # save some useful values for use later
            self.peg_half_sizes = common.to_tensor(np.vstack([peg_half_lengths, peg_radii, peg_radii])).T
            peg_head_offsets = torch.zeros((self.num_envs, 3))
            peg_head_offsets[:, 0] = self.peg_half_sizes[:, 0]
            self.peg_head_offsets = Pose.create_from_pq(p=peg_head_offsets)

            box_hole_offsets = torch.zeros((self.num_envs, 3))
            box_hole_offsets[:, 1:] = common.to_tensor(box_centers)
            self.box_hole_offsets = Pose.create_from_pq(p=box_hole_offsets)
            if self.box_config.randomize_tolerance:
                clearances = common.to_numpy(self._batched_episode_rng.uniform(self.box_config.tolerance_randomization_bounds[0], self.box_config.tolerance_randomization_bounds[1]))
            else:
                clearances = np.ones(self.num_envs) * self.box_config.nominal_tolerance
            self.box_hole_clearances = common.to_tensor(clearances)
            self.box_hole_inner_radii = common.to_tensor(peg_radii + clearances)

            self.box_sizes = common.to_tensor(np.vstack([peg_radii+clearances, peg_half_lengths, peg_half_lengths]).T)
            assert self.box_sizes.shape == (self.num_envs, 3)

            # in each parallel env we build a different box with a hole and peg (the task is meant to be quite difficult)
            pegs = []
            boxes = []

            if self.box_config.randomize_color:
                box_colors = generate_batched_colors_avoiding_target_color(
                    target_color=sapien_utils.hex2rgba("#EC7357"), batch_size=self.num_envs, rng=self._batched_episode_rng, lab_threshold=15.0
                )
            else:
                box_colors = np.ones((self.num_envs, 4))
                box_color = sapien_utils.hex2rgba("#FFD289")
                box_colors[:] = box_color
            self.box_colors = common.to_tensor(box_colors)
            assert self.box_colors.shape == (self.num_envs, 4)

            for i in range(self.num_envs):
                scene_idxs = [i]
                length = peg_half_lengths[i]
                radius = peg_radii[i]
                clearance = clearances[i]
                builder = self.scene.create_actor_builder()
                builder.add_box_collision(half_size=[length, radius, radius])
                # peg head
                mat = sapien.render.RenderMaterial(
                    base_color=sapien_utils.hex2rgba("#EC7357"),
                    roughness=0.5,
                    specular=0.5,
                )
                builder.add_box_visual(
                    sapien.Pose([length / 2, 0, 0]),
                    half_size=[length / 2, radius, radius],
                    material=mat,
                )
                # peg tail
                mat = sapien.render.RenderMaterial(
                    base_color=sapien_utils.hex2rgba("#EDF6F9"),
                    roughness=0.5,
                    specular=0.5,
                )
                builder.add_box_visual(
                    sapien.Pose([-length / 2, 0, 0]),
                    half_size=[length / 2, radius, radius],
                    material=mat,
                )
                builder.initial_pose = sapien.Pose(p=[0, 0, 0.1])
                builder.set_scene_idxs(scene_idxs)
                peg = builder.build(f"peg_{i}")
                self.remove_from_state_dict_registry(peg)
                # box with hole

                inner_radius, outer_radius, depth = (
                    radius + clearance,
                    length,
                    length,
                )

                builder = _build_box_with_hole(
                    self.scene, inner_radius, outer_radius, depth, center=box_centers[i], color=box_colors[i]
                )
                builder.initial_pose = sapien.Pose(p=[0, 1, 0.1])
                builder.set_scene_idxs(scene_idxs)
                box = builder.build_kinematic(f"box_with_hole_{i}")
                self.remove_from_state_dict_registry(box)
                pegs.append(peg)
                boxes.append(box)
            self.non_merged_pegs_list = pegs
            self.peg = Actor.merge(pegs, "peg")
            self.non_merged_box_list = boxes
            self.box = Actor.merge(boxes, "box_with_hole")

            # to support heterogeneous simulation state dictionaries we register merged versions
            # of the parallel actors
            self.add_to_state_dict_registry(self.peg)
            self.add_to_state_dict_registry(self.box)

            self.target_EE_pose = build_coordinate_frame(self.scene, axis_length=0.05, axis_radius=0.005, name="target_EE_pose", body_type="kinematic")
            self._hidden_objects.append(self.target_EE_pose)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            self.table_mesh = get_table_primitive_mesh_list(self.table_scene.table_length, self.table_scene.table_width, self.table_scene.table_height, global_transform=convert_sapien_pose_to_transform_matrix(self.table_scene.table.pose))

            base_camera_intrinsic_cv = self.scene.sensors['base_camera'].get_params()['intrinsic_cv'][0].clone()
            base_camera_cam2world_gl = self.scene.sensors['base_camera'].get_params()['cam2world_gl'][0].clone() # this is world to cam
            self.tm_camera = create_trimesh_camera(base_camera_intrinsic_cv, base_camera_cam2world_gl.cpu().numpy())

            
            # Initialize the robot
            qpos = torch.tensor(
                self.robot_config.init_qpos
            )
            qpos = qpos.repeat(b, 1)
            qpos[:, -2:] = self.peg_half_sizes[env_idx, 1] + .001 # set gripper width close to peg width
            self.agent.robot.set_qpos(qpos)
            
            # This is for the root pose
            # self.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))
            self.agent.robot.set_pose(sapien.Pose([0., 0, 0]))

            end_effector_pose = self.agent.tcp.pose.raw_pose

            # initialize the box and peg
            # compute peg pose such that its spawned in the gripper
            # .038 from tcp to flat surface of gripper
            peg_pos = torch.zeros((b, 3))
            peg_pos[:, :2] = end_effector_pose[:, :2]
            # add offset in direction of x of end effector
            x_offset = torch.zeros((b, 3))
            x_offset[:, 0] = self.peg_half_sizes[env_idx, 0] - .025
            x_offset = transforms.quaternion_apply(end_effector_pose[:, -4:], x_offset)
            peg_pos += x_offset
            # peg_pos[:, 2] = end_effector_pose[:, 2] - (self.peg_half_sizes[:, 2]) + 0.038 - .0015
            peg_pos[:, 2] = end_effector_pose[:, 2] 

            peg_quat = end_effector_pose[:, -4:]
            # apply 180 intrinsic rotation around z-ax
            self.peg.set_pose(Pose.create_from_pq(peg_pos, peg_quat))

            # xy = randomization.uniform(
            #     low=torch.tensor([-0.1, -0.3]), high=torch.tensor([0.1, 0]), size=(b, 2)
            # )
            # pos = torch.zeros((b, 3))
            # pos[:, :2] = xy
            # pos[:, 2] = self.peg_half_sizes[env_idx, 2]
            # quat = randomization.random_quaternions(
            #     b,
            #     self.device,
            #     lock_x=True,
            #     lock_y=True,
            #     bounds=(np.pi / 2 - np.pi / 3, np.pi / 2 + np.pi / 3),
            # )

            if self.box_config.randomize_x_position:
                x = randomization.uniform(
                    low=torch.tensor(self.box_config.x_position_delta_randomization_bounds[0]+self.box_config.nominal_x_position),
                    high=torch.tensor(self.box_config.x_position_delta_randomization_bounds[1]+self.box_config.nominal_x_position),
                    size=(b,),
                )
            else:
                x = torch.tensor([self.box_config.nominal_x_position]*b)
            if self.box_config.randomize_y_position:
                y = randomization.uniform(
                    low=torch.tensor(self.box_config.y_position_delta_randomization_bounds[0]+self.box_config.nominal_y_position),
                    high=torch.tensor(self.box_config.y_position_delta_randomization_bounds[1]+self.box_config.nominal_y_position),
                    size=(b,),
                )
            else:
                y = torch.tensor([self.box_config.nominal_y_position]*b)
            xy = torch.stack([x, y], dim=1)
            # xy = randomization.uniform(
            #     low=torch.tensor([0.15, 0.2]),
            #     high=torch.tensor([0.2, 0.4]),
            #     size=(b, 2),
            # )
            pos = torch.zeros((b, 3))
            pos[:, :2] = xy
            pos[:, 2] = self.peg_half_sizes[env_idx, 0]
            if self.box_config.randomize_yaw:
                quat = randomization.random_quaternions(
                    b,
                    self.device,
                    lock_x=True,
                    lock_y=True,
                    bounds=(self.box_config.yaw_delta_randomization_bounds[0] + self.box_config.nominal_yaw, self.box_config.yaw_delta_randomization_bounds[1] + self.box_config.nominal_yaw),
                )
            else:
                euler_angles = torch.zeros((b, 3), device=self.device)
                euler_angles[:, 2] = self.box_config.nominal_yaw
                quat = transforms.matrix_to_quaternion(transforms.euler_angles_to_matrix(euler_angles=euler_angles, convention="XYZ"))

            self.box.set_pose(Pose.create_from_pq(pos, quat))

            self.target_EE_pose.set_pose(end_effector_pose)

            self.base_camera_intrinsic = self.scene.sensors['base_camera'].get_params()['intrinsic_cv']
            self.base_camera_cam2world_gl = self.scene.sensors['base_camera'].get_params()['cam2world_gl'][0]
            self.base_camera_extrinsic_cv = self.scene.sensors['base_camera'].get_params()['extrinsic_cv']
            # extrinsic cv is bx3x4 so add a row of [0,0,0,1] to make it bx4x4
            self.base_camera_extrinsic_cv = torch.cat([self.base_camera_extrinsic_cv, torch.tensor([[[0,0,0,1.]]])], dim=1)

    def _after_control_step(self):
        controller_state = self.agent.controller.get_state()
        if 'arm' in controller_state:
            target_EE_pose_in_root = Pose(controller_state['arm']['target_pose'])
            root_pose = self.agent.robot.get_pose()
            target_EE_pose = root_pose * target_EE_pose_in_root
            self.target_EE_pose.set_pose(target_EE_pose)

    def _get_obs_extra(self, info: Dict):
        obs = dict()
        end_effector_pose = self.agent.tcp.pose.raw_pose # bx7
        obs['end_effector_pose'] = end_effector_pose

        W_FT_EE = self.agent.get_external_wrench_at_end_effector(in_world_frame=True)
        obs['W_FT_EE'] = W_FT_EE

        obs['end_effector_pixel_coordinates'] = batched_position_to_pixel_coordinates(end_effector_pose[:, :3].unsqueeze(1), self.base_camera_intrinsic, self.base_camera_extrinsic_cv).squeeze(1)
        
        if self.render_contact_map or self.render_contact_forces_map:
            obs.update(get_extrinsic_contact_map_data(self.num_envs, self.device, self.scene, self.max_extrinsic_contacts, self.camera_height, self.camera_width, self.base_camera_intrinsic, self.base_camera_extrinsic_cv, 'peg', 'panda', return_contact_positions_map=self.render_contact_map, return_contact_forces_map=self.render_contact_forces_map))

        if self.render_dtc_maps or self.render_normals_maps:
            env_mesh_list, env_mesh = self.get_env_object_meshes()
            EE_object_mesh_list, EE_object_mesh = self.get_grasped_object_mesh()
            obs.update(get_extra_contact_features(env_mesh_list, env_mesh, EE_object_mesh_list, EE_object_mesh, self.tm_camera, self.render_dtc_maps, self.render_normals_maps))

        if self.obs_mode_struct.use_state:
            obs.update(
                peg_pose=self.peg.pose.raw_pose,
                peg_half_size=self.peg_half_sizes,
                box_hole_pose=self.box_hole_pose.raw_pose,
                box_hole_radius=self.box_hole_inner_radii,
            )
        
        return obs
    
    def get_grasped_object_mesh(self):
        EE_object_length, EE_object_width, EE_object_height = (self.peg_half_sizes[0]*2).tolist()
        EE_object_mesh_list = get_peg_primitive_mesh_list(EE_object_length, EE_object_width, EE_object_height, global_transform=convert_sapien_pose_to_transform_matrix(self.non_merged_pegs_list[0].pose))
        EE_object_mesh = tm.util.concatenate(EE_object_mesh_list)
        return EE_object_mesh_list, EE_object_mesh

    def get_env_object_meshes(self):
        inner_radius, outer_radius, depth = self.box_hole_inner_radii[0].item(), self.peg_half_sizes[0, 0].item(), self.peg_half_sizes[0, 0].item()
        center = self.box_centers[0].tolist()
        env_object_meshes_list = get_box_meshes_list(inner_radius, outer_radius, depth, center, global_transform=convert_sapien_pose_to_transform_matrix(self.non_merged_box_list[0].pose))
        env_mesh_list = env_object_meshes_list + self.table_mesh

        env_mesh = tm.util.concatenate(env_object_meshes_list + self.table_mesh)
        return env_mesh_list, env_mesh

    @property
    def time_between_env_steps(self):
        # time in seconds between each environment step
        return 1.0/self.sim_config.control_freq
    
    # save some commonly used attributes
    @property
    def peg_head_pos(self):
        return self.peg.pose.p + self.peg_head_offsets.p

    @property
    def peg_head_pose(self):
        return self.peg.pose * self.peg_head_offsets

    @property
    def box_hole_pose(self):
        return self.box.pose * self.box_hole_offsets

    @property
    def goal_pose(self):
        # NOTE (stao): this is fixed after each _initialize_episode call. You can cache this value
        # and simply store it after _initialize_episode or set_state_dict calls.
        return self.box.pose * self.box_hole_offsets * self.peg_head_offsets.inv()

    @property
    def peg_info(self):
        info_dict = dict()
        info_dict['sizes'] = self.peg_half_sizes
        info_dict['mass'] = self.peg.mass
        return info_dict

    @property
    def box_info(self):
        info_dict = dict()
        info_dict['sizes'] = self.box_sizes # inner_radius, outer_radius, depth
        info_dict['centers'] = self.box_centers # 2D center
        info_dict['colors'] = self.box_colors
        info_dict['hole_clearances'] = self.box_hole_clearances # b
        return info_dict

    def has_peg_inserted(self):
        # Only head position is used in fact
        peg_head_pos_at_hole = (self.box_hole_pose.inv() * self.peg_head_pose).p
        # x-axis is hole direction
        x_flag = -0.015 <= peg_head_pos_at_hole[:, 0]
        # x_flag = 0.0 <= peg_head_pos_at_hole[:, 0]
        y_flag = (-self.box_hole_inner_radii <= peg_head_pos_at_hole[:, 1]) & (
            peg_head_pos_at_hole[:, 1] <= self.box_hole_inner_radii
        )
        z_flag = (-self.box_hole_inner_radii <= peg_head_pos_at_hole[:, 2]) & (
            peg_head_pos_at_hole[:, 2] <= self.box_hole_inner_radii
        )
        return (
            x_flag & y_flag & z_flag,
            peg_head_pos_at_hole,
        )
    @property
    def gripper_pose(self):
        with torch.device(self.device):
            return self.agent.tcp.pose.raw_pose

    @property
    def gripper_width(self):
        # returns the batched gripper width
        with torch.device(self.device):
            return self.agent.robot.get_qpos()[:, -2:].sum(axis=1, keepdim=True) # bx1
        
    @property    
    def peg_is_possibly_grasped(self):
        # write our own function of whether the grasped peg is still in the gripper by checking if two cuboids intersect
        # cuboid one is defined as the volume in between the gripper fingers, and cuboid two is the grasped peg
        # return a torch boolean tensor of shape (num_envs,)
        # following Stefan Gottschalk's implementation of OBB SAT https://gamma.cs.unc.edu/users/gottschalk/main.pdf
        with torch.device(self.device):
            # first do a quick but conservative check if cubes are NOT intersecting using the cirumscribed spheres
            gripper_cuboid_half_dims = torch.zeros((self.num_envs, 3), device=self.device)
            gripper_cuboid_half_dims[:, 0] = 0.009
            gripper_cuboid_half_dims[:, 1] = self.gripper_width[:,0]/2
            gripper_cuboid_half_dims[:, 2] = 0.009
            cuboid_A_dict = get_cuboid_dict(self.device, self.gripper_pose, gripper_cuboid_half_dims)
            cuboid_B_dict = get_cuboid_dict(self.device, self.peg.pose.raw_pose, self.peg_half_sizes)
            return cuboid_intersection_test(self.device, cuboid_A_dict, cuboid_B_dict)
    
    def evaluate(self):
        success, peg_head_pos_at_hole = self.has_peg_inserted()
        peg_is_possibly_grasped = self.peg_is_possibly_grasped
        return dict(success=success, peg_head_pos_at_hole=peg_head_pos_at_hole, peg_is_possibly_grasped=peg_is_possibly_grasped)

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # Stage 1: Encourage gripper to be rotated to be lined up with the peg

        # Stage 2: Encourage gripper to move close to peg tail and grasp it
        gripper_pos = self.agent.tcp.pose.p
        tgt_gripper_pose = self.peg.pose
        offset = sapien.Pose(
            [-0.06, 0, 0]
        )  # account for panda gripper width with a bit more leeway
        tgt_gripper_pose = tgt_gripper_pose * (offset)
        gripper_to_peg_dist = torch.linalg.norm(
            gripper_pos - tgt_gripper_pose.p, axis=1
        )

        reaching_reward = 1 - torch.tanh(4.0 * gripper_to_peg_dist)

        # check with max_angle=20 to ensure gripper isn't grasping peg at an awkward pose
        is_grasped = self.agent.is_grasping(self.peg, max_angle=20)
        reward = reaching_reward + is_grasped

        # Stage 3: Orient the grasped peg properly towards the hole

        # pre-insertion award, encouraging both the peg center and the peg head to match the yz coordinates of goal_pose
        peg_head_wrt_goal = self.goal_pose.inv() * self.peg_head_pose
        peg_head_wrt_goal_yz_dist = torch.linalg.norm(
            peg_head_wrt_goal.p[:, 1:], axis=1
        )
        peg_wrt_goal = self.goal_pose.inv() * self.peg.pose
        peg_wrt_goal_yz_dist = torch.linalg.norm(peg_wrt_goal.p[:, 1:], axis=1)

        pre_insertion_reward = 3 * (
            1
            - torch.tanh(
                0.5 * (peg_head_wrt_goal_yz_dist + peg_wrt_goal_yz_dist)
                + 4.5 * torch.maximum(peg_head_wrt_goal_yz_dist, peg_wrt_goal_yz_dist)
            )
        )
        reward += pre_insertion_reward * is_grasped
        # stage 3 passes if peg is correctly oriented in order to insert into hole easily
        pre_inserted = (peg_head_wrt_goal_yz_dist < 0.01) & (
            peg_wrt_goal_yz_dist < 0.01
        )

        # Stage 4: Insert the peg into the hole once it is grasped and lined up
        peg_head_wrt_goal_inside_hole = self.box_hole_pose.inv() * self.peg_head_pose
        insertion_reward = 5 * (
            1
            - torch.tanh(
                5.0 * torch.linalg.norm(peg_head_wrt_goal_inside_hole.p, axis=1)
            )
        )
        reward += insertion_reward * (is_grasped & pre_inserted)

        reward[info["success"]] = 10

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs, action, info) / 10
