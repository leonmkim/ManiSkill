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
from mani_skill.utils.scene_builder.table import TableSceneBuilder, SimpleTableSceneBuilder
from mani_skill.utils.structs import Actor, Pose
from mani_skill.utils.structs.types import SimConfig

from pytorch3d import transforms

def _build_box_with_hole(
    scene: ManiSkillScene, inner_radius, outer_radius, depth, center=(0, 0)
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
        base_color=sapien_utils.hex2rgba("#FFD289"), roughness=0.5, specular=0.5
    )

    for half_size, pose in zip(half_sizes, poses):
        builder.add_box_collision(pose, half_size)
        builder.add_box_visual(pose, half_size, material=mat)
    return builder

@dataclass
class RobotConfig:
    """
    Configuration for the robot in the BookInsertionEnv.
    """
    init_qpos: list = field(default_factory=lambda: [0.022516679397616424, 0.11646689505116431, -0.3625673227601117, -1.37265637618617, 0.033468631741809286, 1.4658307538809252, 0.46052758571920294,.04,.04,])
    # additive_y_randomization_bounds: Union[float, list] = 0.0

@dataclass
class HoleConfig:
    """
    Configuration for the hole in the PegInsertionSideCustomEnv.
    """
    randomize_color: bool = True
    randomize_tolerance: bool = True
    tolerance_randomization_bounds: list = field(default_factory=lambda: [0.001, 0.005]) # default tolerance is .003m
    randomize_x_position: bool = True
    x_position_randomization_bounds: list = field(default_factory=lambda: [-0.05, 0.05])
    randomize_y_position: bool = True
    y_position_randomization_bounds: list = field(default_factory=lambda: [-0.1, 0.1])
    randomize_yaw: bool = True
    # yaw_randomization_bounds: list = field(default_factory=lambda: [-np.pi / 8, np.pi / 8])
    yaw_randomization_bounds: list = field(default_factory=lambda: [np.pi/8, np.pi / 4])
    nominal_x_position: float = 0.35
    nominal_y_position: float = 0.3
    nominal_yaw: float = np.pi / 2

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
                isinstance(self.x_position_randomization_bounds, list)
                and len(self.x_position_randomization_bounds) == 2
                and self.x_position_randomization_bounds[0] < self.x_position_randomization_bounds[1]
            ), f"x_position_randomization_bounds must be a list of two values with the first one smaller than the second, but got {self.x_position_randomization_bounds}"
        if self.randomize_y_position:
            # assert bounds for y position are valid
            assert (
                isinstance(self.y_position_randomization_bounds, list)
                and len(self.y_position_randomization_bounds) == 2
                and self.y_position_randomization_bounds[0] < self.y_position_randomization_bounds[1]
            ), f"y_position_randomization_bounds must be a list of two values with the first one smaller than the second, but got {self.y_position_randomization_bounds}"
        if self.randomize_yaw:
            # assert bounds for yaw are valid
            assert (
                isinstance(self.yaw_randomization_bounds, list)
                and len(self.yaw_randomization_bounds) == 2
                and self.yaw_randomization_bounds[0] < self.yaw_randomization_bounds[1]
            ), f"yaw_randomization_bounds must be a list of two values with the first one smaller than the second, but got {self.yaw_randomization_bounds}"

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
    _clearance = 0.003

    cam_resize_factor: float = 0.5

    robot_config: RobotConfig = RobotConfig()

    hole_config: HoleConfig = HoleConfig()

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

        super().__init__(
            *args,
            robot_uids=robot_uids,
            num_envs=num_envs,
            reconfiguration_freq=reconfiguration_freq,
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

            lengths = self._batched_episode_rng.uniform(0.085, 0.125)
            radii = self._batched_episode_rng.uniform(0.015, 0.025)
            centers = (
                0.5
                * (lengths - radii)[:, None]
                * self._batched_episode_rng.uniform(-1, 1, size=(2,))
            )

            # save some useful values for use later
            self.peg_half_sizes = common.to_tensor(np.vstack([lengths, radii, radii])).T
            peg_head_offsets = torch.zeros((self.num_envs, 3))
            peg_head_offsets[:, 0] = self.peg_half_sizes[:, 0]
            self.peg_head_offsets = Pose.create_from_pq(p=peg_head_offsets)

            box_hole_offsets = torch.zeros((self.num_envs, 3))
            box_hole_offsets[:, 1:] = common.to_tensor(centers)
            self.box_hole_offsets = Pose.create_from_pq(p=box_hole_offsets)
            self.box_hole_radii = common.to_tensor(radii + self._clearance)

            # in each parallel env we build a different box with a hole and peg (the task is meant to be quite difficult)
            pegs = []
            boxes = []

            for i in range(self.num_envs):
                scene_idxs = [i]
                length = lengths[i]
                radius = radii[i]
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
                    radius + self._clearance,
                    length,
                    length,
                )
                builder = _build_box_with_hole(
                    self.scene, inner_radius, outer_radius, depth, center=centers[i]
                )
                builder.initial_pose = sapien.Pose(p=[0, 1, 0.1])
                builder.set_scene_idxs(scene_idxs)
                box = builder.build_kinematic(f"box_with_hole_{i}")
                self.remove_from_state_dict_registry(box)
                pegs.append(peg)
                boxes.append(box)
            self.peg = Actor.merge(pegs, "peg")
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

            # Initialize the robot
            qpos = torch.tensor(
                self.robot_config.init_qpos
            )
            qpos = qpos.repeat(b, 1)
            # qpos[:, -2:] = (self.grasped_book_sizes[:, 1])/2 + .001 # set gripper width close to peg width
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
            peg_pos[:, 2] = end_effector_pose[:, 2] - (self.peg_half_sizes[:, 2]) + 0.038 - .0015

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

            if self.hole_config.randomize_x_position:
                x = randomization.uniform(
                    low=torch.tensor(self.hole_config.x_position_randomization_bounds[0]+self.hole_config.nominal_x_position),
                    high=torch.tensor(self.hole_config.x_position_randomization_bounds[1]+self.hole_config.nominal_x_position),
                    size=(b,),
                )
            else:
                x = torch.tensor([self.hole_config.nominal_x_position]*b)
            if self.hole_config.randomize_y_position:
                y = randomization.uniform(
                    low=torch.tensor(self.hole_config.y_position_randomization_bounds[0]+self.hole_config.nominal_y_position),
                    high=torch.tensor(self.hole_config.y_position_randomization_bounds[1]+self.hole_config.nominal_y_position),
                    size=(b,),
                )
            else:
                y = torch.tensor([self.hole_config.nominal_y_position]*b)
            xy = torch.stack([x, y], dim=1)
            # xy = randomization.uniform(
            #     low=torch.tensor([0.15, 0.2]),
            #     high=torch.tensor([0.2, 0.4]),
            #     size=(b, 2),
            # )
            pos = torch.zeros((b, 3))
            pos[:, :2] = xy
            pos[:, 2] = self.peg_half_sizes[env_idx, 0]
            if self.hole_config.randomize_yaw:
                quat = randomization.random_quaternions(
                    b,
                    self.device,
                    lock_x=True,
                    lock_y=True,
                    bounds=(self.hole_config.yaw_randomization_bounds[0] + self.hole_config.nominal_yaw, self.hole_config.yaw_randomization_bounds[1] + self.hole_config.nominal_yaw),
                )
            else:
                euler_angles = torch.zeros((b, 3), device=self.device)
                euler_angles[:, 2] = self.hole_config.nominal_yaw
                quat = transforms.matrix_to_quaternion(transforms.euler_angles_to_matrix(euler_angles=euler_angles, convention="XYZ"))

            self.box.set_pose(Pose.create_from_pq(pos, quat))

            self.target_EE_pose.set_pose(end_effector_pose)

    def _after_control_step(self):
        controller_state = self.agent.controller.get_state()
        if 'arm' in controller_state:
            target_EE_pose_in_root = Pose(controller_state['arm']['target_pose'])
            root_pose = self.agent.robot.get_pose()
            target_EE_pose = root_pose * target_EE_pose_in_root
            self.target_EE_pose.set_pose(target_EE_pose)
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

    def has_peg_inserted(self):
        # Only head position is used in fact
        peg_head_pos_at_hole = (self.box_hole_pose.inv() * self.peg_head_pose).p
        # x-axis is hole direction
        x_flag = -0.015 <= peg_head_pos_at_hole[:, 0]
        y_flag = (-self.box_hole_radii <= peg_head_pos_at_hole[:, 1]) & (
            peg_head_pos_at_hole[:, 1] <= self.box_hole_radii
        )
        z_flag = (-self.box_hole_radii <= peg_head_pos_at_hole[:, 2]) & (
            peg_head_pos_at_hole[:, 2] <= self.box_hole_radii
        )
        return (
            x_flag & y_flag & z_flag,
            peg_head_pos_at_hole,
        )

    def evaluate(self):
        success, peg_head_pos_at_hole = self.has_peg_inserted()
        return dict(success=success, peg_head_pos_at_hole=peg_head_pos_at_hole)

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self.obs_mode_struct.use_state:
            obs.update(
                peg_pose=self.peg.pose.raw_pose,
                peg_half_size=self.peg_half_sizes,
                box_hole_pose=self.box_hole_pose.raw_pose,
                box_hole_radius=self.box_hole_radii,
            )
        return obs

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
