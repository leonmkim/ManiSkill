from typing import Any, Dict, Union

import numpy as np
import sapien
import torch

from mani_skill.agents.robots.panda import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Actor, Pose
from mani_skill.utils.structs.types import SimConfig

from mani_skill.utils.geometry.rotation_conversions import quaternion_multiply, axis_angle_to_quaternion


def _build_book(
    scene: ManiSkillScene, 
    length, width, height, # bounding box of the book
    binding_thickness, cover_thickness, cover_overhang,
    book_color="#FFD289", 
):
    if isinstance(book_color, str):
        book_color = sapien_utils.hex2rgba(book_color)

    builder = scene.create_actor_builder()
    pages_length = length - cover_overhang - binding_thickness
    pages_width = width - 2*cover_thickness
    pages_height = height - 2*cover_overhang
    half_sizes = [
        [pages_length/2, pages_width/2, pages_height/2], # pages
        [binding_thickness, width/2, height/2], # binding
        [length/2, cover_thickness/2, height/2], # cover
        [length/2, cover_thickness/2, height/2], # cover
    ]
    poses = [
        sapien.Pose([(binding_thickness - cover_overhang)/2, 0, 0]), # pages
        sapien.Pose([(binding_thickness - length)/2, 0, 0]), # binding
        sapien.Pose([0, (pages_width + cover_thickness)/2, 0]), # cover
        sapien.Pose([0, -(pages_width + cover_thickness)/2, 0]), # cover
    ]

    for i, (half_size, pose) in enumerate(zip(half_sizes, poses)):
        if i == 0:
            # for pages, set color to white
            mat = sapien.render.RenderMaterial(
                base_color=sapien_utils.hex2rgba("#FFFFFF"), roughness=0.5, specular=0.5
            )
        else:
            mat = sapien.render.RenderMaterial(
                base_color=book_color, roughness=0.5, specular=0.5
            )
        builder.add_box_collision(pose, half_size)
        builder.add_box_visual(pose, half_size, material=mat)
    return builder
    
@register_env("BookInsertion-v0", max_episode_steps=100)
class BookInsertionEnv(BaseEnv):
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
        pose = sapien_utils.look_at([0, -0.3, 0.2], [0, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.5, -0.5, 0.8], [0.05, -0.1, 0.4])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        with torch.device(self.device):
            self.table_scene = TableSceneBuilder(self)
            self.table_scene.build()

            binding_thickness = 0.005
            cover_thickness = 0.003
            cover_overhang = 0.003

            num_env_books = 8
            
            grasped_book_lengths = self._batched_episode_rng.uniform(0.1, 0.15)
            grasped_book_widths = self._batched_episode_rng.uniform(0.015, 0.06) # max gripper width is .08
            grasped_book_heights = self._batched_episode_rng.uniform(0.15, 0.25)

            # lengths = self._batched_episode_rng.uniform(0.085, 0.125)
            # radii = self._batched_episode_rng.uniform(0.015, 0.025)
            # centers = (
            #     0.5
            #     * (lengths - radii)[:, None]
            #     * self._batched_episode_rng.uniform(-1, 1, size=(2,))
            # )

            # # save some useful values for use later
            self.grasped_book_sizes = common.to_tensor(np.vstack([grasped_book_lengths, grasped_book_widths, grasped_book_heights])).T
            # self.peg_half_sizes = common.to_tensor(np.vstack([lengths, radii, radii])).T
            # peg_head_offsets = torch.zeros((self.num_envs, 3))
            # peg_head_offsets[:, 0] = self.peg_half_sizes[:, 0]
            # self.peg_head_offsets = Pose.create_from_pq(p=peg_head_offsets)

            # box_hole_offsets = torch.zeros((self.num_envs, 3))
            # box_hole_offsets[:, 1:] = common.to_tensor(centers)
            # self.box_hole_offsets = Pose.create_from_pq(p=box_hole_offsets)
            # self.box_hole_radii = common.to_tensor(radii + self._clearance)

            env_book_lengths = []
            env_book_widths = []
            env_book_heights = []
            env_book_colors = []
            for i in range(num_env_books):
                env_book_lengths.append(self._batched_episode_rng.uniform(0.1, 0.15))
                env_book_widths.append(self._batched_episode_rng.uniform(0.015, 0.04))
                env_book_heights.append(self._batched_episode_rng.uniform(0.2475, 0.2525))

                color = np.ones((self.num_envs, 4))
                color[:,0] = self._batched_episode_rng.uniform(0.0, 1.0)
                color[:,1] = self._batched_episode_rng.uniform(0.0, 1.0)
                color[:,2] = self._batched_episode_rng.uniform(0.0, 1.0)
                env_book_colors.append(color)

            env_book_lengths = np.vstack(env_book_lengths).T # bxN
            env_book_widths = np.vstack(env_book_widths).T # bxN
            env_book_heights = np.vstack(env_book_heights).T # bxN
            # construct bxNx3 tensor
            self.env_book_sizes = common.to_tensor(np.stack([env_book_lengths, env_book_widths, env_book_heights], axis=2))

            env_book_colors = np.stack(env_book_colors,axis=1) # bxNx4
            assert env_book_colors.shape == (self.num_envs, num_env_books, 4), f"env_book_colors shape is incorrect, {env_book_colors.shape}"

            # # in each parallel env we build a different box with a hole and peg (the task is meant to be quite difficult)
            # pegs = []
            # boxes = []
            grasped_books = []
            for i in range(self.num_envs):
                scene_idxs = [i]
                grasped_book_length = grasped_book_lengths[i]
                grasped_book_width = grasped_book_widths[i]
                grasped_book_height = grasped_book_heights[i]

                builder = _build_book(
                    self.scene, 
                    grasped_book_length, grasped_book_width, grasped_book_height, 
                    binding_thickness, cover_thickness, cover_overhang
                )
                builder.initial_pose = sapien.Pose(p=[0, 1, 0.3])
                builder.set_scene_idxs(scene_idxs)
                grasped_book = builder.build(f"grasped_book_{i}")
                self.remove_from_state_dict_registry(grasped_book)
                grasped_books.append(grasped_book)

                # builder.add_box_collision(half_size=[book_length, radius, radius])
                # # peg head
                # mat = sapien.render.RenderMaterial(
                #     base_color=sapien_utils.hex2rgba("#EC7357"),
                #     roughness=0.5,
                #     specular=0.5,
                # )
                # builder.add_box_visual(
                #     sapien.Pose([book_length / 2, 0, 0]),
                #     half_size=[book_length / 2, radius, radius],
                #     material=mat,
                # )
                # # peg tail
                # mat = sapien.render.RenderMaterial(
                #     base_color=sapien_utils.hex2rgba("#EDF6F9"),
                #     roughness=0.5,
                #     specular=0.5,
                # )
                # builder.add_box_visual(
                #     sapien.Pose([-book_length / 2, 0, 0]),
                #     half_size=[book_length / 2, radius, radius],
                #     material=mat,
                # )
                # builder.initial_pose = sapien.Pose(p=[0, 0, 0.1])
                # builder.set_scene_idxs(scene_idxs)
                # peg = builder.build(f"peg_{i}")
                # self.remove_from_state_dict_registry(peg)
                # # box with hole

                # inner_radius, outer_radius, depth = (
                #     radius + self._clearance,
                #     book_length,
                #     book_length,
                # )
                # builder = _build_box_with_hole(
                #     self.scene, inner_radius, outer_radius, depth, center=centers[i]
                # )
                # builder.initial_pose = sapien.Pose(p=[0, 1, 0.1])
                # builder.set_scene_idxs(scene_idxs)
                # box = builder.build_kinematic(f"box_with_hole_{i}")
                # self.remove_from_state_dict_registry(box)
                # pegs.append(peg)
                # boxes.append(box)

            # self.peg = Actor.merge(pegs, "peg")
            # self.box = Actor.merge(boxes, "box_with_hole")
            self.grasped_book = Actor.merge(grasped_books, "grasped_book")
            self.add_to_state_dict_registry(self.grasped_book)

            env_books = []
            for j in range(num_env_books):
                envs_per_env_book = []
                for i in range(self.num_envs):
                    book_length = env_book_lengths[i, j]
                    book_width = env_book_widths[i, j]
                    book_height = env_book_heights[i, j]
                    builder = _build_book(
                        self.scene, 
                        book_length, book_width, book_height, 
                        binding_thickness, cover_thickness, cover_overhang,
                        book_color=env_book_colors[i, j]
                    )
                    builder.initial_pose = sapien.Pose(p=[0, -1, 0.5*(j+1)])
                    builder.set_scene_idxs(scene_idxs)
                    env_book = builder.build(f"book_{j}_{i}")
                    self.remove_from_state_dict_registry(env_book)
                    envs_per_env_book.append(env_book)

                env_books.append(envs_per_env_book)
            
            # want to make Nxb env books
                
            for j in range(num_env_books):
                envs_per_env_book = env_books[j]
                env_books[j] = Actor.merge(envs_per_env_book, f"book_{j}")
                self.add_to_state_dict_registry(env_books[j])

            self.env_books = env_books

            # to support heterogeneous simulation state dictionaries we register merged versions
            # of the parallel actors
            # self.add_to_state_dict_registry(self.peg)
            # self.add_to_state_dict_registry(self.box)


    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # xy = randomization.uniform(
            #     low=torch.tensor([-0.05, 0.2]),
            #     high=torch.tensor([0.05, 0.4]),
            #     size=(b, 2),
            # )
            # pos = torch.zeros((b, 3))
            # pos[:, :2] = xy
            # pos[:, 2] = self.peg_half_sizes[env_idx, 0]
            # quat = randomization.random_quaternions(
            #     b,
            #     self.device,
            #     lock_x=True,
            #     lock_y=True,
            #     bounds=(np.pi / 2 - np.pi / 8, np.pi / 2 + np.pi / 8),
            # )
            # self.box.set_pose(Pose.create_from_pq(pos, quat))
            # Initialize the robot
            qpos = torch.tensor(
                [
                    0.022516679397616424, 
                    0.11646689505116431, 
                    -0.3625673227601117, 
                    -1.37265637618617, 
                    0.033468631741809286, 
                    1.4658307538809252, 
                    0.46052758571920294,
                    .04,
                    .04,
                ]
            )
            # qpos = np.array(
            #     [
            #         0.0,
            #         np.pi / 8,
            #         0,
            #         -np.pi * 5 / 8,
            #         0,
            #         np.pi * 3 / 4,
            #         -np.pi / 4,
            #         0.04,
            #         0.04,
            #     ]
            # )
            # qpos = self._episode_rng.normal(0, 0.02, (b, len(qpos))) + qpos
            # repeat over batch dimension
            # qpos = np.repeat(qpos[None, :], b, axis=0)
            qpos = qpos.repeat(b, 1)
            qpos[:, -2:] = (self.grasped_book_sizes[:, 1])/2 + .001
            self.agent.robot.set_qpos(qpos)
            self.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))

            if self.backend.sim_backend == 'physx_cuda':
                # ensure all updates to object poses and configurations are applied on GPU after task initialization
                self.scene._gpu_apply_all()
                self.scene.px.gpu_update_articulation_kinematics()
                self.scene._gpu_fetch_all()

            end_effector_pose = self.agent.tcp.pose.raw_pose
            
            # # initialize the box and peg
            # xy = randomization.uniform(
            #     low=torch.tensor([-0.1, -0.3]), high=torch.tensor([0.1, 0]), size=(b, 2)
            # )
            # pos = torch.zeros((b, 3))
            # pos[:, :2] = xy
            # # pos[:, 2] = self.peg_half_sizes[env_idx, 2]
            # pos[:, 2] = 0.3
            # quat = randomization.random_quaternions(
            #     b,
            #     self.device,
            #     lock_x=True,
            #     lock_y=True,
            #     bounds=(np.pi / 2 - np.pi / 3, np.pi / 2 + np.pi / 3),
            # )
            # self.peg.set_pose(Pose.create_from_pq(pos, quat))

            # .038 from tcp to flat surface of gripper
            pos = torch.zeros((b, 3))
            pos[:, :2] = end_effector_pose[:, :2]
            pos[:, 2] = end_effector_pose[:, 2] - (self.grasped_book_sizes[:,2]/2) + 0.038 - .0015

            quat = end_effector_pose[:, -4:]
            # apply 180 intrinsic rotation around z-axis
            quat = quaternion_multiply(quat, axis_angle_to_quaternion(torch.tensor([0, 0, np.pi])))
            self.grasped_book.set_pose(Pose.create_from_pq(pos, quat))

    # # save some commonly used attributes
    # @property
    # def peg_head_pos(self):
    #     return self.peg.pose.p + self.peg_head_offsets.p

    # @property
    # def peg_head_pose(self):
    #     return self.peg.pose * self.peg_head_offsets

    # @property
    # def box_hole_pose(self):
    #     return self.box.pose * self.box_hole_offsets

    # @property
    # def goal_pose(self):
    #     # NOTE (stao): this is fixed after each _initialize_episode call. You can cache this value
    #     # and simply store it after _initialize_episode or set_state_dict calls.
    #     return self.box.pose * self.box_hole_offsets * self.peg_head_offsets.inv()

    # def has_peg_inserted(self):
    #     # Only head position is used in fact
    #     peg_head_pos_at_hole = (self.box_hole_pose.inv() * self.peg_head_pose).p
    #     # x-axis is hole direction
    #     x_flag = -0.015 <= peg_head_pos_at_hole[:, 0]
    #     y_flag = (-self.box_hole_radii <= peg_head_pos_at_hole[:, 1]) & (
    #         peg_head_pos_at_hole[:, 1] <= self.box_hole_radii
    #     )
    #     z_flag = (-self.box_hole_radii <= peg_head_pos_at_hole[:, 2]) & (
    #         peg_head_pos_at_hole[:, 2] <= self.box_hole_radii
    #     )
    #     return (
    #         x_flag & y_flag & z_flag,
    #         peg_head_pos_at_hole,
    #     )

    # def evaluate(self):
    #     # success, peg_head_pos_at_hole = self.has_peg_inserted()
    #     # return dict(success=success, peg_head_pos_at_hole=peg_head_pos_at_hole)
    #     return dict(success=False)

    # def _get_obs_extra(self, info: Dict):
    #     obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
    #     if self._obs_mode in ["state", "state_dict"]:
    #         obs.update(
    #             peg_pose=self.peg.pose.raw_pose,
    #             peg_half_size=self.peg_half_sizes,
    #             box_hole_pose=self.box_hole_pose.raw_pose,
    #             box_hole_radius=self.box_hole_radii,
    #         )
    #     return obs

    # def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
    #     # Stage 1: Encourage gripper to be rotated to be lined up with the peg

    #     # Stage 2: Encourage gripper to move close to peg tail and grasp it
    #     gripper_pos = self.agent.tcp.pose.p
    #     tgt_gripper_pose = self.peg.pose
    #     offset = sapien.Pose(
    #         [-0.06, 0, 0]
    #     )  # account for panda gripper width with a bit more leeway
    #     tgt_gripper_pose = tgt_gripper_pose * (offset)
    #     gripper_to_peg_dist = torch.linalg.norm(
    #         gripper_pos - tgt_gripper_pose.p, axis=1
    #     )

    #     reaching_reward = 1 - torch.tanh(4.0 * gripper_to_peg_dist)

    #     # check with max_angle=20 to ensure gripper isn't grasping peg at an awkward pose
    #     is_grasped = self.agent.is_grasping(self.peg, max_angle=20)
    #     reward = reaching_reward + is_grasped

    #     # Stage 3: Orient the grasped peg properly towards the hole

    #     # pre-insertion award, encouraging both the peg center and the peg head to match the yz coordinates of goal_pose
    #     peg_head_wrt_goal = self.goal_pose.inv() * self.peg_head_pose
    #     peg_head_wrt_goal_yz_dist = torch.linalg.norm(
    #         peg_head_wrt_goal.p[:, 1:], axis=1
    #     )
    #     peg_wrt_goal = self.goal_pose.inv() * self.peg.pose
    #     peg_wrt_goal_yz_dist = torch.linalg.norm(peg_wrt_goal.p[:, 1:], axis=1)

    #     pre_insertion_reward = 3 * (
    #         1
    #         - torch.tanh(
    #             0.5 * (peg_head_wrt_goal_yz_dist + peg_wrt_goal_yz_dist)
    #             + 4.5 * torch.maximum(peg_head_wrt_goal_yz_dist, peg_wrt_goal_yz_dist)
    #         )
    #     )
    #     reward += pre_insertion_reward * is_grasped
    #     # stage 3 passes if peg is correctly oriented in order to insert into hole easily
    #     pre_inserted = (peg_head_wrt_goal_yz_dist < 0.01) & (
    #         peg_wrt_goal_yz_dist < 0.01
    #     )

    #     # Stage 4: Insert the peg into the hole once it is grasped and lined up
    #     peg_head_wrt_goal_inside_hole = self.box_hole_pose.inv() * self.peg_head_pose
    #     insertion_reward = 5 * (
    #         1
    #         - torch.tanh(
    #             5.0 * torch.linalg.norm(peg_head_wrt_goal_inside_hole.p, axis=1)
    #         )
    #     )
    #     reward += insertion_reward * (is_grasped & pre_inserted)

    #     reward[info["success"]] = 10

    #     return reward

    # def compute_normalized_dense_reward(
    #     self, obs: Any, action: torch.Tensor, info: Dict
    # ):
    #     return self.compute_dense_reward(obs, action, info) / 10
