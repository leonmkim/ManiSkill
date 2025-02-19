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
from mani_skill.utils.scene_builder.table import TableSceneBuilder, SimpleTableSceneBuilder
from mani_skill.utils.building.actors.common import build_coordinate_frame
from mani_skill.utils.structs import Actor, Pose
from mani_skill.utils.structs.types import SimConfig

from mani_skill.utils.geometry.rotation_conversions import quaternion_multiply, axis_angle_to_quaternion, quaternion_apply
from mani_skill.utils.geometry.geometry import transform_points
import einops

def _build_book(
    scene: ManiSkillScene, 
    length, width, height, # bounding box of the book
    binding_thickness, cover_thickness, cover_overhang,
    book_color="#FFD289", 
    density=0.705,
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
        # if i != 1: # skip adding collision for binding to try to speed sim
        builder.add_box_collision(pose, half_size, density=density)
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
    
    num_env_books: int = 8
    slot_left_of_book_index: int = 4

    binding_thickness: float = 0.005
    cover_thickness: float = 0.003
    cover_overhang: float = 0.005

    cam_resize_factor: float = 0.5

    max_extrinsic_contacts: int = 50 # for padding

    # success conditions
    book_toppled_angle_with_vertical_threshold: float = np.deg2rad(45)
    # from base of gripper fingers to tip of fingers is .047m
    top_of_grasped_book_distance_to_top_of_slot_threshold: float = 0.047 + .02
    success_duration_threshold: float = 3.0 # seconds

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
        
        # get list of specific kwargs defined above
        # special_kwargs = ['num_env_books', 'slot_left_of_book_index', 'cam_resize_factor']
        # for key in special_kwargs:
        #     if key in kwargs:
        #         setattr(self, key, kwargs[key])
        #         del kwargs[key]

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
        from mani_skill.utils.geometry.rotation_conversions import matrix_to_quaternion
        # pose = sapien_utils.look_at([0, -0.3, 0.2], [0, 0, 0.1])
        self.camera_width = 640
        self.camera_height = 480
        intrinsics = torch.tensor([[596.61175537,0.,323.86328125],
                                [0.,596.96472168,246.78981018],
                                [0.,0.,1.]])
        if self.cam_resize_factor != 1.0:
            intrinsics[:2, :3] *= self.cam_resize_factor
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
        world_tf_cam = sapien_utils.look_at(eye, look_at)

        return [CameraConfig("base_camera", world_tf_cam, width=self.camera_width, height=self.camera_height, intrinsic=intrinsics, near=0.01, far=5.0)]

    @property
    def _default_human_render_camera_configs(self):
        # pose = sapien_utils.look_at([0.5, -0.5, 0.8], [0.05, -0.1, 0.4])
        world_tf_root = self.agent.robot.get_pose()
        look_at = world_tf_root.raw_pose[0,:3] + torch.tensor([0.,0,0.25])
        eye = torch.tensor([1.05775+.1, 0, 0.375615])
        pose = sapien_utils.look_at(eye, look_at)

        # return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)
        return CameraConfig("render_camera", pose, 128, 128, 1, 0.01, 5.0)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        with torch.device(self.device):
            self.table_scene = SimpleTableSceneBuilder(self)
            self.table_scene.build()

            self.target_EE_pose = build_coordinate_frame(self.scene, axis_length=0.05, axis_radius=0.005, name="target_EE_pose", body_type="kinematic")
            self._hidden_objects.append(self.target_EE_pose)

            # >>>>>>>>> for debugging
            # self.top_of_slot_viz_pose = build_coordinate_frame(self.scene, axis_length=0.05, axis_radius=0.005, name="top_of_slot_viz_pose", body_type="kinematic")
            # self._hidden_objects.append(self.top_of_slot_viz_pose)

            # self.bottom_inner_corner_of_book_left_of_slot_viz_pose = build_coordinate_frame(self.scene, axis_length=0.05, axis_radius=0.005, name="bottom_inner_corner_of_book_left_of_slot_viz_pose", body_type="kinematic")
            # self._hidden_objects.append(self.bottom_inner_corner_of_book_left_of_slot_viz_pose)

            # self.bottom_inner_corner_of_book_right_of_slot_viz_pose = build_coordinate_frame(self.scene, axis_length=0.05, axis_radius=0.005, name="bottom_inner_corner_of_book_right_of_slot_viz_pose", body_type="kinematic")
            # self._hidden_objects.append(self.bottom_inner_corner_of_book_right_of_slot_viz_pose)

            # self.bottom_of_grasped_book_viz_pose = build_coordinate_frame(self.scene, axis_length=0.05, axis_radius=0.005, name="bottom_of_grasped_book_viz_pose", body_type="kinematic")
            # self._hidden_objects.append(self.bottom_of_grasped_book_viz_pose)

            # self.top_of_grasped_book_viz_pose = build_coordinate_frame(self.scene, axis_length=0.05, axis_radius=0.005, name="top_of_grasped_book_viz_pose", body_type="kinematic")
            # self._hidden_objects.append(self.top_of_grasped_book_viz_pose)
            # <<<<<<<<< for debugging

            self.camera_pose = build_coordinate_frame(self.scene, axis_length=0.05, axis_radius=0.005, name="camera_pose", body_type="kinematic")
            self._hidden_objects.append(self.camera_pose)

            grasped_book_lengths = self._batched_episode_rng.uniform(0.1, 0.15)
            grasped_book_widths = self._batched_episode_rng.uniform(0.03, 0.065) # max gripper width is .08
            grasped_book_heights = self._batched_episode_rng.uniform(0.165, 0.25)
            grasped_book_densities = self._batched_episode_rng.uniform(650, 850)
            grasped_book_colors = np.ones((self.num_envs, 4))
            grasped_book_colors[:,0] = self._batched_episode_rng.uniform(0.0, 1.0)
            grasped_book_colors[:,1] = self._batched_episode_rng.uniform(0.0, 1.0)
            grasped_book_colors[:,2] = self._batched_episode_rng.uniform(0.0, 1.0)

            # # save some useful values for use later
            self.grasped_book_sizes = common.to_tensor(np.vstack([grasped_book_lengths, grasped_book_widths, grasped_book_heights])).T

            env_book_lengths = []
            env_book_widths = []
            env_book_heights = []
            env_book_colors = []
            env_book_densities = []
            for i in range(self.num_env_books):
                env_book_lengths.append(self._batched_episode_rng.uniform(0.15, 0.2))
                env_book_widths.append(self._batched_episode_rng.uniform(0.015, 0.05))
                env_book_heights.append(self._batched_episode_rng.uniform(0.2475, 0.2525))
                env_book_densities.append(self._batched_episode_rng.uniform(655, 1015))

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
            env_book_densities = common.to_tensor(np.stack(env_book_densities, axis=1)) # bxN
            env_book_colors = np.stack(env_book_colors,axis=1) # bxNx4
            assert env_book_colors.shape == (self.num_envs, self.num_env_books, 4), f"env_book_colors shape is incorrect, {env_book_colors.shape}"

            grasped_books = []
            for i in range(self.num_envs):
                scene_idxs = [i]
                grasped_book_length = grasped_book_lengths[i]
                grasped_book_width = grasped_book_widths[i]
                grasped_book_height = grasped_book_heights[i]

                builder = _build_book(
                    self.scene, 
                    grasped_book_length, grasped_book_width, grasped_book_height, 
                    self.binding_thickness, self.cover_thickness, self.cover_overhang,
                    book_color=grasped_book_colors[i],
                    density=grasped_book_densities[i],
                )
                builder.initial_pose = sapien.Pose(p=[0, 1, 0.3])
                builder.set_scene_idxs(scene_idxs)
                grasped_book = builder.build(f"grasped_book_{i}")
                self.remove_from_state_dict_registry(grasped_book)
                grasped_books.append(grasped_book)

            self.grasped_book = Actor.merge(grasped_books, "grasped_book")
            self.add_to_state_dict_registry(self.grasped_book)

            # self.grasped_book.set_collision_group_bit(group=2, bit_idx=, bit=1)
            # fingers is 00000008 (8)    00000000000000000000000000001000
            # w/o self coll: 20000008    00100000000000000000000000001000
            # hand is 00000067 (103),    00000000000000000000000001100111
            # w/o self coll: 20000067    00100000000000000000000001100111
            # 7th link is 00000034 (52), 00000000000000000000000000110100
            # w/o self coll: 20000034    00100000000000000000000000110100
            # 6th link is 00000042 (66), 00000000000000000000000001000010
            # w/o self coll: 20000042    00100000000000000000000001000010
            # 5th link is 00000011 (17), 00000000000000000000000000010001
            # w/o self coll: 20000011    00100000000000000000000000010001
            # all other links become 20000000 w/o self collision
            # self.grasped_book.set_collision_group(group=2, value=2147483647)

            env_books = []
            for j in range(self.num_env_books):
                envs_per_env_book = []
                for i in range(self.num_envs):
                    book_length = env_book_lengths[i, j]
                    book_width = env_book_widths[i, j]
                    book_height = env_book_heights[i, j]
                    builder = _build_book(
                        self.scene, 
                        book_length, book_width, book_height, 
                        self.binding_thickness, self.cover_thickness, self.cover_overhang,
                        book_color=env_book_colors[i, j],
                        density=env_book_densities[i, j],
                    )
                    builder.initial_pose = sapien.Pose(p=[0, -1, 0.5*(j+1)])
                    builder.set_scene_idxs(scene_idxs)
                    env_book = builder.build(f"book_{j}_{i}")
                    self.remove_from_state_dict_registry(env_book)
                    envs_per_env_book.append(env_book)

                env_books.append(envs_per_env_book)

            # env_book_collision_indices = [0]
            # for j in range(self.num_env_books-2):
            #     env_book_collision_indices.append(env_book_collision_indices[j]+(j+2))


            # env_book_collision_group = 0
            # for idx in env_book_collision_indices:
            #     env_book_collision_group |= 1 << idx

            # want to make Nxb env books
                
            for j in range(self.num_env_books):
                envs_per_env_book = env_books[j]
                env_books[j] = Actor.merge(envs_per_env_book, f"book_{j}")
                self.add_to_state_dict_registry(env_books[j])
            #     env_books[j].set_collision_group(group=2, value=env_book_collision_group)
            #     env_book_collision_group = env_book_collision_group << 1

            self.env_books = env_books

            # to support heterogeneous simulation state dictionaries we register merged versions
            # of the parallel actors
            # self.add_to_state_dict_registry(self.peg)
            # self.add_to_state_dict_registry(self.box)
            
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

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
            qpos = qpos.repeat(b, 1)
            qpos[:, -2:] = (self.grasped_book_sizes[:, 1])/2 + .001
            self.agent.robot.set_qpos(qpos)
            # self.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))
            self.agent.robot.set_pose(sapien.Pose([0., 0, 0]))

            if self.backend.sim_backend == 'physx_cuda':
                # ensure all updates to object poses and configurations are applied on GPU after task initialization
                self.scene._gpu_apply_all()
                self.scene.px.gpu_update_articulation_kinematics()
                self.scene._gpu_fetch_all()

            end_effector_pose = self.agent.tcp.pose.raw_pose
            
            # .038 from tcp to flat surface of gripper
            pos = torch.zeros((b, 3))
            pos[:, :2] = end_effector_pose[:, :2]
            pos[:, 2] = end_effector_pose[:, 2] - (self.grasped_book_sizes[:,2]/2) + 0.038 - .0015

            quat = end_effector_pose[:, -4:]
            # apply 180 intrinsic rotation around z-axis
            quat = quaternion_multiply(quat, axis_angle_to_quaternion(torch.tensor([0, 0, np.pi])))
            self.grasped_book.set_pose(Pose.create_from_pq(pos, quat))

            self.xy_slot_location = torch.zeros((b, 2))
            self.xy_slot_location[:, 0] = end_effector_pose[:, 0]
            self.xy_slot_location[:, 1] = 0

            slot_width = self.grasped_book_sizes[:, 1] - .0035

            quat = torch.tensor([0., 0, 0, 1]).repeat(b, 1)
            # compute the env book poses
            for j in range(len(self.env_books)):
                pos = torch.zeros((b, 3))
                pos[:, 0] = self.xy_slot_location[:, 0] - self.env_book_sizes[:, j, 0]/2 + .15/2
                pos[:, 2] = self.env_book_sizes[:, j, 2]/2 + .001
                if j < self.slot_left_of_book_index:
                    pos[:, 1] = self.xy_slot_location[:, 1] - ((slot_width/2) + self.env_book_sizes[:, j+1:self.slot_left_of_book_index, 1].sum(dim=1))
                    pos[:, 1] += -self.env_book_sizes[:, j, 1]/2
                else:
                    pos[:, 1] = self.xy_slot_location[:, 1] + ((slot_width/2) + self.env_book_sizes[:, self.slot_left_of_book_index:j, 1].sum(dim=1))
                    pos[:, 1] += self.env_book_sizes[:, j, 1]/2

                self.env_books[j].set_pose(Pose.create_from_pq(pos, quat))   

            # target_EE_pose = self.agent.controller.get_state()['arm']['target_pose']
            self.target_EE_pose.set_pose(end_effector_pose)

            from mani_skill.utils.geometry.rotation_conversions import matrix_to_quaternion
            camera_pose = self.scene.sensors['base_camera'].get_params()['cam2world_gl'][0]
            cam_rot = quaternion_multiply(matrix_to_quaternion(camera_pose[:3, :3]), axis_angle_to_quaternion(torch.tensor([np.pi, 0, 0])))
            self.camera_pose.set_pose(Pose.create_from_pq(p=camera_pose[:3, 3], q=cam_rot))

            self.base_camera_intrinsic = self.scene.sensors['base_camera'].get_params()['intrinsic_cv']
            self.base_camera_cam2world_gl = self.scene.sensors['base_camera'].get_params()['cam2world_gl'][0]
            self.base_camera_extrinsic_cv = self.scene.sensors['base_camera'].get_params()['extrinsic_cv']
            # extrinsic cv is bx3x4 so add a row of [0,0,0,1] to make it bx4x4
            self.base_camera_extrinsic_cv = torch.cat([self.base_camera_extrinsic_cv, torch.tensor([[[0,0,0,1.]]])], dim=1)

            self.elapsed_success_duration = torch.zeros(b)
            self.last_eval_bool = torch.zeros(b, dtype=torch.bool)

    # def _after_simulation_step(self):
    #     # update viz poses
    #     self.top_of_slot_viz_pose.set_pose(self.top_of_slot_pose)

    #     self.bottom_inner_corner_of_book_left_of_slot_viz_pose.set_pose(self.bottom_inner_corner_of_book_left_of_slot_pose)

    #     self.bottom_inner_corner_of_book_right_of_slot_viz_pose.set_pose(self.bottom_inner_corner_of_book_right_of_slot_pose)

    #     self.bottom_of_grasped_book_viz_pose.set_pose(self.bottom_of_grasped_book_pose)

    #     self.top_of_grasped_book_viz_pose.set_pose(self.top_of_grasped_book_pose)

    def _after_control_step(self):
        # update target EE pose
        target_EE_pose_in_root = Pose(self.agent.controller.get_state()['arm']['target_pose'])
        root_pose = self.agent.robot.get_pose()
        target_EE_pose = root_pose * target_EE_pose_in_root
        self.target_EE_pose.set_pose(target_EE_pose)
    
    def _get_obs_extra(self, info):
        extra = dict()
        # if 'contact' in self._obs_mode:
        extra['extrinsic_contact_positions'] = self.get_extrinsic_contact_positions()
        extra['extrinsic_contact_map'] = self.project_contact_positions_to_camera(extra['extrinsic_contact_positions'])

        # get current end effector pose
        end_effector_pose = self.agent.tcp.pose.raw_pose # bx7

        extra['end_effector_pose'] = end_effector_pose

        # get end_effector pixel coordinates
        extra['end_effector_pixel_coordinates'] = self.batched_position_to_pixel_coordinates(end_effector_pose[:, :3].unsqueeze(1)).squeeze(1)

        return extra
    
    def batched_position_to_pixel_coordinates(self, positions):
        # positions: bxNx3
        assert positions.shape[-1] == 3, "positions must have shape bxNx3"
        b, N, _ = positions.shape
        positions = einops.rearrange(positions, 'b n c -> (b n) c')
        # bx4x4 @ b*Nx3 -> b*Nx3
        # contact_positions_in_cam = transform_points(contact_positions, self.base_camera_extrinsic_cv)
        positions_in_cam = torch.cat([positions, torch.ones((b*N, 1), device=positions.device)], dim=1)
        positions_in_cam = einops.rearrange(torch.bmm(self.base_camera_extrinsic_cv, (positions_in_cam.T).unsqueeze(0)), 'b c n -> (b n) c')[..., :3]
        # project to image plane
        # bx3x3 @ b*Nx3 -> bxNx3
        projected_points = einops.rearrange(torch.bmm(self.base_camera_intrinsic, (positions_in_cam.T).unsqueeze(0)), 'b c n -> (b n) c')
        projected_points = projected_points[..., :2] / projected_points[..., 2:]
        # b*Nx2
        projected_points = einops.rearrange(projected_points, '(b n) c -> b n c', b=b, n=N)

        # filter out points outside of image plane
        projected_points = projected_points.int()
    
        return projected_points
            
    def project_contact_positions_to_camera(self, contact_positions):
        # TODO extend to multiple envs
        # contact_positions: bxNx3
        # filter out nan rows
        with torch.device(self.device):
            b, N, _ = contact_positions.shape
            contact_positions = contact_positions[~torch.any(torch.isnan(contact_positions), dim=2)].reshape(b, -1, 3)
            b, N, _ = contact_positions.shape
            contact_map = torch.zeros((b, self.camera_height, self.camera_width, 1), device=contact_positions.device)
            # convert contact positions to camera frame
            if N > 0:
                projected_points = self.batched_position_to_pixel_coordinates(contact_positions)
                
                # swap u and v to match image coordinates
                projected_points = projected_points[..., [1, 0]]
                
                # filter out points outside of image plane
                valid_points = (projected_points[..., 0] >= 0) & (projected_points[..., 0] < self.camera_height) & (projected_points[..., 1] >= 0) & (projected_points[..., 1] < self.camera_width)
                projected_points = projected_points[valid_points]

                # add index for batch dimension
                projected_points = torch.cat([torch.zeros((projected_points.shape[0], 1), device=projected_points.device, dtype=torch.int), projected_points], dim=1)
                # index into contact_map and set valid points to 1
                contact_map[tuple(projected_points.T)] = 1
        return contact_map

    def get_extrinsic_contact_positions(self):
        with torch.device(self.device):
            assert self.num_envs == 1, "Only supports single envs for now"
            # TODO extend to multiple envs
            contact_positions = torch.nan*torch.ones((1, self.max_extrinsic_contacts, 3))
            contacts = self.scene.get_contacts()
            filtered_contacts = list()
            # filter contacts to only include contacts between grasped_book
            if len(contacts) > 0:
                for contact in contacts:
                    body_name_0 = contact.bodies[0].entity.name
                    body_name_1 = contact.bodies[1].entity.name
                    if 'grasped_book' in body_name_0 or 'grasped_book' in body_name_1:
                        # and not contact panda
                        if 'panda' not in body_name_0 and 'panda' not in body_name_1:
                            filtered_contacts.append(contact)
            contacts = filtered_contacts
            contact_idx = 0
            if len(contacts) > 0:
                for contact in contacts:
                    for contact_point in contact.points:
                        if np.linalg.norm(contact_point.impulse) > 0:
                            contact_positions[0, contact_idx] = torch.from_numpy(contact_point.position)
                            contact_idx += 1
                            # torch.from_numpy(contact_point.position)
                    # contact_positions.extend([torch.from_numpy(contact_point.position) for contact_point in contact.points])
            return contact_positions # bxNx3
    
    # compute boolean task stages for reward computation
    # first stage: reach to the book
    # def reach_to_book(self):

    # # save some commonly used attributes
    @property
    def time_between_env_steps(self):
        # time in seconds between each environment step
        return 1.0/self.sim_config.control_freq
    
    @property
    def top_of_slot_pose(self):
        with torch.device(self.device):
            # defined as centered in x-y of slot, and with z at the higher of the two neighboring books
            # bx7
            pos = torch.zeros((self.num_envs, 3))
            pos[:, 0] = self.xy_slot_location[:, 0]
            
            # pos[:, 1] = self.xy_slot_location[:, 1]
            # use the midpoint of the two books as the y position
            pos[:, 1] = (self.bottom_inner_corner_of_book_left_of_slot_pose.p[:, 1] + self.bottom_inner_corner_of_book_right_of_slot_pose.p[:, 1])/2
            
            height_of_left_book = self.env_book_sizes[:, self.slot_left_of_book_index-1, 2]
            height_of_right_book = self.env_book_sizes[:, self.slot_left_of_book_index, 2]
            pos[:, 2] = torch.maximum(height_of_left_book, height_of_right_book)
            
            # set orientation to be identity (to world frame)
            pose = Pose.create_from_pq(p=pos)
        return pose
    
    @property
    def bottom_inner_corner_of_book_right_of_slot_pose(self):
        with torch.device(self.device):
            bottom_inner_corner_in_book_frame = Pose.create_from_pq(p=torch.tensor([0, -self.env_book_sizes[:, self.slot_left_of_book_index, 1]/2, -self.env_book_sizes[:, self.slot_left_of_book_index, 2]/2]))
            # then also apply 180 intrinsic rotation around z-axis to get x to point in same direction as world
            quat_to_correct_orientation = axis_angle_to_quaternion(torch.tensor([0, 0, np.pi])).repeat(self.num_envs, 1)
        return self.env_books[self.slot_left_of_book_index].pose * bottom_inner_corner_in_book_frame * Pose.create_from_pq(q=quat_to_correct_orientation)
    
    @property
    def top_inner_corner_of_book_right_of_slot_pose(self):
        with torch.device(self.device):
            top_inner_corner_in_book_frame = Pose.create_from_pq(p=torch.tensor([0, -self.env_book_sizes[:, self.slot_left_of_book_index, 1]/2, self.env_book_sizes[:, self.slot_left_of_book_index, 2]/2]))
            # then also apply 180 intrinsic rotation around z-axis to get x to point in same direction as world
            quat_to_correct_orientation = axis_angle_to_quaternion(torch.tensor([0, 0, np.pi])).repeat(self.num_envs, 1)
        return self.env_books[self.slot_left_of_book_index].pose * top_inner_corner_in_book_frame * Pose.create_from_pq(q=quat_to_correct_orientation)

    @property
    def bottom_inner_corner_of_book_left_of_slot_pose(self):
        with torch.device(self.device):
            bottom_inner_corner_in_book_frame = Pose.create_from_pq(p=torch.tensor([0, self.env_book_sizes[:, self.slot_left_of_book_index-1, 1]/2, -self.env_book_sizes[:, self.slot_left_of_book_index-1, 2]/2]))
            # then also apply 180 intrinsic rotation around z-axis to get x to point in same direction as world
            quat_to_correct_orientation = axis_angle_to_quaternion(torch.tensor([0, 0, np.pi])).repeat(self.num_envs, 1)
        return self.env_books[self.slot_left_of_book_index-1].pose * bottom_inner_corner_in_book_frame * Pose.create_from_pq(q=quat_to_correct_orientation)
    
    @property
    def top_inner_corner_of_book_left_of_slot_pose(self):
        with torch.device(self.device):
            top_inner_corner_in_book_frame = Pose.create_from_pq(p=torch.tensor([0, self.env_book_sizes[:, self.slot_left_of_book_index-1, 1]/2, self.env_book_sizes[:, self.slot_left_of_book_index-1, 2]/2]))
            # then also apply 180 intrinsic rotation around z-axis to get x to point in same direction as world
            quat_to_correct_orientation = axis_angle_to_quaternion(torch.tensor([0, 0, np.pi])).repeat(self.num_envs, 1)
        return self.env_books[self.slot_left_of_book_index-1].pose * top_inner_corner_in_book_frame * Pose.create_from_pq(q=quat_to_correct_orientation)
    
    @property
    def bottom_of_grasped_book_pose(self):
        # recall grasped book is rotated "upside down" as it is initialized with the gripper pose which has z-down
        with torch.device(self.device):
            # bx7
            # apply 180 intrinsic rotation around x-axis
            quat_to_correct_orientation  = axis_angle_to_quaternion(torch.tensor([np.pi, 0, 0])).repeat(self.num_envs, 1)
            # then also apply 180 intrinsic rotation around z-axis to get x to point in same direction as world
            quat_to_correct_orientation = quaternion_multiply(quat_to_correct_orientation, axis_angle_to_quaternion(torch.tensor([0, 0, np.pi])))
            offset_to_bottom_of_book = Pose.create_from_pq(p=torch.tensor([0, 0, self.grasped_book_sizes[:, 2]/2]))
        return self.grasped_book.pose * offset_to_bottom_of_book * Pose.create_from_pq(q=quat_to_correct_orientation)
    
    @property
    def top_of_grasped_book_pose(self):
        # recall grasped book is rotated "upside down" as it is initialized with the gripper pose which has z-down. Also x points away from binding towards pages
        with torch.device(self.device):
            # bx7
            # apply 180 intrinsic rotation around x-axis
            quat_to_correct_orientation  = axis_angle_to_quaternion(torch.tensor([np.pi, 0, 0])).repeat(self.num_envs, 1)
            # then also apply 180 intrinsic rotation around z-axis to get x to point in same direction as world
            quat_to_correct_orientation = quaternion_multiply(quat_to_correct_orientation, axis_angle_to_quaternion(torch.tensor([0, 0, np.pi])))
            offset_to_top_of_book = Pose.create_from_pq(p=torch.tensor([0, 0, -self.grasped_book_sizes[:, 2]/2]))
        return self.grasped_book.pose * offset_to_top_of_book * Pose.create_from_pq(q=quat_to_correct_orientation)
    
    @property
    def distance_between_bottom_of_grasped_book_and_top_of_slot(self):
        with torch.device(self.device):
            # b
            top_of_slot_pos = self.top_of_slot_pose.p
            bottom_of_grasped_book_pos = self.bottom_of_grasped_book_pose.p
            distance = torch.linalg.norm(top_of_slot_pos - bottom_of_grasped_book_pos, axis=1, ord=2)
        return distance
    
    @property
    def z_distance_between_top_of_grasped_book_and_top_of_slot(self):
        with torch.device(self.device):
            # b
            top_of_slot_pos = self.top_of_slot_pose.p
            top_of_grasped_book_pos = self.top_of_grasped_book_pose.p
            z_distance = top_of_grasped_book_pos[:, 2] - top_of_slot_pos[:, 2]
            # distance = torch.linalg.norm(top_of_slot_pos - top_of_grasped_book_pos, axis=1, ord=2)
        return z_distance

    @property    
    def grasped_book_is_grasped(self):
        # check whether grasped_book is still in gripper
        return self.agent.is_grasping(self.grasped_book)
    
    @property
    def grasped_book_pushing_book_right_of_slot(self):
        with torch.device(self.device):
            assert self.num_envs == 1, "Only supports single envs for now"
            contact_forces = self.scene.get_pairwise_contact_forces(
                self.grasped_book, self.env_books[self.slot_left_of_book_index]
            )        
            force = torch.linalg.norm(contact_forces, axis=1)
            # make sure force is pointing rightward (pushing in positive y direction)
            right_direction = torch.tensor([0., 1, 0]).repeat(self.num_envs, 1)
            # dot product should be positive
            dot_product = torch.sum(contact_forces * right_direction, dim=1)
            pushing = torch.logical_and(force > 0, dot_product > 0)
        return pushing
    
    @property
    def grasped_book_pushing_book_left_of_slot(self):
        with torch.device(self.device):
            assert self.num_envs == 1, "Only supports single envs for now"
            contact_forces = self.scene.get_pairwise_contact_forces(
                self.grasped_book, self.env_books[self.slot_left_of_book_index-1]
            )        
            force = torch.linalg.norm(contact_forces, axis=1)
            # make sure force is pointing leftward (pushing in negative y direction)
            left_direction = torch.tensor([0., -1, 0]).repeat(self.num_envs, 1)
            # dot product should be positive
            dot_product = torch.sum(contact_forces * left_direction, dim=1)
            pushing = torch.logical_and(force > 0, dot_product > 0)
        return pushing

    def angle_of_pose_with_vertical(self, pose: Pose):
        with torch.device(self.device):
            # get the direction of the actor
            actor_direction = quaternion_apply(pose.q, torch.tensor([0, 0, 1.]).repeat(self.num_envs, 1))
            # get the vertical direction
            vertical_direction = torch.tensor([0, 0, 1.]).repeat(self.num_envs, 1)
            angle_with_vertical = common.compute_angle_between(actor_direction, vertical_direction)
        return angle_with_vertical
    
    @property
    def any_env_books_toppled(self):
        with torch.device(self.device):
            # check if any env books have toppled
            # check if the angle of the books with the vertical is greater than 45 degrees
            # create a bxN tensor of angles
            angles = torch.zeros((self.num_envs, self.num_env_books))
            for j in range(self.num_env_books):
                angles[:, j] = self.angle_of_pose_with_vertical(self.env_books[j].pose)
            # check if any of the angles are greater than 45 degrees
            toppled = torch.any(angles > self.book_toppled_angle_with_vertical_threshold, dim=1)
        return toppled
    
    @property
    def bottom_of_grasped_book_within_slot(self):
        # just evaluates whether y is between the two books
        with torch.device(self.device):
            bottom_of_grasped_book_y = self.bottom_of_grasped_book_pose.p[:, 1]
            left_book_y = self.bottom_inner_corner_of_book_left_of_slot_pose.p[:, 1]
            right_book_y = self.bottom_inner_corner_of_book_right_of_slot_pose.p[:, 1]
            within = torch.logical_and(bottom_of_grasped_book_y > left_book_y, bottom_of_grasped_book_y < right_book_y)
        return within
    
    @property
    def top_of_grasped_book_within_slot(self):
        # just evaluates whether y is between the two books
        with torch.device(self.device):
            top_of_grasped_book_y = self.top_of_grasped_book_pose.p[:, 1]
            left_book_y = self.top_inner_corner_of_book_left_of_slot_pose.p[:, 1]
            right_book_y = self.top_inner_corner_of_book_right_of_slot_pose.p[:, 1]
            within = torch.logical_and(top_of_grasped_book_y > left_book_y, top_of_grasped_book_y < right_book_y)
        return within
    
    def evaluate(self):
        # to succeed: 
        # no env books should be toppled 
        # & the grasped book top and bottom must be within the slot in x-y
        # & top of grasped book must be close to the top of the slot
        # & above conditions must have held for 
        not_toppled = ~self.any_env_books_toppled
        bottom_within_slot = self.bottom_of_grasped_book_within_slot
        top_within_slot = self.top_of_grasped_book_within_slot
        z_distance_bw_top_of_grasped_book_and_top_of_slot = self.z_distance_between_top_of_grasped_book_and_top_of_slot
        close_to_top_of_slot = z_distance_bw_top_of_grasped_book_and_top_of_slot < self.top_of_grasped_book_distance_to_top_of_slot_threshold
        transient_success = torch.logical_and(
            not_toppled, torch.logical_and(
                bottom_within_slot, torch.logical_and(
                    top_within_slot, 
                        close_to_top_of_slot)))
        success_in_a_row = torch.logical_and(transient_success, self.last_eval_bool)
        self.elapsed_success_duration += success_in_a_row.float() * self.time_between_env_steps
        self.elapsed_success_duration *= transient_success.float() # reset to 0 if not transient success
        success = self.elapsed_success_duration > self.success_duration_threshold
        self.last_eval_bool = transient_success
        return dict(
            success=success,
            transient_success=transient_success, 
            elapsed_success_duration=self.elapsed_success_duration,
            z_distance_bw_top_of_grasped_book_and_top_of_slot=z_distance_bw_top_of_grasped_book_and_top_of_slot,
            not_toppled=not_toppled,
            top_within_slot=top_within_slot,
            bottom_within_slot=bottom_within_slot,
            )

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
