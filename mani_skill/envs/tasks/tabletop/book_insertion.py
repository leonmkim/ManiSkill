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
from mani_skill.utils.structs import Actor, Pose, Articulation
from mani_skill.utils.structs.types import SimConfig
from mani_skill.utils.geometry.rotation_conversions import matrix_to_quaternion

from mani_skill.utils.geometry.rotation_conversions import quaternion_multiply, axis_angle_to_quaternion, quaternion_apply
from mani_skill.utils.geometry.geometry import transform_points

import einops
import trimesh as tm
from scipy.spatial.transform import Rotation as R

import logging

from pathlib import Path
import sys, os
# add contact_estimation to the path
path_to_this_file = Path(os.path.abspath(__file__))
path_to_contact_estimation = path_to_this_file.parents[5] / "contact_estimation"
sys.path.append(str(path_to_contact_estimation))
from src.dataset.gazebo_to_trimesh import create_trimesh_camera, generate_rays_from_camera, generate_min_distances_image, normals_to_xyz_map, get_min_grasped_obj_sdf_at_env_hits_data, get_min_env_sdf_at_grasped_obj_hits_data, camera_marker_transformed

# logging.basicConfig(level=logging.DEBUG)

book_insertion_env_logger = logging.getLogger("book_insertion_env_logger")

# book_insertion_env_logger.setLevel(logging.INFO)

def get_book_primitive_mesh_list(length, width, height, binding_thickness, cover_thickness, cover_overhang, global_transform=None):
    pages_length = length - cover_overhang - binding_thickness
    pages_width = width - 2*cover_thickness
    pages_height = height - 2*cover_overhang
    full_sizes = [
        [pages_length, pages_width, pages_height], # pages
        [binding_thickness*2, width, height], # binding
        [length, cover_thickness, height], # cover
        [length, cover_thickness, height], # cover
    ]
    poses = [
        sapien.Pose([(binding_thickness - cover_overhang)/2, 0, 0]).to_transformation_matrix(), # pages
        sapien.Pose([(binding_thickness - length)/2, 0, 0]).to_transformation_matrix(), # binding
        sapien.Pose([0, (pages_width + cover_thickness)/2, 0]).to_transformation_matrix(), # cover
        sapien.Pose([0, -(pages_width + cover_thickness)/2, 0]).to_transformation_matrix(), # cover
    ]
    book_geometries = []
    for i, (full_size, pose) in enumerate(zip(full_sizes, poses)):
        # builder.add_box_collision(pose, half_size, density=density)
        object_geometry = tm.primitives.Box(extents=full_size)
        object_geometry.apply_transform(pose)
        if global_transform is not None:
            object_geometry.apply_transform(global_transform)
        book_geometries.append(object_geometry)

    return book_geometries

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

def get_table_primitive_mesh_list(length, width, height, global_transform=None):
    table_box_offset_pose = np.eye(4)
    table_box_offset_pose[2, 3] = height/2
    table_mesh = tm.primitives.Box(extents=[length, width, height], transform=table_box_offset_pose)
    if global_transform is not None:
        table_mesh.apply_transform(global_transform)
    return [table_mesh]

def get_env_object_meshes_list(env_book_sizes_list, env_object_transforms_list, binding_thickness, cover_thickness, cover_overhang):
    env_object_meshes_list = []
    # for i, env_book_over_envs in enumerate(env.non_merged_env_books_list):
    for (env_book_sizes, env_object_transform) in zip(env_book_sizes_list, env_object_transforms_list):
        # env_object_mesh = env_book_over_envs[0].get_collision_meshes()
        # env_object_meshes_list.extend(env_object_mesh)
        length, width, height = env_book_sizes
        env_object_mesh = get_book_primitive_mesh_list(length, width, height, binding_thickness, cover_thickness, cover_overhang, global_transform=env_object_transform)
        env_object_meshes_list.extend(env_object_mesh)
    return env_object_meshes_list

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
    
def _build_book_end(
    scene: ManiSkillScene, 
    length, width, height,
    mode='static',
    color="#FFD289", 
    mass=0.5,
    friction=0.3, # default for both static and dynamic is 0.3
    wall_length=None, wall_width=None, wall_height=None,
    travel_limit=None,
    book_end_wrt_wall=None, # +y or -y
):
    # book end is just a box
    if isinstance(color, str):
        color = sapien_utils.hex2rgba(color)
    if mode in ['static', 'dynamic']:
        builder = scene.create_actor_builder()
        half_size = [length/2, width/2, height/2]
        pose = sapien.Pose([0, 0, 0])

        # compute the density from the mass and volume
        density = mass / (length * width * height)
        phys_mat = sapien.physx.PhysxMaterial(
            static_friction=friction, dynamic_friction=friction, restitution=0.0
        )
        builder.add_box_collision(pose, half_size, density=density, material=phys_mat)

        viz_mat = sapien.render.RenderMaterial(
            base_color=color, roughness=0.5, specular=0.5
        )
        builder.add_box_visual(pose, half_size, material=viz_mat)
    elif mode == 'spring':
        assert wall_length is not None, "wall_length must be specified for spring book end"
        assert wall_width is not None, "wall_width must be specified for spring book end"
        assert wall_height is not None, "wall_height must be specified for spring book end"
        assert travel_limit is not None, "travel_limit must be specified for spring book end"
        assert book_end_wrt_wall in ['+y', '-y'], "book_end_wrt_wall must be either '+y' or '-y'"

        builder = scene.create_articulation_builder()
        wall = builder.create_link_builder()
        wall.set_name("wall")

        wall_origin_pose = sapien.Pose([0, 0, +wall_height/2 - height/2])# center height to the book_end origin
        wall.add_box_collision(
            pose=wall_origin_pose, 
            half_size=[wall_length/2, wall_width/2, wall_height/2],
            density=mass/(wall_length*wall_width*wall_height),
        )
        wall.add_box_visual(
            pose=wall_origin_pose,
            half_size=[wall_length/2, wall_width/2, wall_height/2],
            material=sapien.render.RenderMaterial(
                base_color=color, roughness=0.5, specular=0.5
            ),
        )

        book_end = builder.create_link_builder(wall)
        book_end.set_name("book_end")
        book_end_phys_material = sapien.physx.PhysxMaterial(
            static_friction=friction, dynamic_friction=friction, restitution=0.0
        )
        book_end.add_box_collision(
            half_size=[length/2, width/2, height/2],
            density=mass/(length*width*height),
            material=book_end_phys_material,
        )
        book_end.add_box_visual(
            half_size=[length/2, width/2, height/2],
            material=sapien.render.RenderMaterial(
                base_color=color, roughness=0.5, specular=0.5
            ),
        )

        book_end.set_joint_name("book_end_joint")
        if book_end_wrt_wall == '+y':
            pose_in_parent = sapien.Pose(p=[0, wall_width/2 + travel_limit, 0], q=axis_angle_to_quaternion(torch.tensor([0, 0, np.pi/2.]))) # Parent_T_Joint
            pose_in_child = sapien.Pose(p=[0, -(width/2), 0], q=axis_angle_to_quaternion(torch.tensor([0, 0, np.pi/2.]))) # Child_T_Joint
        elif book_end_wrt_wall == '-y':
            pose_in_parent = sapien.Pose(p=[0, -(wall_width/2+ travel_limit), 0], q=axis_angle_to_quaternion(torch.tensor([0, 0, -np.pi/2.])))
            pose_in_child = sapien.Pose(p=[0, (width/2), 0], q=axis_angle_to_quaternion(torch.tensor([0, 0, -np.pi/2.]))) # Child_T_Joint
        else:
            raise ValueError("book_end_wrt_wall must be either '+y' or '-y'")
        
        book_end.set_joint_properties(
            type='prismatic',
            # limits=[[0, travel_limit]],
            limits=[[-travel_limit, 0]],
            pose_in_parent=pose_in_parent, # Parent_T_Joint
            pose_in_child=pose_in_child, # Child_T_Joint
            friction=0.0,
            damping=0.0,
        )

    return builder
    
from dataclasses import dataclass, field

@dataclass
class EnvBooksConfig:
    """
    Configuration for the env books in the BookInsertionEnv.
    """
    num_env_books: int = 8
    randomize_color: bool = True
    randomize_density: bool = True
    density_randomization_bounds: list = field(default_factory=lambda: [655, 1015])
    randomize_height: bool = True
    height_randomization_bounds: list = field(default_factory=lambda: [0.2475, 0.2525])
    randomize_width: bool = True
    width_randomization_bounds: list = field(default_factory=lambda: [0.015, 0.05])
    randomize_length: bool = True
    length_randomization_bounds: list = field(default_factory=lambda: [0.15, 0.2])
    shuffle_mode: str = 'none'

    def __post_init__(self):
        any_randomize = any([
            self.randomize_color,
            self.randomize_density,
            self.randomize_height,
            self.randomize_width,
            self.randomize_length,
        ])
        assert not (self.shuffle_mode != 'none' and any_randomize), "Cannot shuffle env books and randomize at the same time"
        assert self.shuffle_mode in ['none', 'left', 'right', 'all'], f"shuffle_env_books_mode must be one of ['none', 'left', 'right', 'all'], but got {self.shuffle_mode}"
        assert self.density_randomization_bounds[1] > self.density_randomization_bounds[0], f"density_randomization_bounds must be in the form [min, max], but got {self.density_randomization_bounds}"
        assert self.height_randomization_bounds[1] > self.height_randomization_bounds[0], f"height_randomization_bounds must be in the form [min, max], but got {self.height_randomization_bounds}"
        assert self.width_randomization_bounds[1] > self.width_randomization_bounds[0], f"width_randomization_bounds must be in the form [min, max], but got {self.width_randomization_bounds}"
        assert self.length_randomization_bounds[1] > self.length_randomization_bounds[0], f"length_randomization_bounds must be in the form [min, max], but got {self.length_randomization_bounds}"
        assert self.num_env_books > 0, f"num_env_books must be greater than 0, but got {self.num_env_books}"

@dataclass
class RobotConfig:
    """
    Configuration for the robot in the BookInsertionEnv.
    """
    init_qpos: list = field(default_factory=lambda: [0.022516679397616424, 0.11646689505116431, -0.3625673227601117, -1.37265637618617, 0.033468631741809286, 1.4658307538809252, 0.46052758571920294,.04,.04,])
    # additive_y_randomization_bounds: Union[float, list] = 0.0

@dataclass
class SlotConfig:
    """
    Configuration for the slot in the BookInsertionEnv.
    """
    negative_tolerance: float = 0.0035
    left_of_book_index: int = 4
    # y_randomization_bounds: Union[float, list] = 0.0
    # change type to any to make omegaconf accept it: https://github.com/omry/omegaconf/issues/144
    y_randomization_bounds: Any = 0.0

@dataclass
class GraspedBookConfig:
    """
    Configuration for the grasped book in the BookInsertionEnv.
    """
    randomize_color: bool = True
    randomize_density: bool = True
    density_randomization_bounds: list = field(default_factory=lambda: [650, 850])
    randomize_height: bool = True
    height_randomization_bounds: list = field(default_factory=lambda: [0.165, 0.25])
    randomize_width: bool = True
    width_randomization_bounds: list = field(default_factory=lambda: [0.03, 0.065])
    randomize_length: bool = True
    length_randomization_bounds: list = field(default_factory=lambda: [0.1, 0.15])

    def __post_init__(self):
        assert self.density_randomization_bounds[1] > self.density_randomization_bounds[0], f"density_randomization_bounds must be in the form [min, max], but got {self.density_randomization_bounds}"
        assert self.height_randomization_bounds[1] > self.height_randomization_bounds[0], f"height_randomization_bounds must be in the form [min, max], but got {self.height_randomization_bounds}"
        assert self.width_randomization_bounds[1] > self.width_randomization_bounds[0], f"width_randomization_bounds must be in the form [min, max], but got {self.width_randomization_bounds}"
        assert self.length_randomization_bounds[1] > self.length_randomization_bounds[0], f"length_randomization_bounds must be in the form [min, max], but got {self.length_randomization_bounds}"

from typing import Optional
@dataclass
class BookEndsConfig:
    """
    Configuration for the book ends in the BookInsertionEnv.
    """
    mode: str = 'none'
    length: float = 0.2
    width: float = 0.025
    height: float = 0.3
    mass: float = 0.5
    color: str = "#808080" # default color
    friction: float = 0.3 # default friction for sapien objects is 0.3

    wall_height: float = 0.3
    wall_length: float = 0.2
    wall_width: float = 0.025
    travel_limit: float = 0.05
    joint_stiffness: float = 100
    joint_damping: float = 20

    def __post_init__(self):
        assert self.mode in ['none', 'static', 'dynamic', 'spring'], f"book_ends_mode must be one of ['none', 'static', 'dynamic', 'spring'], but got {self.mode}"


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
    

    binding_thickness: float = 0.005
    cover_thickness: float = 0.003
    cover_overhang: float = 0.005

    cam_resize_factor: float = 0.5

    render_contact_map: bool = False
    render_dtc_maps: bool = False
    render_normals_maps: bool = False

    max_extrinsic_contacts: int = 50 # for padding

    # success conditions
    success_criteria_params: Dict[str, Any] = dict(
        book_toppled_angle_with_vertical_threshold=np.deg2rad(45),
        top_of_grasped_book_distance_to_top_of_slot_threshold=0.047 + .02,
        success_duration_threshold=3.0, # seconds        
    )

    # num_env_books: int = 8
    # spawn_new_env_books: bool = True
    # shuffle_env_books_mode: str = 'none'
    env_books_config: EnvBooksConfig = EnvBooksConfig()
    # Dict[str, Any] = dict(
    #     num_env_books=8,
    #     randomize_color=True,
    #     randomize_density=True,
    #     density_randomization_bounds=[655, 1015],
    #     randomize_height=True,
    #     height_randomization_bounds=[0.2475, 0.2525],
    #     randomize_width=True,
    #     width_randomization_bounds=[0.015, 0.05],
    #     randomize_length=True,
    #     length_randomization_bounds=[0.15, 0.2],
    #     shuffle_mode='none',
    # )

    robot_config: RobotConfig = RobotConfig()

    # slot_left_of_book_index: int = 4
    slot_config: SlotConfig = SlotConfig()
    # Dict[str, Any] = dict(
    #     negative_tolerance=0.0035,
    #     left_of_book_index=4,
    #     y_randomization_bounds=0,
    # )

    grasped_book_config: GraspedBookConfig = GraspedBookConfig()

    # Dict[str, Any] = dict(
    #     randomize_color=True,
    #     randomize_density=True,
    #     density_randomization_bounds=[650, 850],
    #     randomize_height=True,
    #     height_randomization_bounds=[0.165, 0.25],
    #     randomize_width=True,
    #     width_randomization_bounds=[0.03, 0.065],
    #     randomize_length=True,
    #     length_randomization_bounds=[0.1, 0.15],
    # )
    # spawn_new_grasped_book: bool = True

    book_ends_config: BookEndsConfig = BookEndsConfig()

    # Dict[str, Any] = dict(
    #     mode='none',
    #     height=0.25,
    #     mass=0.5,
    #     color="#FFD289", # default color
    #     friction=0.3, # default friction for sapien objects is 0.3
    # )

    suppress_evaluation: bool = False
    
    # book_toppled_angle_with_vertical_threshold: float = np.deg2rad(45)
    # # from base of gripper fingers to tip of fingers is .047m
    # top_of_grasped_book_distance_to_top_of_slot_threshold: float = 0.047 + .02
    # success_duration_threshold: float = 3.0 # seconds

    def __init__(
        self,
        *args,
        robot_uids="panda",
        num_envs=1,
        reconfiguration_freq=None,
        **kwargs,
    ):
        self.new_env_books_are_spawned = False
        self.new_grasped_book_is_spawned = False

        # self.times_spawned_new_env_books = 0

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

        # print(kwargs)
        # add custom kwargs to the env
        for key in kwargs:
            # if key in self.__dict__:
            if key in BookInsertionEnv.__dict__:
                setattr(self, key, kwargs[key])
                # del kwargs[key]

        self.grasped_book_colors = None
        self.grasped_book_densities = None
        self.grasped_book_heights = None
        self.grasped_book_widths = None
        self.grasped_book_lengths = None

        self.env_book_colors = None
        self.env_book_densities = None
        self.env_book_heights = None
        self.env_book_widths = None
        self.env_book_lengths = None

        super().__init__(
            *args,
            robot_uids=robot_uids,
            num_envs=num_envs,
            reconfiguration_freq=reconfiguration_freq,
            **kwargs,
        )

        # print(f"BookInsertionEnv: {self.__dict__}")

    @property
    def _default_sim_config(self):
        return SimConfig()

    @property
    def _default_sensor_configs(self):
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
            
            # self.camera_pose = build_coordinate_frame(self.scene, axis_length=0.05, axis_radius=0.005, name="camera_pose", body_type="kinematic")
            # self._hidden_objects.append(self.camera_pose)
            # <<<<<<<<< for debugging

            # if self.spawn_new_grasped_book or (not self.spawn_new_grasped_book and not self.new_grasped_book_is_spawned):
            # if isinstance(self.grasped_book_dict['length_randomization_bounds'], list):
            if self.grasped_book_config.randomize_length or self.grasped_book_lengths is None:
                # grasped_book_lengths = self._batched_episode_rng.uniform(0.1, 0.15)
                self.grasped_book_lengths = self._batched_episode_rng.uniform(self.grasped_book_config.length_randomization_bounds[0], self.grasped_book_config.length_randomization_bounds[1])
            # else:
            #     grasped_book_lengths = torch.ones((self.num_envs,)) * self.grasped_book_dict['length_randomization_bounds']

            # if isinstance(self.grasped_book_dict['width_randomization_bounds'], list):
            if self.grasped_book_config.randomize_width or self.grasped_book_widths is None:
                # grasped_book_widths = self._batched_episode_rng.uniform(0.03, 0.065) # max gripper width is .08
                self.grasped_book_widths = self._batched_episode_rng.uniform(self.grasped_book_config.width_randomization_bounds[0], self.grasped_book_config.width_randomization_bounds[1])
            # else:
            #     grasped_book_widths = torch.ones((self.num_envs,)) * self.grasped_book_dict['width_randomization_bounds']
            
            # if isinstance(self.grasped_book_dict['height_randomization_bounds'], list):
            if self.grasped_book_config.randomize_height or self.grasped_book_heights is None:
                # grasped_book_heights = self._batched_episode_rng.uniform(0.165, 0.25)
                self.grasped_book_heights = self._batched_episode_rng.uniform(self.grasped_book_config.height_randomization_bounds[0], self.grasped_book_config.height_randomization_bounds[1])
            # else:
            #     grasped_book_heights = torch.ones((self.num_envs,)) * self.grasped_book_dict['height_randomization_bounds']
                
            # # save some useful values for use later
            self.grasped_book_sizes = common.to_tensor(np.vstack([self.grasped_book_lengths, self.grasped_book_widths, self.grasped_book_heights])).T

            # if isinstance(self.grasped_book_dict['randomize_density'], list):
            if self.grasped_book_config.randomize_density or self.grasped_book_densities is None:
                # self.grasped_book_densities = self._batched_episode_rng.uniform(650, 850)
                self.grasped_book_densities = self._batched_episode_rng.uniform(self.grasped_book_config.density_randomization_bounds[0], self.grasped_book_config.density_randomization_bounds[1])
            # else:
            #     self.grasped_book_densities = torch.ones((self.num_envs,)) * self.grasped_book_dict['randomize_density']
                
            if self.grasped_book_config.randomize_color or self.grasped_book_colors is None:
                self.grasped_book_colors = np.ones((self.num_envs, 4))
                self.grasped_book_colors[:,0] = self._batched_episode_rng.uniform(0.0, 1.0)
                self.grasped_book_colors[:,1] = self._batched_episode_rng.uniform(0.0, 1.0)
                self.grasped_book_colors[:,2] = self._batched_episode_rng.uniform(0.0, 1.0)

            grasped_books = []
            for i in range(self.num_envs):
                scene_idxs = [i]
                # grasped_book_length = grasped_book_lengths[i]
                # grasped_book_width = grasped_book_widths[i]
                # grasped_book_height = grasped_book_heights[i]
                grasped_book_length, grasped_book_width, grasped_book_height = self.grasped_book_sizes[i]

                builder = _build_book(
                    self.scene, 
                    grasped_book_length, grasped_book_width, grasped_book_height, 
                    self.binding_thickness, self.cover_thickness, self.cover_overhang,
                    book_color=self.grasped_book_colors[i],
                    density=self.grasped_book_densities[i],
                )
                builder.initial_pose = sapien.Pose(p=[0, 1, 0.3])
                builder.set_scene_idxs(scene_idxs)
                grasped_book = builder.build(f"grasped_book_{i}")
                self.remove_from_state_dict_registry(grasped_book)
                grasped_books.append(grasped_book)
            
            self.non_merged_grasped_books_list = grasped_books
            self.grasped_book = Actor.merge(grasped_books, "grasped_book")
            self.add_to_state_dict_registry(self.grasped_book)

            self.new_grasped_book_is_spawned = True

            # if self.spawn_new_env_books or (not self.spawn_new_env_books and not self.new_env_books_are_spawned):
            if self.env_books_config.randomize_height or self.env_book_heights is None:
                env_book_heights = []
            if self.env_books_config.randomize_width or self.env_book_widths is None:
                env_book_widths = []
            if self.env_books_config.randomize_length or self.env_book_lengths is None:
                env_book_lengths = []
            if self.env_books_config.randomize_density or self.env_book_densities is None:
                env_book_densities = []
            if self.env_books_config.randomize_color or self.env_book_colors is None:
                env_book_colors = []
            
            for i in range(self.env_books_config.num_env_books):
                if self.env_books_config.randomize_length or self.env_book_lengths is None:
                    env_book_lengths.append(self._batched_episode_rng.uniform(self.env_books_config.length_randomization_bounds[0], self.env_books_config.length_randomization_bounds[1]))
                if self.env_books_config.randomize_width or self.env_book_widths is None:
                    env_book_widths.append(self._batched_episode_rng.uniform(self.env_books_config.width_randomization_bounds[0], self.env_books_config.width_randomization_bounds[1]))
                if self.env_books_config.randomize_height or self.env_book_heights is None:
                    env_book_heights.append(self._batched_episode_rng.uniform(self.env_books_config.height_randomization_bounds[0], self.env_books_config.height_randomization_bounds[1]))
                if self.env_books_config.randomize_density or self.env_book_densities is None:
                    env_book_densities.append(self._batched_episode_rng.uniform(self.env_books_config.density_randomization_bounds[0], self.env_books_config.density_randomization_bounds[1]))
                if self.env_books_config.randomize_color or self.env_book_colors is None:
                    color = np.ones((self.num_envs, 4))
                    color[:,0] = self._batched_episode_rng.uniform(0.0, 1.0)
                    color[:,1] = self._batched_episode_rng.uniform(0.0, 1.0)
                    color[:,2] = self._batched_episode_rng.uniform(0.0, 1.0)
                    env_book_colors.append(color)

            if self.env_books_config.randomize_height or self.env_book_heights is None:
                self.env_book_heights = np.vstack(env_book_heights).T # bxN
            if self.env_books_config.randomize_width or self.env_book_widths is None:
                self.env_book_widths = np.vstack(env_book_widths).T # bxN
            if self.env_books_config.randomize_length or self.env_book_lengths is None:
                self.env_book_lengths = np.vstack(env_book_lengths).T # bxN
            self.env_book_sizes = common.to_tensor(np.stack([self.env_book_lengths, self.env_book_widths, self.env_book_heights], axis=2))

            if self.env_books_config.randomize_density or self.env_book_densities is None:
                self.env_book_densities = np.vstack(env_book_densities).T
            
            if self.env_books_config.randomize_color or self.env_book_colors is None:
                # construct bxNx3 tensor
                self.env_book_colors = np.stack(env_book_colors,axis=1) # bxNx4
                assert self.env_book_colors.shape == (self.num_envs, self.env_books_config.num_env_books, 4), f"env_book_colors shape is incorrect, {self.env_book_colors.shape}"

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

            # if self.spawn_new_env_books or (not self.spawn_new_env_books and not self.new_env_books_are_spawned) or (not self.spawn_new_env_books and self.shuffle_env_books):
            # if self.shuffle_env_books_mode != 'none':
            if self.env_books_config.shuffle_mode != 'none':
                # apply the same shuffling to env_book_sizes, env_book_colors, and env_book_densities
                indices = np.arange(self.env_books_config.num_env_books)
                if self.env_books_config.shuffle_mode == 'all':
                    self._batched_episode_rng.shuffle(indices)
                elif self.env_books_config.shuffle_mode == 'left':
                    # shuffle the first half of the env books
                    left_indices = indices[:self.env_books_config.num_env_books//2]
                    self._batched_episode_rng.shuffle(left_indices)
                    indices[:self.env_books_config.num_env_books//2] = left_indices
                elif self.env_books_config.shuffle_mode == 'right':
                    # shuffle the second half of the env books
                    right_indices = indices[self.env_books_config.num_env_books//2:]
                    self._batched_episode_rng.shuffle(right_indices)
                    indices[self.env_books_config.num_env_books//2:] = right_indices

                self.env_book_sizes = self.env_book_sizes[:, indices, :]
                self.env_book_colors = self.env_book_colors[:, indices, :]
                self.env_book_densities = self.env_book_densities[:, indices]

            env_books_list = []
            for j in range(self.env_books_config.num_env_books):
                envs_per_env_book = []
                for i in range(self.num_envs):
                    scene_idxs = [i]
                    book_length = self.env_book_sizes[i, j, 0]
                    book_width = self.env_book_sizes[i, j, 1]
                    book_height = self.env_book_sizes[i, j, 2]
                    builder = _build_book(
                        self.scene, 
                        book_length, book_width, book_height, 
                        self.binding_thickness, self.cover_thickness, self.cover_overhang,
                        book_color=self.env_book_colors[i, j],
                        density=self.env_book_densities[i, j],
                    )
                    builder.initial_pose = sapien.Pose(p=[0, -1, 0.5*(j+1)])
                    builder.set_scene_idxs(scene_idxs)
                    env_book = builder.build(f"book_{j}_{i}")
                    self.remove_from_state_dict_registry(env_book)
                    envs_per_env_book.append(env_book)

                env_books_list.append(envs_per_env_book)

            # env_book_collision_indices = [0]
            # for j in range(self.env_books_config.num_env_books-2):
            #     env_book_collision_indices.append(env_book_collision_indices[j]+(j+2))


            # env_book_collision_group = 0
            # for idx in env_book_collision_indices:
            #     env_book_collision_group |= 1 << idx

            # want to make Nxb env books
            # create a copy of the env books list to keep track of the non-merged env books
            self.non_merged_env_books_list = []
            for j in range(self.env_books_config.num_env_books):
                envs_per_env_book = env_books_list[j]
                self.non_merged_env_books_list.append(envs_per_env_book)
                env_books_list[j] = Actor.merge(envs_per_env_book, f"book_{j}")
                self.add_to_state_dict_registry(env_books_list[j])
            #     env_books[j].set_collision_group(group=2, value=env_book_collision_group)
            #     env_book_collision_group = env_book_collision_group << 1

            self.env_books_list = env_books_list

            self.new_env_books_are_spawned = True

            if self.book_ends_config.mode != 'none':
                left_book_ends = []
                right_book_ends = []
                self.book_end_sizes = [self.book_ends_config.length, self.book_ends_config.width, self.book_ends_config.height]
                for i in range(self.num_envs):
                    scene_idxs = [i]
                    left_book_end_builder = _build_book_end(
                        self.scene,
                        mode=self.book_ends_config.mode,
                        length=self.book_end_sizes[0],
                        width=self.book_end_sizes[1],
                        height=self.book_end_sizes[2],
                        color=self.book_ends_config.color,
                        mass=self.book_ends_config.mass,
                        friction=self.book_ends_config.friction,
                        wall_height=self.book_ends_config.wall_height,
                        wall_length=self.book_ends_config.wall_length,
                        wall_width=self.book_ends_config.wall_width,
                        travel_limit=self.book_ends_config.travel_limit,
                        book_end_wrt_wall='-y',
                    )
                    left_book_end_builder.set_initial_pose(sapien.Pose(p=[0, -1, 2]))
                    left_book_end_builder.set_scene_idxs(scene_idxs)

                    right_book_end_builder = _build_book_end(
                        self.scene,
                        mode=self.book_ends_config.mode,
                        length=self.book_end_sizes[0],
                        width=self.book_end_sizes[1],
                        height=self.book_end_sizes[2],
                        color=self.book_ends_config.color,
                        mass=self.book_ends_config.mass,
                        friction=self.book_ends_config.friction,
                        wall_height=self.book_ends_config.wall_height,
                        wall_length=self.book_ends_config.wall_length,
                        wall_width=self.book_ends_config.wall_width,
                        travel_limit=self.book_ends_config.travel_limit,
                        book_end_wrt_wall='+y',
                    )
                    right_book_end_builder.set_initial_pose(sapien.Pose(p=[0, -1, 3]))
                    right_book_end_builder.set_scene_idxs(scene_idxs)
                    if self.book_ends_config.mode == 'static':
                        left_book_end = left_book_end_builder.build_static(f"left_book_end_{i}")
                        right_book_end = right_book_end_builder.build_static(f"right_book_end_{i}")
                    elif self.book_ends_config.mode == 'dynamic':
                        left_book_end = left_book_end_builder.build_dynamic(f"left_book_end_{i}")
                        right_book_end = right_book_end_builder.build_dynamic(f"right_book_end_{i}")
                    elif self.book_ends_config.mode == 'spring':
                        left_book_end = left_book_end_builder.build(f"left_book_end_{i}", fix_root_link=True)
                        left_book_end.find_joint_by_name("book_end_joint").set_drive_properties(
                            stiffness=self.book_ends_config.joint_stiffness,
                            damping=self.book_ends_config.joint_damping,
                        )
                        # left_book_end.find_joint_by_name("book_end_joint").set_drive_target(self.book_ends_config.travel_limit)
                        left_book_end.find_joint_by_name("book_end_joint").set_drive_target(0)
                        
                        right_book_end = right_book_end_builder.build(f"right_book_end_{i}", fix_root_link=True)
                        right_book_end.find_joint_by_name("book_end_joint").set_drive_properties(
                            stiffness=self.book_ends_config.joint_stiffness,
                            damping=self.book_ends_config.joint_damping,
                        )
                        # right_book_end.find_joint_by_name("book_end_joint").set_drive_target(self.book_ends_config.travel_limit)
                        right_book_end.find_joint_by_name("book_end_joint").set_drive_target(0)

                    else:
                        raise ValueError(f"Invalid book ends mode {self.book_ends_config.mode}")
                    self.remove_from_state_dict_registry(left_book_end)
                    self.remove_from_state_dict_registry(right_book_end)
                    left_book_ends.append(left_book_end)
                    right_book_ends.append(right_book_end)
                # merge the book ends
                if self.book_ends_config.mode == 'spring':
                    self.left_book_end = Articulation.merge(left_book_ends, "left_book_end")
                    self.right_book_end = Articulation.merge(right_book_ends, "right_book_end")
                else:
                    self.left_book_end = Actor.merge(left_book_ends, "left_book_end")
                    self.right_book_end = Actor.merge(right_book_ends, "right_book_end")
                self.add_to_state_dict_registry(self.left_book_end)
                self.add_to_state_dict_registry(self.right_book_end)

            self.target_EE_pose = build_coordinate_frame(self.scene, axis_length=0.05, axis_radius=0.005, name="target_EE_pose", body_type="kinematic")
            self._hidden_objects.append(self.target_EE_pose)
            
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # book_insertion_env_logger.debug(f"times spawned new env books: {self.times_spawned_new_env_books}")
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
            qpos[:, -2:] = (self.grasped_book_sizes[:, 1])/2 + .001
            self.agent.robot.set_qpos(qpos)
            
            # This is for the root pose
            # self.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))
            self.agent.robot.set_pose(sapien.Pose([0., 0, 0]))

            if self.backend.sim_backend == 'physx_cuda':
                # ensure all updates to object poses and configurations are applied on GPU after task initialization
                self.scene._gpu_apply_all()
                self.scene.px.gpu_update_articulation_kinematics()
                self.scene._gpu_fetch_all()

            end_effector_pose = self.agent.tcp.pose.raw_pose
            # if isinstance(self.robot_config.additive_y_randomization_bounds, list):
            #     new_end_effector_pose = end_effector_pose.clone()
            #     new_end_effector_pose[:, 1] += self._batched_episode_rng.uniform(self.robot_config.additive_y_randomization_bounds[0], self.robot_config.additive_y_randomization_bounds[1], size=(b,))
                # use inverse kinematics to get the new qpos

                # qpos = self.agent.robot.inverse_kinematics(new_end_effector_pose, max_iter=1000)
            
            # .038 from tcp to flat surface of gripper
            pos = torch.zeros((b, 3))
            pos[:, :2] = end_effector_pose[:, :2]
            pos[:, 2] = end_effector_pose[:, 2] - (self.grasped_book_sizes[:,2]/2) + 0.038 - .0015

            quat = end_effector_pose[:, -4:]
            # apply 180 intrinsic rotation around z-axis
            quat = quaternion_multiply(quat, axis_angle_to_quaternion(torch.tensor([0, 0, np.pi])))
            self.grasped_book.set_pose(Pose.create_from_pq(pos, quat))

            self.xy_slot_location = torch.zeros((b, 2))
            if not (isinstance(self.slot_config.y_randomization_bounds, float) or isinstance(self.slot_config.y_randomization_bounds, int)):
                self.xy_slot_location[:, 1] = common.to_tensor(self._batched_episode_rng.uniform(self.slot_config.y_randomization_bounds[0], self.slot_config.y_randomization_bounds[1], size=(b,)))
            else:
                self.xy_slot_location[:, 1] = self.slot_config.y_randomization_bounds
            self.xy_slot_location[:, 0] = end_effector_pose[:, 0]

            self.slot_width = self.grasped_book_sizes[:, 1] - self.slot_config.negative_tolerance

            quat = torch.tensor([0., 0, 0, 1]).repeat(b, 1)
            # compute the env book poses
            for j in range(len(self.env_books_list)):
                pos = torch.zeros((b, 3))
                pos[:, 0] = self.xy_slot_location[:, 0] - self.env_book_sizes[:, j, 0]/2 + .15/2
                pos[:, 2] = self.env_book_sizes[:, j, 2]/2 + .001
                if j < self.slot_config.left_of_book_index:
                    pos[:, 1] = self.xy_slot_location[:, 1] - ((self.slot_width/2) + self.env_book_sizes[:, j+1:self.slot_config.left_of_book_index, 1].sum(dim=1))
                    pos[:, 1] += -self.env_book_sizes[:, j, 1]/2
                else:
                    pos[:, 1] = self.xy_slot_location[:, 1] + ((self.slot_width/2) + self.env_book_sizes[:, self.slot_config.left_of_book_index:j, 1].sum(dim=1))
                    pos[:, 1] += self.env_book_sizes[:, j, 1]/2

                self.env_books_list[j].set_pose(Pose.create_from_pq(pos, quat))
            
            if self.book_ends_config.mode != 'none':
                identity_quat = torch.tensor([0., 0, 0, 1]).repeat(b, 1)
                
                left_book_end_pos = torch.zeros((b, 3))
                left_book_end_pos[:, 0] = self.xy_slot_location[:, 0]
                left_book_end_pos[:, 2] = self.book_end_sizes[2]/2
                # place the left book end to the left of the left-most env book
                leftmost_env_book_pos = self.env_books_list[0].pose.raw_pose[:, :3]
                left_book_end_pos[:, 1] = leftmost_env_book_pos[:, 1] - (self.env_book_sizes[:, 0, 1]/2) - (self.book_end_sizes[1]/2)
                if self.book_ends_config.mode == 'spring':
                    left_book_end_pos[:, 1] += - (self.book_end_sizes[1]/2 + self.book_ends_config.travel_limit + self.book_ends_config.wall_width/2)
                self.left_book_end.set_pose(Pose.create_from_pq(left_book_end_pos, identity_quat))

                right_book_end_pos = torch.zeros((b, 3))
                right_book_end_pos[:, 0] = self.xy_slot_location[:, 0]
                right_book_end_pos[:, 2] = self.book_end_sizes[2]/2
                # place the right book end to the right of the right-most env book
                rightmost_env_book_pos = self.env_books_list[-1].pose.raw_pose[:, :3]
                right_book_end_pos[:, 1] = rightmost_env_book_pos[:, 1] + (self.env_book_sizes[:, -1, 1]/2) + (self.book_end_sizes[1]/2)
                if self.book_ends_config.mode == 'spring':
                    right_book_end_pos[:, 1] += (self.book_end_sizes[1]/2 + self.book_ends_config.travel_limit + self.book_ends_config.wall_width/2)
                self.right_book_end.set_pose(Pose.create_from_pq(right_book_end_pos, identity_quat))

            # target_EE_pose = self.agent.controller.get_state()['arm']['target_pose']
            self.target_EE_pose.set_pose(end_effector_pose)

            camera_pose = self.scene.sensors['base_camera'].get_params()['cam2world_gl'][0]
            self.cam_rot = quaternion_multiply(matrix_to_quaternion(camera_pose[:3, :3]), axis_angle_to_quaternion(torch.tensor([np.pi, 0, 0])))
            # self.camera_pose.set_pose(Pose.create_from_pq(p=camera_pose[:3, 3], q=cam_rot))

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
        controller_state = self.agent.controller.get_state()
        if 'arm' in controller_state:
            target_EE_pose_in_root = Pose(controller_state['arm']['target_pose'])
            root_pose = self.agent.robot.get_pose()
            target_EE_pose = root_pose * target_EE_pose_in_root
            self.target_EE_pose.set_pose(target_EE_pose)
        # pass
    
    def _get_obs_extra(self, info):
        extra = dict()
        # if 'contact' in self._obs_mode:
        
        if self.render_contact_map:
            extra['extrinsic_contact_positions'] = self.get_extrinsic_contact_positions()
            extra['extrinsic_contact_map'] = self.project_contact_positions_to_camera(extra['extrinsic_contact_positions'])

        if self.render_dtc_maps or self.render_normals_maps:
            extra.update(self.get_extra_contact_features(self.render_dtc_maps, self.render_normals_maps))

        # get current end effector pose
        end_effector_pose = self.agent.tcp.pose.raw_pose # bx7

        extra['end_effector_pose'] = end_effector_pose

        # get end_effector pixel coordinates
        extra['end_effector_pixel_coordinates'] = self.batched_position_to_pixel_coordinates(end_effector_pose[:, :3].unsqueeze(1)).squeeze(1)

        return extra
    
    def get_grasped_object_mesh(self):
        EE_object_length, EE_object_width, EE_object_height = self.grasped_book_sizes[0].tolist()
        EE_object_mesh_list = get_book_primitive_mesh_list(EE_object_length, EE_object_width, EE_object_height, self.binding_thickness, self.cover_thickness, self.cover_overhang, global_transform=convert_sapien_pose_to_transform_matrix(self.non_merged_grasped_books_list[0].pose))
        EE_object_mesh = tm.util.concatenate(EE_object_mesh_list)
        return EE_object_mesh_list, EE_object_mesh

    def get_env_object_meshes(self):
        env_book_sizes_list = [self.env_book_sizes[0,i].tolist() for i in range(len(self.non_merged_env_books_list))]
        env_book_poses_list = [convert_sapien_pose_to_transform_matrix(env_book_over_envs[0].pose) for env_book_over_envs in self.non_merged_env_books_list]
        env_object_meshes_list = get_env_object_meshes_list(env_book_sizes_list, env_book_poses_list, self.binding_thickness, self.cover_thickness, self.cover_overhang)
        env_mesh_list = env_object_meshes_list + self.table_mesh

        env_mesh = tm.util.concatenate(env_object_meshes_list + self.table_mesh)
        return env_mesh_list, env_mesh

    def get_extra_contact_features(self, render_dtc_maps, render_normals_maps):
        # TODO handle parallel envs
        extra_contact_features_dict = dict()

        env_mesh_list, env_mesh = self.get_env_object_meshes()
        EE_object_mesh_list, EE_object_mesh = self.get_grasped_object_mesh()

        ray_origins, ray_directions, pixels_uv = generate_rays_from_camera(self.tm_camera)

        env_hit_min_locations, env_hit_min_pixels_uv, env_hit_min_distances, env_hit_min_index_tri, env_hit_min_ray_directions = get_min_grasped_obj_sdf_at_env_hits_data(ray_origins, ray_directions, pixels_uv, env_mesh, EE_object_mesh_list)
        if render_dtc_maps:
            EE_obj_sdf_on_env_image, EE_obj_sdf_on_env_mask = generate_min_distances_image(env_hit_min_pixels_uv, env_hit_min_distances, self.tm_camera.resolution[::-1])
            EE_obj_sdf_on_env_image = EE_obj_sdf_on_env_image.astype(np.float32)[:240, :320, np.newaxis]
            # EE_obj_sdf_on_env_mask = EE_obj_sdf_on_env_mask.astype(bool)[:240, :320]
            # assert EE_obj_sdf_on_env_image.shape == image_shape + (1,)

            EE_obj_sdf_on_env_image = common.to_tensor(EE_obj_sdf_on_env_image).unsqueeze(0) # hack to add env/batch dimension
            extra_contact_features_dict['env_dtc_map'] = EE_obj_sdf_on_env_image

        if render_normals_maps:
            min_env_surface_normals = env_mesh.face_normals[env_hit_min_index_tri]
            env_xyz_normals_image, env_xyz_normals_image_mask = normals_to_xyz_map(min_env_surface_normals, self.tm_camera.resolution[::-1], env_hit_min_pixels_uv)#, fill_value=1.0/np.sqrt(3.0))
            env_xyz_normals_image = env_xyz_normals_image.astype(np.float32)[:240, :320]
            # env_xyz_normals_image_mask = env_xyz_normals_image_mask.astype(bool)[:240, :320]

            env_xyz_normals_image = common.to_tensor(env_xyz_normals_image).unsqueeze(0) # hack to add env/batch dimension
            extra_contact_features_dict['env_normals_map'] = env_xyz_normals_image

        EE_obj_hit_min_locations, EE_obj_hit_min_pixels_uv, EE_obj_hit_min_distances, EE_obj_hit_min_index_tri, EE_obj_hit_min_ray_directions = get_min_env_sdf_at_grasped_obj_hits_data(ray_origins, ray_directions, pixels_uv, env_mesh_list, EE_object_mesh)
        if render_dtc_maps:
            env_sdf_on_EE_obj_image, env_sdf_on_EE_obj_mask = generate_min_distances_image(EE_obj_hit_min_pixels_uv, EE_obj_hit_min_distances, self.tm_camera.resolution[::-1])
            env_sdf_on_EE_obj_image = env_sdf_on_EE_obj_image.astype(np.float32)[:240, :320, np.newaxis]
            # env_sdf_on_EE_obj_mask = env_sdf_on_EE_obj_mask.astype(bool)[:240, :320]

            env_sdf_on_EE_obj_image = common.to_tensor(env_sdf_on_EE_obj_image).unsqueeze(0) # hack to add env/batch dimension
            extra_contact_features_dict['EE_dtc_map'] = env_sdf_on_EE_obj_image
        
        if render_normals_maps:       
            min_EE_object_surface_normals = EE_object_mesh.face_normals[EE_obj_hit_min_index_tri] # these are normalized already
            EE_object_xyz_normals_image, EE_object_xyz_normals_image_mask = normals_to_xyz_map(min_EE_object_surface_normals, self.tm_camera.resolution[::-1], EE_obj_hit_min_pixels_uv)#, fill_value=1.0/np.sqrt(3.0))
            EE_object_xyz_normals_image = EE_object_xyz_normals_image.astype(np.float32)[:240, :320]
            # EE_object_xyz_normals_image_mask = EE_object_xyz_normals_image_mask.astype(bool)[:240, :320]
            
            EE_object_xyz_normals_image = common.to_tensor(EE_object_xyz_normals_image).unsqueeze(0) # hack to add env/batch dimension
            extra_contact_features_dict['EE_normals_map'] = EE_object_xyz_normals_image
        
        return extra_contact_features_dict

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
            
            height_of_left_book = self.env_book_sizes[:, self.slot_config.left_of_book_index-1, 2]
            height_of_right_book = self.env_book_sizes[:, self.slot_config.left_of_book_index, 2]
            pos[:, 2] = torch.maximum(height_of_left_book, height_of_right_book)
            
            # set orientation to be identity (to world frame)
            pose = Pose.create_from_pq(p=pos)
        return pose
    
    @property
    def grasped_book_info(self):
        info_dict = dict()
        info_dict['sizes'] = self.grasped_book_sizes # num_envs x 3
        info_dict['color'] = self.grasped_book_colors # num_envs x 4
        info_dict['mass'] = self.grasped_book.mass # num_envs
        return info_dict

    @property
    def env_books_info(self):
        info_dict = dict()
        info_dict['sizes'] = self.env_book_sizes # num_envs x 8 x 3
        info_dict['colors'] = self.env_book_colors # num_envs x 8 x 4
        info_dict['masses'] = torch.stack([self.env_books_list[i].mass for i in range(8)]).transpose(1,0) # num_envs x 8
        return info_dict
    
    @property
    def bottom_inner_corner_of_book_right_of_slot_pose(self):
        with torch.device(self.device):
            bottom_inner_corner_in_book_frame = Pose.create_from_pq(p=torch.tensor([0, -self.env_book_sizes[:, self.slot_config.left_of_book_index, 1]/2, -self.env_book_sizes[:, self.slot_config.left_of_book_index, 2]/2]))
            # then also apply 180 intrinsic rotation around z-axis to get x to point in same direction as world
            quat_to_correct_orientation = axis_angle_to_quaternion(torch.tensor([0, 0, np.pi])).repeat(self.num_envs, 1)
        return self.env_books_list[self.slot_config.left_of_book_index].pose * bottom_inner_corner_in_book_frame * Pose.create_from_pq(q=quat_to_correct_orientation)
    
    @property
    def top_inner_corner_of_book_right_of_slot_pose(self):
        with torch.device(self.device):
            top_inner_corner_in_book_frame = Pose.create_from_pq(p=torch.tensor([0, -self.env_book_sizes[:, self.slot_config.left_of_book_index, 1]/2, self.env_book_sizes[:, self.slot_config.left_of_book_index, 2]/2]))
            # then also apply 180 intrinsic rotation around z-axis to get x to point in same direction as world
            quat_to_correct_orientation = axis_angle_to_quaternion(torch.tensor([0, 0, np.pi])).repeat(self.num_envs, 1)
        return self.env_books_list[self.slot_config.left_of_book_index].pose * top_inner_corner_in_book_frame * Pose.create_from_pq(q=quat_to_correct_orientation)

    @property
    def bottom_inner_corner_of_book_left_of_slot_pose(self):
        with torch.device(self.device):
            bottom_inner_corner_in_book_frame = Pose.create_from_pq(p=torch.tensor([0, self.env_book_sizes[:, self.slot_config.left_of_book_index-1, 1]/2, -self.env_book_sizes[:, self.slot_config.left_of_book_index-1, 2]/2]))
            # then also apply 180 intrinsic rotation around z-axis to get x to point in same direction as world
            quat_to_correct_orientation = axis_angle_to_quaternion(torch.tensor([0, 0, np.pi])).repeat(self.num_envs, 1)
        return self.env_books_list[self.slot_config.left_of_book_index-1].pose * bottom_inner_corner_in_book_frame * Pose.create_from_pq(q=quat_to_correct_orientation)
    
    @property
    def top_inner_corner_of_book_left_of_slot_pose(self):
        with torch.device(self.device):
            top_inner_corner_in_book_frame = Pose.create_from_pq(p=torch.tensor([0, self.env_book_sizes[:, self.slot_config.left_of_book_index-1, 1]/2, self.env_book_sizes[:, self.slot_config.left_of_book_index-1, 2]/2]))
            # then also apply 180 intrinsic rotation around z-axis to get x to point in same direction as world
            quat_to_correct_orientation = axis_angle_to_quaternion(torch.tensor([0, 0, np.pi])).repeat(self.num_envs, 1)
        return self.env_books_list[self.slot_config.left_of_book_index-1].pose * top_inner_corner_in_book_frame * Pose.create_from_pq(q=quat_to_correct_orientation)
    
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
                self.grasped_book, self.env_books_list[self.slot_config.left_of_book_index]
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
                self.grasped_book, self.env_books_list[self.slot_config.left_of_book_index-1]
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
            angles = torch.zeros((self.num_envs, self.env_books_config.num_env_books))
            for j in range(self.env_books_config.num_env_books):
                angles[:, j] = self.angle_of_pose_with_vertical(self.env_books_list[j].pose)
            # check if any of the angles are greater than 45 degrees
            toppled = torch.any(angles > self.success_criteria_params['book_toppled_angle_with_vertical_threshold'], dim=1)
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
        if not self.suppress_evaluation:
            not_toppled = ~self.any_env_books_toppled
            bottom_within_slot = self.bottom_of_grasped_book_within_slot
            top_within_slot = self.top_of_grasped_book_within_slot
            z_distance_bw_top_of_grasped_book_and_top_of_slot = self.z_distance_between_top_of_grasped_book_and_top_of_slot
            close_to_top_of_slot = z_distance_bw_top_of_grasped_book_and_top_of_slot < self.success_criteria_params['top_of_grasped_book_distance_to_top_of_slot_threshold']
            transient_success = torch.logical_and(
                not_toppled, torch.logical_and(
                    bottom_within_slot, torch.logical_and(
                        top_within_slot, 
                            close_to_top_of_slot)))
            success_in_a_row = torch.logical_and(transient_success, self.last_eval_bool)
            self.elapsed_success_duration += success_in_a_row.float() * self.time_between_env_steps
            self.elapsed_success_duration *= transient_success.float() # reset to 0 if not transient success
            success = self.elapsed_success_duration > self.success_criteria_params['success_duration_threshold']
            self.last_eval_bool = transient_success
            return dict(
                success=success,
                transient_success=transient_success, 
                elapsed_success_duration=self.elapsed_success_duration,
                z_distance_bw_top_of_grasped_book_and_top_of_slot=z_distance_bw_top_of_grasped_book_and_top_of_slot,
                not_toppled=not_toppled,
                top_within_slot=top_within_slot,
                bottom_within_slot=bottom_within_slot,
                # grasped_book_is_grasped=self.grasped_book_is_grasped,
                )
        else:
            return dict()

