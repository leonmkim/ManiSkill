from typing import Any, Dict, Union

import numpy as np
import sapien
import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import Fetch, Panda, XArm6Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.geometry.rotation_conversions import quaternion_multiply, axis_angle_to_quaternion, quaternion_apply


@register_env("SpringArticulationEnv-v0", max_episode_steps=50)
class SpringArticulationEnv(BaseEnv):
    """
    **Task Description:**
    A simple task where the objective is to grasp a red cube and move it to a target goal position.

    **Randomizations:**
    - the cube's xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1]. It is placed flat on the table
    - the cube's z-axis rotation is randomized to a random angle
    - the target goal position (marked by a green sphere) of the cube has its xy position randomized in the region [0.1, 0.1] x [-0.1, -0.1] and z randomized in [0, 0.3]

    **Success Conditions:**
    - the cube position is within `goal_thresh` (default 0.025m) euclidean distance of the goal position
    - the robot is static (q velocity < 0.2)
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PickCube-v1_rt.mp4"
    SUPPORTED_ROBOTS = [
        "panda",
        "fetch",
        "xarm6_robotiq",
    ]
    agent: Union[Panda, Fetch, XArm6Robotiq]
    cube_half_size = 0.02
    goal_thresh = 0.025

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        # self.cube = actors.build_cube(
        #     self.scene,
        #     half_size=self.cube_half_size,
        #     color=[1, 0, 0, 1],
        #     name="cube",
        #     initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        # )
        # self.goal_site = actors.build_sphere(
        #     self.scene,
        #     radius=self.goal_thresh,
        #     color=[0, 1, 0, 1],
        #     name="goal_site",
        #     body_type="kinematic",
        #     add_collision=False,
        #     initial_pose=sapien.Pose(),
        # )
        # self._hidden_objects.append(self.goal_site)
        articulation_builder = self.scene.create_articulation_builder()
        articulation_builder.set_initial_pose(
            sapien.Pose(p=[0, 0, self.cube_half_size+0.01])
        )
        root_box = articulation_builder.create_link_builder()
        root_box.set_name("root_box")
        root_box.add_box_collision(
            half_size=[self.cube_half_size] * 3, density=1000,
        )
        root_box.add_box_visual(
            half_size=[self.cube_half_size] * 3,
            material=[1,0,0],
        )
        # root_box.set_physx_body_type("static")
        child_box = articulation_builder.create_link_builder(root_box)
        child_box.set_name("child_box")
        child_box.set_joint_name("child_box_joint")
        child_box.add_box_collision(
            half_size=[self.cube_half_size] * 3, density=1000,
        )
        child_box.add_box_visual(
            half_size=[self.cube_half_size] * 3,
            material=[0,0,1],
        )

        child_box.set_joint_properties(
            type='prismatic',
            limits=[[0, 0.2]],
            pose_in_parent=sapien.Pose(p=[0, 0, self.cube_half_size], q=axis_angle_to_quaternion(torch.tensor([0, -np.pi/2, 0]))), # Parent_T_Joint
            pose_in_child=sapien.Pose(p=[0, 0, -(self.cube_half_size)], q=axis_angle_to_quaternion(torch.tensor([0, -np.pi/2., 0]))), # Child_T_Joint
            friction=0.0,
            damping=0.0,
        )
        
        self.two_boxes_with_spring = articulation_builder.build(name="two_boxes_with_spring", fix_root_link=True)


        self.two_boxes_with_spring.find_joint_by_name("child_box_joint").set_drive_properties(
            stiffness=50.0,
            damping=np.sqrt(50.0),
        )
        self.two_boxes_with_spring.find_joint_by_name("child_box_joint").set_drive_target(0.1)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[:, 0] = -0.2
            xyz[:, 2] = self.cube_half_size
            # qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.two_boxes_with_spring.set_pose(Pose.create_from_pq(xyz, [1, 0, 0, 0]))
            # self.cube.set_pose(Pose.create_from_pq(xyz, qs))

            # goal_xyz = torch.zeros((b, 3))
            # goal_xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1
            # goal_xyz[:, 2] = torch.rand((b)) * 0.3 + xyz[:, 2]
            # self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))


if __name__ == "__main__":
    import gym
    from mani_skill.utils.registration import register_env

    env = gym.make("SpringArticulationEnv-v1")
    env.reset()
    env.render()
