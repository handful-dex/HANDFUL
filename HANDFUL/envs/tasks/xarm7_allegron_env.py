from typing import Any, Dict, Union

import numpy as np
import sapien
import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import XArm7Ability, XArm6Robotiq
from agents.xarm7_allegro import XArm7Allegro
from agents.xarm7_leap import XArm7Leap
from mani_skill.envs.sapien_env import BaseEnv
from envs.config.xarm7_allegro_config import PICK_CUBE_CONFIGS
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors, articulations
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
import gymnasium as gym

PICK_CUBE_DOC_STRING = """**Task Description:**
A simple task where the objective is to grasp a red cube with the {robot_id} robot and move it to a target goal position. This is also the *baseline* task to test whether a robot with manipulation
capabilities can be simulated and trained properly. Hence there is extra code for some robots to set them up properly in this environment as well as the table scene builder.

**Randomizations:**
- the cube's xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1]. It is placed flat on the table
- the cube's z-axis rotation is randomized to a random angle
- the target goal position (marked by a green sphere) of the cube has its xy position randomized in the region [0.1, 0.1] x [-0.1, -0.1] and z randomized in [0, 0.3]

**Success Conditions:**
- the cube position is within `goal_thresh` (default 0.025m) euclidean distance of the goal position
- the robot is static (q velocity < 0.2)
"""


@register_env("xArm7-v1", max_episode_steps=1000)
class XArm7TableTop(BaseEnv):

    SUPPORTED_ROBOTS = [
        "xarm7_ability", "xarm6_robotiq", "xarm7_allegro_right", "xarm7_leap_right"
    ]
    agent: Union[XArm7Ability, XArm6Robotiq, XArm7Allegro, XArm7Leap]
    cube_half_size = 0.1
    goal_thresh = 0.025
    cube_spawn_half_size = 0.05
    cube_spawn_center = (0, 0)

    def __init__(self, *args, robot_uids="xarm7_leap_right", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        if robot_uids in PICK_CUBE_CONFIGS:
            cfg = PICK_CUBE_CONFIGS[robot_uids]
        else:
            cfg = PICK_CUBE_CONFIGS["panda"]
        self.cube_half_size = cfg["cube_half_size"]
        self.goal_thresh = cfg["goal_thresh"]
        self.cube_spawn_half_size = cfg["cube_spawn_half_size"]
        self.cube_spawn_center = cfg["cube_spawn_center"]
        self.max_goal_height = cfg["max_goal_height"]
        self.sensor_cam_eye_pos = cfg["sensor_cam_eye_pos"]
        self.sensor_cam_target_pos = cfg["sensor_cam_target_pos"]
        self.human_cam_eye_pos = cfg["human_cam_eye_pos"]
        self.human_cam_target_pos = cfg["human_cam_target_pos"]
        self.time_step = 0

        self.residual_scale = 0.05

        # Residual scale for residual learning
        # Loading base trajectory

        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(
            eye=self.sensor_cam_eye_pos, target=self.sensor_cam_target_pos
        )
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(
            eye=self.human_cam_eye_pos, target=self.human_cam_target_pos
        )
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],
            name="cube",
            initial_pose=sapien.Pose(p=[0, 0.25, self.cube_half_size]),
        )
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)

        # self.box = articulations.get_articulation_builder(
        #     self.scene, f"partnet-mobility:{1025}"
        #     )
        # self.box.inital_pose = sapien.Pose(p=[0, 0, 0.5])
        # self.box.build(name="object")

        # Construct one shelf
        self.shelf_top, self.shelf_legs = self.build_shelf(self.scene)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = (
                torch.rand((b, 2)) * self.cube_spawn_half_size * 2
                - self.cube_spawn_half_size
            )
            xyz[:, 0] += self.cube_spawn_center[0]
            xyz[:, 1] += self.cube_spawn_center[1] + 0.2

            xyz[:, 2] = self.cube_half_size
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cube.set_pose(Pose.create_from_pq(xyz, qs))

            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, :2] = (
                torch.rand((b, 2)) * self.cube_spawn_half_size * 2
                - self.cube_spawn_half_size
            )
            goal_xyz[:, 0] += self.cube_spawn_center[0]
            # Move goal position to middle of the table
            goal_xyz[:, 1] += self.cube_spawn_center[1] + 0.2
            # Goal height = table height + object height/2
            goal_xyz[:, 2] = 0.25 #torch.rand((b)) * self.max_goal_height + xyz[:, 2]
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

    def _get_obs_extra(self, info: Dict):
        # in reality some people hack is_grasped into observations by checking if the gripper can close fully or not
        obs = dict(
            # is_grasped=info["is_grasped"],
            tcp_pose=self.agent.tcp.pose.raw_pose,
            goal_pos=self.goal_site.pose.p,
            robot_state=self.agent.robot.get_qpos(),
            robot_qvel=self.agent.robot.get_qvel(),
            cube_pose=self.cube.pose.raw_pose,
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.cube.pose.raw_pose,
                tcp_to_obj_pos=self.cube.pose.p - self.agent.tcp.pose.p,
                obj_to_goal_pos=self.goal_site.pose.p - self.cube.pose.p,
            )
        return obs
    
    def residual_step(self, residual_action: torch.Tensor):
        """
        Perform a step in the environment with the given action.
        This method is used for residual learning, where the action is applied to the current state.
        """
        # TODO: Residual learning

        # Scale residual to env’s action scale
        a_res = np.clip(residual_action, -1.0, 1.0) * self.residual_scale * (self.env.action_space.high - self.env.action_space.low) / 2.0
        a_base = self._base_action_at_t()

        # Compose and clip to env bounds
        a = np.clip(a_base + a_res, self.env.action_space.low, self.env.action_space.high)

        obs, rew, terminated, truncated, info = self.env.step(a)
        self.t += 1

        if self.augment_obs:
            next_base = self._base_action_at_t()
            t_norm = 2.0 * (self.t / max(1, self.T - 1)) - 1.0
            obs = np.concatenate([obs, next_base, np.array([t_norm], dtype=np.float32)]).astype(np.float32)

        return obs, rew, terminated, truncated, info

    def evaluate(self):
        is_obj_placed = (
            torch.linalg.norm(self.goal_site.pose.p - self.cube.pose.p, axis=1)
            <= self.goal_thresh
        )
        # is_grasped = self.agent.is_grasping(self.cube)
        is_robot_static = self.agent.is_static(0.2)
        return {
            "success": is_obj_placed & is_robot_static,
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
            # "is_grasped": is_grasped,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # 1. TCP to object (reaching reward)
        tcp_to_obj_dist = torch.linalg.norm(self.cube.pose.p - self.agent.tcp_pose.p, axis=1)
        reaching_reward = 1.0 - torch.tanh(5.0 * tcp_to_obj_dist)  # Range: (0, 1)

        # 2. Finger proximity reward (soft reward instead of binary)
        finger_object_dist = torch.linalg.norm(
            self.agent.tip_poses[..., :3] - self.cube.pose.p[:, None, :], dim=-1
        )  # shape: (B, num_fingers)
        finger_dist_reward = torch.exp(-20 * finger_object_dist)  # sharper than exp(-5 * mean)

        # Optional: mean over all fingers or require 3/4 to be close
        finger_close_fraction = (finger_object_dist < 0.03).float().mean(dim=-1)

        # Grasp reward (soft version)
        grasp_reward = finger_close_fraction  # or use: torch.mean(finger_dist_reward, dim=-1)

        # 3. Object to goal (only meaningful if object is lifted/moved)
        obj_to_goal_dist = torch.linalg.norm(self.goal_site.pose.p - self.cube.pose.p, axis=1)
        place_reward = (1.0 - torch.tanh(5.0 * obj_to_goal_dist)) * finger_close_fraction
        
        # give higher reward if object height is above a threshold
        cube_height = self.cube.pose.p[:, 2]
        place_reward += cube_height  # small bonus for height, adjust as needed
        
        # 4. Penalize joint velocity to reduce jerky motion
        qvel = self.agent.robot.get_qvel()
        if self.robot_uids in ["panda", "widowxai"]:
            qvel = qvel[..., :-2]
        elif self.robot_uids == "so100":
            qvel = qvel[..., :-1]
        elif self.robot_uids == "xarm7_leap_right":
            qvel = qvel[..., :-16]
        static_penalty = -0.1 * torch.linalg.norm(qvel, axis=1)  # small penalty

        # 5. Final reward
        reward = (
            5.0 * reaching_reward +
            5.0 * torch.mean(finger_dist_reward, dim=-1) +
            2.0 * grasp_reward +
            5.0 * place_reward +
            static_penalty
        )

        # Bonus for success
        reward[info["success"]] = 5.0
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5

    @staticmethod
    def build_shelf(scene, x=0.0, y=-0.3):
        # Dimensions
        tabletop_half = np.array([0.3, 0.15, 0.02], dtype=np.float32)  # 60x60x4cm
        table_height = 0.2  # total height from ground
        leg_half = np.array([0.01, 0.01, table_height/2], dtype=np.float32)      # 4x4x60cm
        leg_offset = tabletop_half[:2] - leg_half[:2]

        # Tabletop position
        tabletop_z = table_height
        tabletop_pose = sapien.Pose(p=[x, y, tabletop_z])

        # Build tabletop
        tabletop = actors.build_box(
            scene,
            half_sizes=tabletop_half,
            initial_pose=tabletop_pose,
            color=[0.8, 0.6, 0.4, 1],
            name="tabletop",
            body_type="static",
        )

        # Build 4 legs
        legs = []
        leg_xy_offsets = [
            [+leg_offset[0], +leg_offset[1]],
            [-leg_offset[0], +leg_offset[1]],
            [+leg_offset[0], -leg_offset[1]],
            [-leg_offset[0], -leg_offset[1]],
        ]

        for i, (dx, dy) in enumerate(leg_xy_offsets):
            leg_pose = sapien.Pose(
                p=[
                    x + dx,
                    y + dy,
                    tabletop_z - tabletop_half[2] - leg_half[2]
                ]
            )
            leg = actors.build_box(
                scene,
                half_sizes=leg_half,
                initial_pose=leg_pose,
                # color=[0.5, 0.3, 0.1, 1],
                color=[0.6, 0.6, 0.6, 1],
                name=f"table_leg_{i}",
                body_type="static",
            )
            legs.append(leg)

        return tabletop, legs