from typing import Any, Dict, Union

import numpy as np
import sapien
import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import XArm7Ability, XArm6Robotiq
from agents.xarm7_allegro import XArm7Allegro
from agents.xarm7_leap import XArm7Leap
from agents.franka_allegro import FrankaAllegro
from mani_skill.envs.sapien_env import BaseEnv
from envs.config.xarm7_allegro_config import PICK_CUBE_CONFIGS, PICK_REWARD_CONFIGS
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors, articulations
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.utils.geometry.rotation_conversions import quaternion_multiply, quaternion_invert, standardize_quaternion

from collections import deque

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

def flatten_dict(dictionary):
    """Flatten nested dict with keys 'actors' and 'articulations' into one map."""
    flat = {}
    for category, sub in dictionary.items():
        # skip non-dict entries just in case
        if not isinstance(sub, dict):
            continue
        for k, v in sub.items():
            flat[k] = v
    return flat

@register_env("xArm7-v1-pick", max_episode_steps=100)
class XArm7TableTop(BaseEnv):

    SUPPORTED_ROBOTS = [
        "xarm7_ability", "xarm6_robotiq", "xarm7_allegro_right", "xarm7_leap_right", "franka_allegro_right"
    ]
    agent: Union[XArm7Ability, XArm6Robotiq, XArm7Allegro, XArm7Leap, FrankaAllegro]
    cube_half_size = 0.1
    goal_thresh = 0.025
    cube_spawn_half_size = 0.05
    cube_spawn_center = (0, 0)
    push_block_spawn_center = (0, 0.5)

    def __init__(self, *args, **kwargs):

        known_custom_args = ["robot_uids", "reward_type", "robot_init_qpos_noise", "max_steps"]

        # environment args
        for arg in known_custom_args:
            setattr(self, arg, kwargs.pop(arg, None))

        self.robot_uids = "xarm7_leap_right" if self.robot_uids is None else self.robot_uids
        self.robot_init_qpos_noise = 0.02 if self.robot_init_qpos_noise is None else self.robot_init_qpos_noise
        self.reward_type = PICK_REWARD_CONFIGS["sac"] if self.reward_type is None else PICK_REWARD_CONFIGS[reward_type]
        
        if self.robot_uids in PICK_CUBE_CONFIGS:
            cfg = PICK_CUBE_CONFIGS[self.robot_uids]
        else:
            cfg = PICK_CUBE_CONFIGS["panda"]

        self.cube_half_size = cfg["cube_half_size"]
        self.goal_thresh = cfg["goal_thresh"]
        self.cube_spawn_half_size = cfg["cube_spawn_half_size"]
        self.cube_spawn_center = cfg["cube_spawn_center"]
        self.push_block_spawn_center = cfg["push_block_spawn_center"]
        self.push_block_half_sizes = cfg["push_block_half_sizes"]
        self.push_block_rot = cfg["push_block_rot"]
        self.max_goal_height = cfg["max_goal_height"]
        self.sensor_cam_eye_pos = cfg["sensor_cam_eye_pos"]
        self.sensor_cam_target_pos = cfg["sensor_cam_target_pos"]
        self.human_cam_eye_pos = cfg["human_cam_eye_pos"]
        self.human_cam_target_pos = cfg["human_cam_target_pos"]
        self.t = 0

        self.residual_scale = 0.05
        self.kwargs = kwargs.copy()

        super().__init__(*args, robot_uids=self.robot_uids, **kwargs)


    @property
    def _default_sim_config(self):
        sim_config = SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=self.num_envs * max(1024, self.num_envs) * 8,
                max_rigid_patch_count=self.num_envs * max(1024, self.num_envs) * 2,
                found_lost_pairs_capacity=2**26,
            )
        )
        return sim_config

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
        self.push_block = actors.build_box(
            self.scene,
            half_sizes=self.push_block_half_sizes,
            color=[0, 0, 1, 1],
            name="push_block",
            initial_pose=sapien.Pose(p=[0, 0.25, self.cube_half_size * 2], q=self.push_block_rot),
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
        self.push_goal_site = actors.build_box(
            self.scene,
            half_sizes=self.push_block_half_sizes,
            color=[0, 0, 1, 0.3],
            name="push_goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(q=self.push_block_rot),
        )

        self._hidden_objects.append(self.goal_site)
        self._hidden_objects.append(self.push_goal_site)

        # self.box = articulations.get_articulation_builder(
        #     self.scene, f"partnet-mobility:{1025}"
        #     )
        # self.box.inital_pose = sapien.Pose(p=[0, 0, 0.5])
        # self.box.build(name="object")

        # Construct one shelf
        # self.shelf_top, self.shelf_legs = self.build_shelf(self.scene)


    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):

            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            
            self.agent.robot.set_qpos(self.agent.keyframes["rest"].qpos)

            # pick cube
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

            # push block
            push_xyz = torch.zeros((b, 3))
            # push_xyz[:, :2] = (
            #     torch.rand((b, 2)) * self.cube_spawn_half_size * 2
            #     - self.cube_spawn_half_size
            # )
            push_xyz[:, 0] += self.push_block_spawn_center[0]
            push_xyz[:, 1] += self.push_block_spawn_center[1]

            push_xyz[:, 2] = self.push_block_half_sizes[2]

            # random rotation for push block and its goal
            push_block_rot = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.push_block.set_pose(Pose.create_from_pq(push_xyz, push_block_rot))

            # goal
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

            push_goal_xyz = torch.zeros((b, 3))
            # push_goal_xyz[:, :2] = (
            #     torch.rand((b, 2)) * self.cube_spawn_half_size * 2
            #     - self.cube_spawn_half_size
            # )
            push_goal_xyz[:, 0] += self.push_block_spawn_center[0]
            # Move goal position to middle of the table
            push_goal_xyz[:, 1] += self.push_block_spawn_center[1] + 0.3
            # Goal height = table height + object height/2
            push_goal_xyz[:, 2] = self.push_block_half_sizes[2]
            self.push_goal_site.set_pose(Pose.create_from_pq(push_goal_xyz, push_block_rot))


    def _get_obs_extra(self, info: Dict):
        # in reality some people hack is_grasped into observations by checking if the gripper can close fully or not

        obs = dict(
            # is_grasped=info["is_grasped"],
            tcp_pose=self.agent.tcp.pose.raw_pose,
            goal_pos=self.goal_site.pose.p,
            push_goal_pos=self.push_goal_site.pose.p,
            robot_state=self.agent.robot.get_qpos(),
            robot_qvel=self.agent.robot.get_qvel(),
            cube_pose=self.cube.pose.raw_pose,
            push_block_pose=self.push_block.pose.raw_pose,
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.cube.pose.raw_pose,
                tcp_to_obj_pos=self.cube.pose.p - self.agent.tcp.pose.p,
                obj_to_goal_pos=self.goal_site.pose.p - self.cube.pose.p,
            )
        return obs


    def _base_action_at_t(self):

        ee_pose = self.agent.tcp_pose.raw_pose
        
        # should change later
        if self.t <= 25:
            target = self.base_waypoints[0]
        else:
            target = self.base_waypoints[1]

        ee_delta = target[:,:3] - ee_pose[:,:3]

        # need to fill rest of action space with 0s
        num_envs = ee_delta.shape[0]
        a_base_full = np.zeros((num_envs, len(self.action_space.low)), dtype=np.float32)
        a_base_full[:,:3] = ee_delta

        return np.clip(a_base_full, self.action_space.low, self.action_space.high)


    def evaluate(self):
        is_obj_placed = (
            torch.linalg.norm(self.goal_site.pose.p - self.cube.pose.p, axis=1)
            <= self.goal_thresh
        )
        # is_grasped = self.agent.is_grasping(self.cube)
        is_robot_static = self.agent.is_static(0.2)
        return {
            "success": is_obj_placed,
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
            # "is_grasped": is_grasped,
        }



    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict): 

        # 1. TCP to object (reaching reward)
        tcp_to_obj_dist = torch.linalg.norm(self.cube.pose.p - self.agent.tcp_pose.p, axis=1)
        reaching_reward = 1.0 - torch.tanh(5.0 * tcp_to_obj_dist)  # Range: (0, 1) - softer, for allegro hand (longer fingers)

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
        place_reward += cube_height * 0.4  # small bonus for height, adjust as needed
        
        # 4. Penalize joint velocity to reduce jerky motion
        qvel = self.agent.robot.get_qvel()
        if self.robot_uids in ["panda", "widowxai"]:
            qvel = qvel[..., :-2]
        elif self.robot_uids == "so100":
            qvel = qvel[..., :-1]
        elif self.robot_uids == "xarm7_leap_right":
            qvel = qvel[..., :-16]
        elif self.robot_uids == "franka_allegro_right":
            qvel = qvel[..., :-16]
        static_penalty = -0.1 * torch.linalg.norm(qvel, axis=1)  # small penalty

        # 5 Contact forces between table and hand penalized:
        total_force = torch.zeros(self.num_envs, device=self.device)
        for link in self.agent.robot.get_links():
            contacts = self.scene.get_pairwise_contact_forces(link, self.table_scene.table)
            if contacts is not None and len(contacts) > 0:
                total_force += contacts.norm(dim=-1)

        has_collision = total_force > 0.01
        collision_penalty = -4.0 * has_collision.float() - 0.05 * total_force

        # 6. Final reward
        reward = (
            self.reward_type["reach_weight"] * reaching_reward +
            self.reward_type["finger_dist_weight"] * torch.mean(finger_dist_reward, dim=-1) +
            self.reward_type["grasp_weight"] * grasp_reward +
            self.reward_type["place_weight"] * place_reward +
            self.reward_type["static_penalty_weight"] * static_penalty +
            self.reward_type["collision_penalty_weight"] * collision_penalty
        )

        # Bonus for success
        reward[info["success"]] += 1.0
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5
