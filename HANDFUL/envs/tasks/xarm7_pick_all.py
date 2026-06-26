from typing import Any, Dict, Union

import numpy as np
import sapien
import torch

from mani_skill.agents.robots import XArm7Ability, XArm6Robotiq
from agents.xarm7_allegro import XArm7Allegro
from agents.xarm7_leap import XArm7Leap
from agents.franka_allegro import FrankaAllegro
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.envs.sapien_env import BaseEnv
from envs.config.xarm7_allegro_config import (PICK_CUBE_CONFIGS, PICK_REWARD_CONFIGS,
                                             MASS_VARIATIONS, FRICTION_VARIATIONS, RESTITUTION_VARIATIONS)
from envs.utils.batched_pose import random_quaternions_batched
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors, articulations
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Actor, Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig

import sapien.physx as physx
from sapien.physx import PhysxRigidBodyComponent
from sapien.render import RenderBodyComponent

from collections import deque

MAX_EPISODE_STEPS = 75
N_MATERIAL_POOL = 512

PICK_CUBE_DOC_STRING = """**Task Description:**
TBD
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


@register_env("xArm7-v1-pick-all", max_episode_steps=MAX_EPISODE_STEPS)
class XArm7TableTopPickAll(BaseEnv):

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
        # somehow couldn't find max epsiode steps 
        known_custom_args = ["robot_uids", "reward_type", "robot_init_qpos_noise"]
        for arg in known_custom_args:
            setattr(self, arg, kwargs.pop(arg, None))

        self.max_steps = MAX_EPISODE_STEPS

        self.robot_uids = "xarm7_leap_right" if self.robot_uids is None else self.robot_uids
        self.robot_init_qpos_noise = 0.02 if self.robot_init_qpos_noise is None else self.robot_init_qpos_noise
        self.reward_type = PICK_REWARD_CONFIGS["sac"] if self.reward_type is None else PICK_REWARD_CONFIGS[self.reward_type]
        
        if self.robot_uids in PICK_CUBE_CONFIGS:
            cfg = PICK_CUBE_CONFIGS[self.robot_uids]
        else:
            cfg = PICK_CUBE_CONFIGS["xarm7_leap_right"]

        self.cube_variation_dimensions = cfg["cube_variation_dimensions"]
        self.goal_thresh = cfg["goal_thresh"]
        self.cube_spawn_half_size = cfg["cube_spawn_half_size"]
        self.cube_spawn_center = cfg["cube_spawn_center"]

        
        self.max_goal_height = cfg["max_goal_height"]
        self.sensor_cam_eye_pos = cfg["sensor_cam_eye_pos"]
        self.sensor_cam_target_pos = cfg["sensor_cam_target_pos"]
        self.human_cam_eye_pos = cfg["human_cam_eye_pos"]
        self.human_cam_target_pos = cfg["human_cam_target_pos"]
        self.t = 0

        self.residual_scale = 0.05

        super().__init__(*args, robot_uids=self.robot_uids, **kwargs)


    @property
    def _default_sim_config(self):
        sim_config = SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=self.num_envs * max(1024, self.num_envs) * 128,
                max_rigid_patch_count=self.num_envs * max(1024, self.num_envs) * 64,
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
        
        # sample a handful of different cube sizes
        cube_size_indices = self._batched_episode_rng.randint(0, high=len(self.cube_variation_dimensions))

        # create different cube for each parallel environment
        self._cubes = []
        for i in range(self.num_envs):
            cube_half_sizes = self.cube_variation_dimensions[cube_size_indices[i]]
            
            cube = actors.build_box(
                self.scene,
                half_sizes=cube_half_sizes,
                color=[1, 0, 0, 1],
                name=f"cube_{i}",
                scene_idxs=[i],
                initial_pose=sapien.Pose(p=[0, 0.25, cube_half_sizes[2]]),
            )
            self.remove_from_state_dict_registry(cube)
            self._cubes.append(cube)

        # Merge all cubes into a single Actor
        self.cube = Actor.merge(self._cubes, name="cube")
        self.add_to_state_dict_registry(self.cube)

        # Store the half_sizes for each environment for later reference
        self.cube_half_sizes_per_env = torch.tensor(
            [self.cube_variation_dimensions[idx] for idx in cube_size_indices],
            dtype=torch.float32,
            device=self.device
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

        # object randomizations
        if not hasattr(self, '_material_pool'):
            self._material_pool = [
                physx.PhysxMaterial(
                    static_friction=self._batched_episode_rng[i % self.num_envs].uniform(
                        low=FRICTION_VARIATIONS[0], high=FRICTION_VARIATIONS[1]),
                    dynamic_friction=self._batched_episode_rng[i % self.num_envs].uniform(
                        low=FRICTION_VARIATIONS[0], high=FRICTION_VARIATIONS[1]),
                    restitution=self._batched_episode_rng[i % self.num_envs].uniform(
                        low=RESTITUTION_VARIATIONS[0], high=RESTITUTION_VARIATIONS[1]),
                )
                for i in range(N_MATERIAL_POOL)
            ]

        for i, obj in enumerate(self.cube._objs):
            
            rb: PhysxRigidBodyComponent = obj.find_component_by_type(PhysxRigidBodyComponent)

            if rb is not None:
                rb.mass *= self._batched_episode_rng[i].uniform(low=MASS_VARIATIONS[0], high=MASS_VARIATIONS[1])
                mat = self._material_pool[self._batched_episode_rng[i].randint(0, N_MATERIAL_POOL)]

                for shape in rb.collision_shapes:
                    shape.physical_material = mat

        # robot randomizations
        for link in self.agent.robot.links:
            for i, obj in enumerate(link._objs):
                rb: PhysxRigidBodyComponent = obj.entity.find_component_by_type(PhysxRigidBodyComponent)
                if rb is not None:

                    rb.mass *= self._batched_episode_rng[i].uniform(low=MASS_VARIATIONS[0], high=MASS_VARIATIONS[1])
                    mat = self._material_pool[self._batched_episode_rng[i].randint(0, N_MATERIAL_POOL)]

                    for shape in rb.collision_shapes:
                        shape.physical_material = mat

    def hide_robot(self):
        for link in self.agent.robot.links:
            for obj in link._objs:
                render_body: RenderBodyComponent = obj.entity.find_component_by_type(RenderBodyComponent)
                if render_body is not None:
                    for render_shape in render_body.render_shapes:
                        for part in render_shape.parts:
                            mat = part.material
                            # Preserve existing color, just lower opacity
                            mat.set_base_color([
                                mat.base_color[0],
                                mat.base_color[1],
                                mat.base_color[2],
                                0.0
                            ])

    def show_robot(self):

        ACTIVE   = [0.0, 0.8, 0.2, 1.0]   # green
        INACTIVE = [0.85, 0.2, 0.2, 1.0]  # red
        DEFAULT  = [0.7, 0.7, 0.7, 1.0]   # arm / palm neutral

        robot = self.agent.robot
        agent = self.agent

        # Vector env → use env 0 indices for visualization
        active_set = set(self.active_finger_indices[0].tolist()) \
            if hasattr(self, "active_finger_indices") else set()

        inactive_set = set(self.inactive_finger_indices[0].tolist()) \
            if hasattr(self, "inactive_finger_indices") else set()

        # -------------------------------------------------
        # Build link name → finger id mapping
        # -------------------------------------------------

        finger_link_map = {}

        link_groups = [
            agent.mcp_link_names,
            agent.pip_link_names,
            agent.front_link_names,
            agent.tip_link_names,
        ]

        num_fingers = len(agent.tip_link_names)

        for fid in range(num_fingers):
            for group in link_groups:
                if fid < len(group):
                    link_name = group[fid]
                    finger_link_map[link_name] = fid

        # -------------------------------------------------
        # Apply colors
        # -------------------------------------------------

        for link in robot.get_links():

            name = link.name
            color = None

            if name in finger_link_map:
                fid = finger_link_map[name]

                if fid in active_set:
                    color = ACTIVE
                elif fid in inactive_set:
                    color = INACTIVE
            
            for obj in link._objs:
                render_body = obj.entity.find_component_by_type(
                    sapien.render.RenderBodyComponent
                )
                if render_body is not None:

                    for shape in render_body.render_shapes:
                        for part in shape.parts:
                            mat = part.material
                            if color:
                                mat.set_base_color(color)
                            else:
                                mat.set_base_color([
                                    mat.base_color[0],
                                    mat.base_color[1],
                                    mat.base_color[2],
                                    1
                                ])


    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):

            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            current_cube_half_sizes = self.cube_half_sizes_per_env[env_idx]
            
            # pick cube
            xyz = torch.zeros((b, 3))
            random_xy = self._batched_episode_rng.uniform(
                low=-self.cube_spawn_half_size,
                high=self.cube_spawn_half_size,
                size=(2,)
            )
            xyz[:, :2] = torch.from_numpy(random_xy).float().to(self.device)

            xyz[:, 0] += self.cube_spawn_center[0]
            xyz[:, 1] += self.cube_spawn_center[1]
            xyz[:, 2] = current_cube_half_sizes[:, 2]

            qs = random_quaternions_batched(self._batched_episode_rng, device=self.device, lock_x=True, lock_y=True)
            self.cube.set_pose(Pose.create_from_pq(xyz, qs))

            # goal
            goal_xyz = torch.zeros((b, 3))
            random_goal_xy = self._batched_episode_rng.uniform(
                low=-self.cube_spawn_half_size, 
                high=self.cube_spawn_half_size, 
                size=(2,)
            )
            goal_xyz[:, :2] = torch.from_numpy(random_goal_xy).float().to(self.device)

            goal_xyz[:, 0] += self.cube_spawn_center[0]
            goal_xyz[:, 1] += self.cube_spawn_center[1]
            goal_xyz[:, 2] = self.max_goal_height
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

            qpos = self.agent.keyframes["rest"].qpos

            qpos = (self._batched_episode_rng.normal(0, self.robot_init_qpos_noise, (len(qpos))) + qpos)
            self.agent.reset(init_qpos=qpos)

    def _get_obs_extra(self, info: Dict):
        # in reality some people hack is_grasped into observations by checking if the gripper can close fully or not

        obs = dict(
            # is_grasped=info["is_grasped"],
            goal_pos=self.goal_site.pose.p,
            cube_pose=self.cube.pose.raw_pose,
            cube_half_sizes=self.cube_half_sizes_per_env,
        )
        if "state" in self.obs_mode:
            obs.update(
                tcp_to_obj_pos=self.cube.pose.p - self.agent.tcp.pose.p,
                obj_to_goal_pos=self.goal_site.pose.p - self.cube.pose.p,
            )
        return obs

    def evaluate(self):

        finger_object_dist = torch.linalg.norm(
            self.agent.tip_poses[..., :3] - self.cube.pose.p[:, None, :], dim=-1
        )

        finger_close_fraction = (finger_object_dist < 0.07).float().mean(dim=-1)
        is_obj_grasped = (finger_close_fraction > 0.5)

        is_obj_placed = (
            torch.linalg.norm(self.goal_site.pose.p - self.cube.pose.p, axis=1)
            <= self.goal_thresh
        )

        total_force = torch.zeros(self.num_envs, device=self.device)
        for link in self.agent.robot.get_links():
            contacts = self.scene.get_pairwise_contact_forces(link, self.table_scene.table)
            if contacts is not None and len(contacts) > 0:
                total_force += contacts.norm(dim=-1)

        is_robot_static = self.agent.is_static(0.2)
        return {
            "success": is_obj_placed & is_obj_grasped,
            "is_robot_static": is_robot_static,
            "hand_table_force": total_force,
        }



    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict): 

        # 1. TCP to object (reaching reward)
        tcp_to_obj_dist = torch.linalg.norm(self.cube.pose.p - self.agent.tcp_pose.p, axis=1)
        reaching_reward = 1.0 - torch.tanh(5.0 * tcp_to_obj_dist) 

        # 2. Finger proximity reward (soft reward instead of binary)
        finger_object_dist = torch.linalg.norm(
            self.agent.tip_poses[..., :3] - self.cube.pose.p[:, None, :], dim=-1
        )  # shape: (B, num_fingers)

        # All fingers are active, so use all of them
        finger_dist_reward = torch.exp(-20 * finger_object_dist)

        # palm contacts
        palm_object_dist = torch.linalg.norm(
            self.agent.palm_contact_points[..., :3] - self.cube.pose.p[:, None, :], dim=-1
        )  # shape: (B, num_palm_points)
        palm_min_dist = torch.min(palm_object_dist, dim=-1)[0]  # shape: (B,)
        palm_dist_reward = torch.exp(-20 * palm_min_dist)
        palm_close = (palm_min_dist < 0.05).float()
        num_active_contacts = 5
        
        combined_dist_reward = (torch.sum(finger_dist_reward, dim=-1) + palm_dist_reward) / num_active_contacts
        finger_close_fraction  = (finger_object_dist < 0.05).float().mean(dim=-1)
        grasp_reward = (finger_close_fraction * 4 + palm_close) / num_active_contacts

        # 3. Object to goal (only meaningful if object is lifted/moved) + height
        obj_to_goal_dist = torch.linalg.norm(self.goal_site.pose.p - self.cube.pose.p, axis=1)
        place_reward = (1.0 - torch.tanh(5.0 * obj_to_goal_dist))



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

        static_penalty = -1.0 * torch.linalg.norm(qvel, axis=1) 

        # 5 Contact forces between table and hand penalized:
        total_force = torch.zeros(self.num_envs, device=self.device)
        for link in self.agent.robot.get_links():
            contacts = self.scene.get_pairwise_contact_forces(link, self.table_scene.table)
            if contacts is not None and len(contacts) > 0:
                total_force += contacts.norm(dim=-1)

        collision_penalty = torch.clamp((-0.05 * total_force), min=-1.0, max=0.0)

        # 6. Final reward 
        reward = (
            self.reward_type["reach_weight"] * reaching_reward +
            self.reward_type["finger_dist_weight"] * combined_dist_reward +
            self.reward_type["grasp_weight"] * grasp_reward +
            self.reward_type["place_weight"] * place_reward +
            self.reward_type["static_penalty_weight"] * static_penalty +
            self.reward_type["collision_penalty_weight"] * collision_penalty
        )

        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        sum_rewards = 0

        for key in ["reach_weight", "finger_dist_weight", "grasp_weight", "place_weight"]:
            sum_rewards += self.reward_type[key]

        return self.compute_dense_reward(obs=obs, action=action, info=info) / sum_rewards