from typing import Any, Dict, Union

import numpy as np
import sapien
import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import XArm7Ability, XArm6Robotiq
from agents.xarm7_allegro import XArm7Allegro
from agents.xarm7_leap import XArm7Leap
from agents.franka_allegro import FrankaAllegro
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.envs.sapien_env import BaseEnv
from envs.config.xarm7_allegro_config import (TWO_PICK_CONFIGS, TWO_PICK_REWARD_CONFIGS,
                                             MASS_VARIATIONS, FRICTION_VARIATIONS, RESTITUTION_VARIATIONS)
from envs.utils.batched_pose import random_quaternions_batched
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors, articulations
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Actor, Pose

from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.utils.geometry.rotation_conversions import quaternion_multiply, quaternion_invert, standardize_quaternion

import sapien.physx as physx
from sapien.physx import PhysxRigidBodyComponent
from sapien.render import RenderBodyComponent

DEFAULT_ACTIVE_FINGER_INDICES = [[1]]
DEFAULT_INACTIVE_FINGER_INDICES = [[0, 3, 0]]

N_MATERIAL_POOL = 1024

PICK_CUBE_DOC_STRING = """**Task Description:**
TBD
"""



@register_env("xArm7-v1-two-pick", max_episode_steps=100)
class XArm7TableTopTwoPick(BaseEnv):

    SUPPORTED_ROBOTS = [
        "xarm7_ability", "xarm6_robotiq", "xarm7_allegro_right", "xarm7_leap_right"
    ]
    agent: Union[XArm7Ability, XArm6Robotiq, XArm7Allegro, XArm7Leap]
    # goal_thresh = 0.025
    # cube_spawn_half_size = 0.05

    def __init__(self, *args, state_file_path: str = "",  **kwargs):
        
        known_custom_args = ["robot_uids", "reward_type", "robot_init_qpos_noise", "difficulty"]

        # push_env args
        for arg in known_custom_args:
            setattr(self, arg, kwargs.pop(arg, None))

        self.robot_uids = "xarm7_leap_right" if self.robot_uids is None else self.robot_uids
        self.robot_init_qpos_noise = 0.02 if self.robot_init_qpos_noise is None else self.robot_init_qpos_noise
        self.reward_type = TWO_PICK_REWARD_CONFIGS["sac"] if self.reward_type is None else TWO_PICK_REWARD_CONFIGS[self.reward_type]
        self.difficulty = 3 if self.difficulty is None else self.difficulty

        if self.robot_uids in TWO_PICK_CONFIGS:
            cfg = TWO_PICK_CONFIGS[self.robot_uids]
        else:
            cfg = TWO_PICK_CONFIGS["panda"]

        self.cube_variation_dimensions = cfg["cube_variation_dimensions"]
        self.push_block_variation_dimensions = cfg["push_block_variation_dimensions"]
        self.goal_thresh = cfg["goal_thresh"]
        self.cube_spawn_half_size = cfg["cube_spawn_half_size"]

        self.push_block_spawn_center = cfg["push_block_spawn_center"]
        self.push_block_rot = cfg["push_block_rot"]
        
        self.max_goal_height = cfg["max_goal_height"]
        self.sensor_cam_eye_pos = cfg["sensor_cam_eye_pos"]
        self.sensor_cam_target_pos = cfg["sensor_cam_target_pos"]
        self.human_cam_eye_pos = cfg["human_cam_eye_pos"]
        self.human_cam_target_pos = cfg["human_cam_target_pos"]

        if self.difficulty <= 1:
            self.cube_spawn_half_size = 0
        elif self.difficulty == 2:
            self.cube_spawn_half_size /= 2
        else:
            self.cube_spawn_half_size = self.cube_spawn_half_size

        # Load states from file
        self.state_file_path = state_file_path
        self.all_episodes = None
        if self.state_file_path:
            try:
                self.all_episodes = torch.load(self.state_file_path)
                print(f"Successfully loaded {len(self.all_episodes)} states from {self.state_file_path}")
            except FileNotFoundError:
                print(f"Warning: The state file '{self.state_file_path}' was not found.")
                print("The environment will not be able to reset to saved states.")
                self.all_episodes = []
            except Exception as e:
                print(f"Error loading states from '{self.state_file_path}': {e}")
                self.all_episodes = []

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
        return CameraConfig("render_camera", pose, 512, 512, 1.25, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):

        # In your task's _load_scene
        self.scene.add_point_light(position=[0, 0.7, 0.7], color=[1, 1, 1])
        self.scene.set_ambient_light([1, 1, 1])


        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # check if we passed specific states to load, if so, take them
        if options is not None and "intermediate_states" in options:
            self.intermediate_states = options["intermediate_states"]
            
            cube_sizes_for_creation = [state['cube_half_sizes'] for state in self.intermediate_states]

        elif self.all_episodes and len(self.all_episodes) > 0:

            # Sample a random episode for each environment
            num_episodes = len(self.all_episodes)
            episode_indices = self._batched_episode_rng.choice(num_episodes, replace=True)
            
            # Sample the exact last step of each selected episode
            selected_states = []
            cube_sizes_for_creation = []

            for i in episode_indices:
                episode_trajectory = self.all_episodes[i]
                if not episode_trajectory:
                    raise RuntimeError(f"Episode at index {i} is empty.")
                
                # Select the very last step from the trajectory
                step_to_load = episode_trajectory[-1]
                selected_states.append(step_to_load)

                cube_sizes_for_creation.append(step_to_load['cube_half_sizes'])

            self.intermediate_states = selected_states
        else:
            # load some default active finger indices and sample random cube sizes
            # mostly for debug code where you don't load intermediate states
            self.intermediate_states = None
            if not hasattr(self, 'active_finger_indices'):
                self.active_finger_indices = torch.tensor(
                    DEFAULT_ACTIVE_FINGER_INDICES, device=self.device
                ).expand(self.num_envs, -1)
                self.inactive_finger_indices = torch.tensor(
                    DEFAULT_INACTIVE_FINGER_INDICES, device=self.device
                ).expand(self.num_envs, -1)

            cube_size_indices = self._batched_episode_rng.randint(0, high=len(self.cube_variation_dimensions))
            cube_sizes_for_creation = [self.cube_variation_dimensions[i] for i in cube_size_indices]

        # Store the sizes
        cube_sizes_for_creation = [
            torch.as_tensor(s, dtype=torch.float32, device=self.device)
            for s in cube_sizes_for_creation
        ]

        self.cube_half_sizes_per_env = torch.stack(cube_sizes_for_creation, dim=0)
        
        # create cubes from loaded sizes
        self._cubes = []
        for i in range(self.num_envs):
            cube_half_sizes = self.cube_half_sizes_per_env[i]
            cube_half_sizes = cube_half_sizes.cpu().numpy()
            
            cube = actors.build_box(
                self.scene,
                half_sizes=cube_half_sizes,
                color=[0.6, 0, 0, 1],
                name=f"cube_{i}",
                scene_idxs=[i],
                initial_pose=sapien.Pose(p=[-0.1, 0.0, cube_half_sizes[2]]),
            )
            self.remove_from_state_dict_registry(cube)
            self._cubes.append(cube)
        
        self.cube = Actor.merge(self._cubes, name="cube")
        self.add_to_state_dict_registry(self.cube)

        push_block_size_indices = self._batched_episode_rng.randint(0, high=len(self.push_block_variation_dimensions))
        self._push_blocks = []
        for i in range(self.num_envs):
            push_block_half_sizes = self.push_block_variation_dimensions[push_block_size_indices[i]]

            push_block = actors.build_box(
                self.scene,
                half_sizes=push_block_half_sizes,
                color=[0, 1, 0.3, 1],
                name=f"push_block_{i}",
                scene_idxs=[i],
                initial_pose=sapien.Pose(p=[0, 0.25, push_block_half_sizes[2]]),
            )
            self.remove_from_state_dict_registry(push_block)
            self._push_blocks.append(push_block)

        self.push_block = Actor.merge(self._push_blocks, name="push_block")
        self.add_to_state_dict_registry(self.push_block)

        # Store the half_sizes for each environment for later reference
        self.push_block_half_sizes_per_env = torch.tensor(
            [self.push_block_variation_dimensions[idx] for idx in push_block_size_indices],
            dtype=torch.float32,
            device=self.device
        )

        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[1, 0.3, 0.0, 0],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(p=[0, 0.25, 0.25]),
        )

        self.push_goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 0.4, 0.6, 0],
            name="push_goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(q=self.push_block_rot),
        )

        self._hidden_objects.append(self.goal_site)
        self._hidden_objects.append(self.push_goal_site)

        # object randomizations - only if difficulty is high enough
        if self.difficulty >= 3:

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


            for i, obj in enumerate(self.push_block._objs):
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

    def _load_intermediate_states(self, intermediate_states: list, env_idx: torch.Tensor = None):
        """
        Load intermediate states from Task 1 into the environment.
        
        Args:
            intermediate_states: List of state dicts (one per environment to load)
            env_idx: Optional tensor of environment indices. If None, assumes sequential [0, 1, 2, ...]
        """
        if env_idx is None:
            env_idx = torch.arange(len(intermediate_states), device=self.device)
        
        b = len(intermediate_states)

        # Extract states only for the environments we're resetting
        selected_states = [intermediate_states[i] for i in env_idx.cpu().numpy()]

        # Collect the pose data from the selected states
        cube_states = torch.stack([s["cube"] for s in selected_states], dim=0)
        goal_states = torch.stack([s["goal_site"] for s in selected_states], dim=0)
        robot_states_full = torch.stack([s["xarm7_leap_right"] for s in selected_states], dim=0)

        # Set the full state for the specified environments
        self.cube.set_state(state=cube_states, env_idx=env_idx)
        self.goal_site.set_state(state=goal_states, env_idx=env_idx)
        
        # Index manually with robot
        current_qpos = self.agent.robot.get_qpos()
        current_qvel = self.agent.robot.get_qvel()
        
        current_qpos[env_idx] = robot_states_full[..., 13:36]
        current_qvel[env_idx] = robot_states_full[..., 36:59]
        
        self.agent.robot.set_qpos(current_qpos)
        self.agent.robot.set_qvel(current_qvel)

        # Handle finger indices if present in the states
        has_active_fingers = 'active_finger_indices' in selected_states[0]
        
        if has_active_fingers:
            # Initialize storage tensors if not already created
            if not hasattr(self, 'active_finger_indices'):
                first_active = selected_states[0]['active_finger_indices']
                num_active_fingers = first_active.shape[0] if first_active.dim() > 0 else 1
                num_inactive_fingers = 4 - num_active_fingers

                self.active_finger_indices = torch.zeros(
                    (self.num_envs, num_active_fingers),
                    dtype=torch.long,
                    device=self.device
                )
                self.inactive_finger_indices = torch.zeros(
                    (self.num_envs, num_inactive_fingers),
                    dtype=torch.long, 
                    device=self.device
                )
            
            # Load finger indices for the specified environments
            for i, state in enumerate(selected_states):
                env_id = env_idx[i]
                self.active_finger_indices[env_id] = state['active_finger_indices']
                self.inactive_finger_indices[env_id] = state['inactive_finger_indices']
        else:
            # Default finger configuration if not specified
            if not hasattr(self, 'active_finger_indices'):
                self.active_finger_indices = torch.tensor(
                    DEFAULT_ACTIVE_FINGER_INDICES, device=self.device
                ).expand(self.num_envs, -1)
                self.inactive_finger_indices = torch.tensor(
                    DEFAULT_INACTIVE_FINGER_INDICES, device=self.device
                ).expand(self.num_envs, -1)


    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            current_push_block_half_sizes = self.push_block_half_sizes_per_env[env_idx]
            
            if self.intermediate_states:
                self._load_intermediate_states(self.intermediate_states, env_idx)

            # push block
            push_xyz = torch.zeros((b, 3))
            random_xy = self._batched_episode_rng.uniform(
                low=-self.cube_spawn_half_size,
                high=self.cube_spawn_half_size,
                size=(2,)
            )
            push_xyz[:, :2] = torch.from_numpy(random_xy).float().to(self.device)
            push_xyz[:, 0] += self.push_block_spawn_center[0]
            push_xyz[:, 1] += self.push_block_spawn_center[1]

            push_xyz[:, 2] = current_push_block_half_sizes[:, 2]

            # random rotation for push block and its goal
            push_block_rot = random_quaternions_batched(self._batched_episode_rng, device=self.device, lock_x=True, lock_y=True)
            self.push_block.set_pose(Pose.create_from_pq(push_xyz, push_block_rot))

            push_goal_xyz = torch.zeros((b, 3))
            random_xy = self._batched_episode_rng.uniform(
                low=-self.cube_spawn_half_size,
                high=self.cube_spawn_half_size,
                size=(2,)
            )
            push_goal_xyz[:, :2] = torch.from_numpy(random_xy).float().to(self.device)
            push_goal_xyz[:, 0] += self.push_block_spawn_center[0]
            # Move goal position to middle of the table
            push_goal_xyz[:, 1] += self.push_block_spawn_center[1]
            # Goal height = table height + object height/2
            push_goal_xyz[:, 2] = self.max_goal_height
            self.push_goal_site.set_pose(Pose.create_from_pq(push_goal_xyz, push_block_rot))

    def _get_obs_extra(self, info: Dict):
        # in reality some people hack is_grasped into observations by checking if the gripper can close fully or not
        obs = dict(
            # is_grasped=info["is_grasped"],
            push_goal_pos=self.push_goal_site.pose.p,
            cube_pose=self.cube.pose.raw_pose,
            push_block_pose=self.push_block.pose.raw_pose,
            cube_half_sizes=self.cube_half_sizes_per_env,
            push_block_half_sizes=self.push_block_half_sizes_per_env,
            picking_fingers=self.active_finger_indices,
        )
        if "state" in self.obs_mode:
            
            obs.update(
                tcp_to_obj_pos=self.push_block.pose.p - self.agent.tcp.pose.p,
                obj_to_goal_pos=self.push_goal_site.pose.p - self.push_block.pose.p,
            )

        return obs


    def evaluate(self):

        # 1. Finger proximity reward (for picking block)
        finger_object_dist = torch.linalg.norm(
            self.agent.tip_poses[..., :3] - self.cube.pose.p[:, None, :], dim=-1
        )  # shape: (B, num_fingers)

        # Only for pick block do the fingers have to close
        active_finger_dist = torch.gather(finger_object_dist, dim=1, index=self.active_finger_indices)
        finger_close_fraction = (active_finger_dist < 0.07).float().mean(dim=-1)
        is_obj_grasped = (finger_close_fraction > 0.5)

        is_obj_placed = (
            torch.linalg.norm(self.push_goal_site.pose.p - self.push_block.pose.p, axis=1)
            <= self.goal_thresh
        )

        # 2. Check cube-table contact
        total_force = torch.zeros(self.num_envs, device=self.device)
        contacts = self.scene.get_pairwise_contact_forces(self.cube, self.table_scene.table)
        if contacts is not None and len(contacts) > 0:
                total_force += contacts.norm(dim=-1)

        has_collision = total_force > 0.01

        is_robot_static = self.agent.is_static(0.2)
        result = {
            "success": is_obj_grasped & is_obj_placed & ~has_collision,
            "is_obj_grasped": is_obj_grasped,
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
            # "is_grasped": is_grasped,
        }

        num_inactive_fingers = self.inactive_finger_indices.shape[1]
        if num_inactive_fingers >= 1:
            result["inactive_finger1"] = self.inactive_finger_indices[:, 0]
        if num_inactive_fingers >= 2:
            result["inactive_finger2"] = self.inactive_finger_indices[:, 1]
        if num_inactive_fingers >= 3:
            result["inactive_finger3"] = self.inactive_finger_indices[:, 2]

        return result

    def compute_inactive_finger_penalty(self):
        """
        Compute penalty for contact forces between inactive fingers and the cube.
        
        Returns:
            torch.Tensor: Penalty for each environment, shape (num_envs,)
        """

        # Get contact forces for all fingers at once
        all_finger_forces = torch.zeros(self.num_envs, 4, device=self.device)  # (num_envs, 4 fingers)

        for finger_idx in range(4):
            # Tip link
            tip_link = self.agent.tip_links[finger_idx]
            contacts = self.scene.get_pairwise_contact_forces(tip_link, self.cube)
            if contacts is not None and len(contacts) > 0:
                all_finger_forces[:, finger_idx] += contacts.norm(dim=-1)
            
            # Front link (DIP)
            front_link = self.agent.hand_front_links[finger_idx]
            contacts = self.scene.get_pairwise_contact_forces(front_link, self.cube)
            if contacts is not None and len(contacts) > 0:
                all_finger_forces[:, finger_idx] += contacts.norm(dim=-1)
            
            # PIP link
            pip_link = self.agent.pip_links[finger_idx]
            contacts = self.scene.get_pairwise_contact_forces(pip_link, self.cube)
            if contacts is not None and len(contacts) > 0:
                all_finger_forces[:, finger_idx] += contacts.norm(dim=-1)

            # MCP link
            if finger_idx < 3:
                mcp_link = self.agent.mcp_links[finger_idx]
                contacts = self.scene.get_pairwise_contact_forces(mcp_link, self.cube)
                if contacts is not None and len(contacts) > 0:
                    all_finger_forces[:, finger_idx] += contacts.norm(dim=-1)

        # Create mask for inactive fingers (shape: num_envs, 4)
        inactive_mask = torch.ones(self.num_envs, 4, dtype=torch.bool, device=self.device)
        inactive_mask.scatter_(1, self.active_finger_indices, False)

        # Apply mask and sum inactive finger forces
        inactive_finger_force = (all_finger_forces * inactive_mask).sum(dim=1)

        return torch.clamp((-1.0 * inactive_finger_force), min=-1.0, max=0.0)


    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):

        # # 1. Finger proximity reward (for picking block)
        finger_object_dist = torch.linalg.norm(
            self.agent.tip_poses[..., :3] - self.cube.pose.p[:, None, :], dim=-1
        )  # shape: (B, num_fingers)
        active_finger_dist = torch.gather(finger_object_dist, dim=1, index=self.active_finger_indices)
        grasp_reward = torch.exp(-20 * active_finger_dist).mean(dim=-1)
        grasp_fraction = (active_finger_dist < 0.05).float().mean(dim=-1)

        # Penalize inactive finger contacts
        inactive_finger_penalty = self.compute_inactive_finger_penalty()

        # 2. TCP to object (reaching reward)
        tcp_to_obj_dist = torch.linalg.norm(self.push_block.pose.p - self.agent.tcp_pose.p, dim=-1)
        reaching_reward = 1.0 - torch.tanh(5.0 * tcp_to_obj_dist) 

        # 3. Finger proximity reward (soft reward instead of binary)
        finger_object_dist = torch.linalg.norm(
            self.agent.tip_poses[..., :3] - self.push_block.pose.p[:, None, :], dim=-1
        )  # shape: (B, num_fingers)

        # Select only the inactive fingers for the push block using indexing
        inactive_finger_dist = torch.gather(finger_object_dist, dim=1, index=self.inactive_finger_indices) 
        finger_dist_reward = torch.exp(-20 * inactive_finger_dist)

        # Palm proximity to push block
        palm_object_dist = torch.linalg.norm(
            self.agent.palm_contact_points[...,:3] - self.push_block.pose.p[:, None, :], dim=-1
        )  # shape: (B, num_palm_points)
        palm_min_dist = torch.min(palm_object_dist, dim=-1)[0]  # shape: (B,)
        palm_dist_reward = torch.exp(-20 * palm_min_dist)
        # palm_dist_reward = 0

        # Combine inactive fingers + palm for push block grasping
        num_push_contacts = self.inactive_finger_indices.shape[1] + 2  # inactive fingers + palm (counts as 2)
        combined_push_dist_reward = (torch.sum(finger_dist_reward, dim=-1) + palm_dist_reward * 2) / num_push_contacts

        finger_close_fraction  = (inactive_finger_dist < 0.05).float().mean(dim=-1)
        push_block_grasp_reward = finger_close_fraction

        # 4. Object to goal (only meaningful if object is lifted/moved) + height
        obj_to_goal_dist = torch.linalg.norm(self.push_goal_site.pose.p - self.push_block.pose.p, axis=1)
        place_reward = (1.0 - torch.tanh(5.0 * obj_to_goal_dist))

        # Reward for lifting relative to initial height, capped at +0.1 m
        max_lift_bonus_height = 0.1 
        push_block_height = self.push_block.pose.p[:, 2]
        initial_height = self.push_block_half_sizes_per_env[:, 2]
        height_reward = torch.clamp((push_block_height - initial_height) / max_lift_bonus_height, min=0.0, max=1.0) * grasp_fraction


        # 5. Penalize joint velocity to reduce jerky motion
        qvel = self.agent.robot.get_qvel()
        if self.robot_uids in ["panda", "widowxai"]:
            qvel = qvel[..., :-2]
        elif self.robot_uids == "so100":
            qvel = qvel[..., :-1]
        elif self.robot_uids == "xarm7_leap_right":
            qvel = qvel[..., :-16]
        static_penalty = -0.1 * torch.linalg.norm(qvel, axis=1)  # small penalty

        # 6 Contact forces between hand and table/obstacle penalized:
        total_force = torch.zeros(self.num_envs, device=self.device)
        for link in self.agent.robot.get_links():
            table_contacts = self.scene.get_pairwise_contact_forces(link, self.table_scene.table)
            if table_contacts is not None and len(table_contacts) > 0:
                total_force += table_contacts.norm(dim=-1)

        # penalize between cube and table as well
        contacts = self.scene.get_pairwise_contact_forces(self.cube, self.table_scene.table)
        if contacts is not None and len(contacts) > 0:
                total_force += contacts.norm(dim=-1)

        collision_penalty = -0.05 * total_force

        # 7. Final reward
        reward = (
            self.reward_type["grasp_reward"] * grasp_reward +
            self.reward_type["reach_weight"] * reaching_reward +
            self.reward_type["finger_dist_weight"] * combined_push_dist_reward +
            self.reward_type["push_block_grasp_weight"] * push_block_grasp_reward +
            self.reward_type["inactive_finger_penalty_weight"] * inactive_finger_penalty +
            self.reward_type["height_weight"] * height_reward + 
            self.reward_type["place_weight"] * place_reward + 
            self.reward_type["static_penalty_weight"] * static_penalty +
            self.reward_type["collision_penalty_weight"] * collision_penalty
        )

        # Bonus for success
        reward[info["success"]] += 5.0
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        
        sum_rewards = 5.0
        for key in ["grasp_reward", "reach_weight", "finger_dist_weight", "push_block_grasp_weight", "height_weight", "place_weight"]:
            sum_rewards += self.reward_type[key]

        return self.compute_dense_reward(obs=obs, action=action, info=info) / sum_rewards