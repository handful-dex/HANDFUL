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

from envs.config.xarm7_allegro_config import (PICK_CUBE_CONFIGS, PICK_REWARD_CONFIGS, PICK_PRESS_CONFIGS, PICK_PRESS_REWARD_CONFIGS,
                                            MASS_VARIATIONS, FRICTION_VARIATIONS, RESTITUTION_VARIATIONS)
from envs.utils.batched_pose import random_quaternions_batched
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors, articulations
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Actor, Pose, Link
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.utils.geometry.rotation_conversions import quaternion_multiply

import sapien.physx as physx
from sapien.physx import PhysxRigidBodyComponent
from sapien.render import RenderBodyComponent

MAX_EPISODE_STEPS = 175
N_MATERIAL_POOL = 1024

PICK_CUBE_DOC_STRING = """**Task Description:**
TBD
"""

def _build_box_with_hole(scene: ManiSkillScene, inner_radius, outer_radius, depth, center=(0, 0)):
    
    builder = scene.create_actor_builder()
    thickness = (outer_radius - inner_radius) * 0.5

    # z-axis is hole direction 
    half_center = [x * 0.5 for x in center]
    half_sizes = [
        [thickness - half_center[0], outer_radius, depth], 
        [thickness + half_center[0], outer_radius, depth],
        [outer_radius, thickness - half_center[1], depth], 
        [outer_radius, thickness + half_center[1], depth],
    ]
    offset = thickness + inner_radius
    poses = [
        sapien.Pose([offset + half_center[0], 0, 0]), 
        sapien.Pose([-offset + half_center[0], 0, 0]),
        sapien.Pose([0, offset + half_center[1], 0]),
        sapien.Pose([0, -offset + half_center[1], 0]),
    ]
    mat = sapien.render.RenderMaterial(
        base_color=[1, 0.82, 0.54, 1], roughness=0.5, specular=0.5
    )
    for half_size, pose in zip(half_sizes, poses):
        builder.add_box_collision(pose, half_size)
        builder.add_box_visual(pose, half_size, material=mat)
    return builder

@register_env("xArm7-v1-press-button-unified", max_episode_steps=MAX_EPISODE_STEPS)
class XArm7TableTopPressUnified(BaseEnv):
    """
    Unified environment combining two sequential picking tasks:
    Phase 1 (0-74 steps): Pick and place first cube to goal
    Phase 2 (75-174 steps): Press button 

    Rewards are automatically switched based on current phase.
    """

    SUPPORTED_ROBOTS = [
        "xarm7_ability", "xarm6_robotiq", "xarm7_allegro_right", "xarm7_leap_right", "franka_allegro_right"
    ]
    agent: Union[XArm7Ability, XArm6Robotiq, XArm7Allegro, XArm7Leap, FrankaAllegro]

    def __init__(self, *args, **kwargs):

        known_custom_args = ["robot_uids", "reward_type", "robot_init_qpos_noise",
                             "finger_selection", "palm_use", "num_active_fingers", "difficulty"]
        for arg in known_custom_args:
            setattr(self, arg, kwargs.pop(arg, None))

        self.phase1_steps = 75

        self.robot_uids = "xarm7_leap_right" if self.robot_uids is None else self.robot_uids
        self.robot_init_qpos_noise = 0.02 if self.robot_init_qpos_noise is None else self.robot_init_qpos_noise
        self.pick_reward = PICK_REWARD_CONFIGS["sac"] if self.reward_type is None else PICK_REWARD_CONFIGS[self.reward_type]
        self.press_reward = PICK_PRESS_REWARD_CONFIGS["sac"] if self.reward_type is None else PICK_PRESS_REWARD_CONFIGS[self.reward_type]
        self.difficulty = 3 if self.difficulty is None else self.difficulty

        self.finger_selection = "fixed" if self.finger_selection is None else self.finger_selection
        self.palm_use = False if self.palm_use is None else self.palm_use
        self.num_active_fingers = 2 if self.num_active_fingers is None else self.num_active_fingers

        # Pick cube configs — cube spawning, goal, camera
        if self.robot_uids in PICK_CUBE_CONFIGS:
            pick_cfg = PICK_CUBE_CONFIGS[self.robot_uids]
        else:
            pick_cfg = PICK_CUBE_CONFIGS["xarm7_leap_right"]

        if self.robot_uids in PICK_PRESS_CONFIGS:
            press_cfg = PICK_PRESS_CONFIGS[self.robot_uids]
        else:
            press_cfg = PICK_PRESS_CONFIGS["panda"]

        self.cube_variation_dimensions = pick_cfg["cube_variation_dimensions"]
        self.goal_thresh = pick_cfg["goal_thresh"]
        self.cube_spawn_half_size = pick_cfg["cube_spawn_half_size"]
        self.cube_spawn_center = pick_cfg["cube_spawn_center"]
        self.max_cube_goal_height = pick_cfg["max_goal_height"]

        self.sensor_cam_eye_pos = press_cfg["sensor_cam_eye_pos"]
        self.sensor_cam_target_pos = press_cfg["sensor_cam_target_pos"]
        self.human_cam_eye_pos = press_cfg["human_cam_eye_pos"]
        self.human_cam_target_pos = press_cfg["human_cam_target_pos"]

        # From press config
        self.press_goal_thresh = press_cfg["goal_thresh"]
        self.push_block_spawn_half_size = press_cfg["push_block_spawn_half_size"]
        self.push_block_spawn_rot_noise = press_cfg["push_block_spawn_rot_noise"]
        self.push_block_spawn_center = press_cfg["push_block_spawn_center"]
        self.push_block_half_sizes = press_cfg["push_block_half_sizes"]
        self.push_block_rot = press_cfg["push_block_rot"]
        self.box_with_hole_radii_and_depth = press_cfg["box_with_hole_radii_and_depth"]

        if self.difficulty <= 1:
            self.cube_spawn_half_size = 0
            self.push_block_spawn_half_size = 0
            self.push_block_spawn_rot_noise = 0
        elif self.difficulty == 2:
            self.cube_spawn_half_size /= 2
            self.push_block_spawn_half_size /= 2
            self.push_block_spawn_rot_noise /= 2
        else:
            self.cube_spawn_half_size = self.cube_spawn_half_size
            self.push_block_spawn_half_size = self.push_block_spawn_half_size
            self.push_block_spawn_rot_noise = self.push_block_spawn_rot_noise



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

        # sample a handful of different cube sizes and push block sizes
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

        self.push_block = actors.build_box(
            self.scene,
            half_sizes=self.push_block_half_sizes,
            color=[0, 0, 1, 1],
            name="push_block",
            body_type="kinematic",
            initial_pose=sapien.Pose(p=self.push_block_spawn_center, q=self.push_block_rot),
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

        boxes = []

        for i in range(self.num_envs):
            scene_idxs = [i]
            builder = _build_box_with_hole(
                self.scene, 
                self.box_with_hole_radii_and_depth[0], 
                self.box_with_hole_radii_and_depth[1], 
                self.box_with_hole_radii_and_depth[2],
            )
            builder.initial_pose = sapien.Pose(p=self.push_block_spawn_center, q=self.push_block_rot)
            builder.set_scene_idxs(scene_idxs)
            box = builder.build_kinematic(f"box_with_hole_{i}")
            self.remove_from_state_dict_registry(box)
            boxes.append(box)

        self.box = Actor.merge(boxes, "box_with_hole")
        self.add_to_state_dict_registry(self.box)

        self._hidden_objects.append(self.goal_site)

        # object randomizations
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

            # robot randomizations
            for link in self.agent.robot.links:
                for i, obj in enumerate(link._objs):
                    rb: PhysxRigidBodyComponent = obj.entity.find_component_by_type(PhysxRigidBodyComponent)
                    if rb is not None:

                        rb.mass *= self._batched_episode_rng[i].uniform(low=MASS_VARIATIONS[0], high=MASS_VARIATIONS[1])
                        mat = self._material_pool[self._batched_episode_rng[i].randint(0, N_MATERIAL_POOL)]

                        for shape in rb.collision_shapes:
                            shape.physical_material = mat


    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):

            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            current_cube_half_sizes = self.cube_half_sizes_per_env[env_idx]

            # pick cube
            xyz = torch.zeros((b, 3), device=self.device)
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
            goal_xyz = torch.zeros((b, 3), device=self.device)
            random_goal_xy = self._batched_episode_rng.uniform(
                low=-self.cube_spawn_half_size, 
                high=self.cube_spawn_half_size, 
                size=(2,)
            )
            goal_xyz[:, :2] = torch.from_numpy(random_goal_xy).float().to(self.device)
            
            goal_xyz[:, 0] += self.cube_spawn_center[0]
            goal_xyz[:, 1] += self.cube_spawn_center[1]
            goal_xyz[:, 2] = self.max_cube_goal_height
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

            # push block
            push_xyz = torch.zeros((b, 3), device=self.device)
            random_yz = self._batched_episode_rng.uniform(
                low=-self.push_block_spawn_half_size,
                high=self.push_block_spawn_half_size,
                size=(2,)
            )
            random_yz = torch.from_numpy(random_yz).float().to(self.device)

            # assign y and z
            push_xyz[:, 0] = self.push_block_spawn_center[0]
            push_xyz[:, 1] = self.push_block_spawn_center[1] + random_yz[:, 0]
            push_xyz[:, 2] = self.push_block_spawn_center[2] + random_yz[:, 1]


            random_q = random_quaternions_batched(self._batched_episode_rng,
                                        device=self.device, lock_x=True, lock_y=True,
                                        bounds=(-self.push_block_spawn_rot_noise, self.push_block_spawn_rot_noise))

            base_q = torch.tensor(self.push_block_rot, device=self.device, dtype=torch.float32)
            base_q_batched = base_q.unsqueeze(0).expand(b, -1)
            final_q = quaternion_multiply(random_q, base_q_batched)

            self.push_block.set_pose(Pose.create_from_pq(push_xyz, final_q))
            self.box.set_pose(Pose.create_from_pq(push_xyz, final_q))

            # Finger selection
            if self.finger_selection == "random":
                all_fingers = torch.arange(4, device=self.device).unsqueeze(0).expand(b, -1)
                rand_values = self._batched_episode_rng.uniform(0, 1, size=(b, 4))
                rand_values = torch.from_numpy(rand_values).float().to(self.device)
                rand_perm = rand_values.argsort(dim=1)
                shuffled_fingers = torch.gather(all_fingers, 1, rand_perm)
                self.num_active_fingers = 2

            elif self.finger_selection == "fixed":
                shuffled_fingers = torch.tensor([3,2,0,1], device=self.device).unsqueeze(0).expand(b, -1)
                self.palm_use = False
                self.num_active_fingers = 2

            elif isinstance(self.finger_selection, (list, tuple)):
                shuffled_fingers = torch.tensor(self.finger_selection, device=self.device).unsqueeze(0).expand(b, -1)

            self.active_finger_indices = shuffled_fingers[:, :self.num_active_fingers].sort(dim=1)[0]
            self.inactive_finger_indices = shuffled_fingers[:, self.num_active_fingers:].sort(dim=1)[0]

            if self.palm_use:
                qpos = self.agent.keyframes["rest_palm"].qpos
            else:
                qpos = self.agent.keyframes["rest"].qpos

            qpos = (self._batched_episode_rng.normal(0, self.robot_init_qpos_noise, (len(qpos))) + qpos)
            self.agent.reset(init_qpos=qpos)

    
    def _get_current_phase(self):
        """Determine which phase each environment is in"""
        return (self._elapsed_steps < self.phase1_steps).long()

    def _get_obs_extra(self, info: Dict):

        phase = self._get_current_phase()

        obs = dict(
            # is_grasped=info["is_grasped"],
            goal_pos=self.goal_site.pose.p,
            cube_pose=self.cube.pose.raw_pose,
            cube_half_sizes=self.cube_half_sizes_per_env,
            push_block_pose=self.push_block.pose.raw_pose,
            picking_fingers=self.active_finger_indices,
            phase=phase,
        )
        if "state" in self.obs_mode:
            
            obs.update(
                tcp_to_cube_pos=self.cube.pose.p - self.agent.tcp.pose.p,
                cube_to_goal_pos=self.goal_site.pose.p - self.cube.pose.p,
            )

        return obs

    def get_split_observation(self, info: Dict):

        phase = self._get_current_phase()

        if phase:
            extra = dict(
                # is_grasped=info["is_grasped"],
                goal_pos=self.goal_site.pose.p,
                cube_pose=self.cube.pose.raw_pose,
                cube_half_sizes=self.cube_half_sizes_per_env,
                picking_fingers=self.active_finger_indices,
            )
            if "state" in self.obs_mode:
                extra.update(
                    tcp_to_obj_pos=self.cube.pose.p - self.agent.tcp.pose.p,
                    obj_to_goal_pos=self.goal_site.pose.p - self.cube.pose.p,
                )
        else:
            extra = dict(
                # is_grasped=info["is_grasped"],
                cube_pose=self.cube.pose.raw_pose,
                cube_half_sizes=self.cube_half_sizes_per_env,
                push_block_pose=self.push_block.pose.raw_pose,
                picking_fingers=self.active_finger_indices,
            )
            if "state" in self.obs_mode:
                
                extra.update(
                    tcp_to_obj_pos=self.cube.pose.p - self.agent.tcp.pose.p,
                )

        
        obs = {
            "agent": self.agent.get_proprioception(),
            "extra": extra,
            }
                
        return obs

    def evaluate(self):

        # 1. Finger proximity reward (for picking block)
        finger_object_dist = torch.linalg.norm(
            self.agent.tip_poses[..., :3] - self.cube.pose.p[:, None, :], dim=-1
        )

        # Only for pick block do the fingers have to close
        active_finger_dist = torch.gather(finger_object_dist, dim=1, index=self.active_finger_indices)
        finger_close_fraction = (active_finger_dist < 0.07).float().mean(dim=-1)
        is_obj_grasped = (finger_close_fraction > 0.5)


        finger_to_obj_dist = torch.linalg.norm(self.push_block.pose.p[:, None, :] - self.agent.tip_poses[..., :3], dim=-1)
        min_finger_dist = torch.min(finger_to_obj_dist, dim=-1)[0]  # (B,)
        is_button_pressed = (min_finger_dist < self.press_goal_thresh)

        # 2. Check cube-table contact and link-block contact
        total_force = torch.zeros(self.num_envs, device=self.device)
        for link in self.agent.robot.get_links():

            obstacle_contacts = self.scene.get_pairwise_contact_forces(link, self.box)
            if obstacle_contacts is not None and len(obstacle_contacts) > 0:
                total_force += obstacle_contacts.norm(dim=-1)
        
        contacts = self.scene.get_pairwise_contact_forces(self.cube, self.table_scene.table)
        if contacts is not None and len(contacts) > 0:
                total_force += contacts.norm(dim=-1)

        has_collision = total_force > 0.01

        # is_grasped = self.agent.is_grasping(self.cube)
        is_robot_static = self.agent.is_static(0.2)
        result = {
            "success": is_obj_grasped & is_button_pressed & ~has_collision,
            "is_obj_grasped": is_obj_grasped,
            "is_button_pressed": is_button_pressed,
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

        return torch.clamp((-1.0 * inactive_finger_force), min=-1.5, max=0.0)

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict): 

        phase = self._get_current_phase()

        phase1_reward = self._compute_phase1_reward(obs, action, info)
        phase2_reward = self._compute_phase2_reward(obs, action, info)

        reward = torch.where(phase == 1, phase1_reward, phase2_reward)
        return reward


    def _compute_phase1_reward(self, obs, action, info):
        
        """

        Phase 1: Pick first cube with active fingers and place at goal.

        """
        
        # 1. TCP to object (reaching reward)
        tcp_to_obj_dist = torch.linalg.norm(self.cube.pose.p - self.agent.tcp_pose.p, dim=-1)
        reaching_reward = 1.0 - torch.tanh(5.0 * tcp_to_obj_dist) 

        # 2. Finger proximity reward (soft reward instead of binary)
        finger_object_dist = torch.linalg.norm(
            self.agent.tip_poses[..., :3] - self.cube.pose.p[:, None, :], dim=-1
        )  # shape: (B, num_fingers)

        # Select only the active fingers using indexing
        active_finger_dist = torch.gather(finger_object_dist, dim=1, index=self.active_finger_indices) 
        finger_dist_reward = torch.exp(-20 * active_finger_dist)

        # Palm contacts
        palm_object_dist = torch.linalg.norm(
            self.agent.palm_contact_points[..., :3] - self.cube.pose.p[:, None, :], dim=-1
        )  # shape: (B, num_palm_points)
        palm_min_dist = torch.min(palm_object_dist, dim=-1)[0]  # shape: (B,)
        palm_dist_reward = torch.exp(-20 * palm_min_dist)
        palm_close = (palm_min_dist < 0.05).float()
        num_active_contacts = self.active_finger_indices.shape[1] + 1  # fingers + palm
        
        combined_dist_reward = (torch.sum(finger_dist_reward, dim=-1) + palm_dist_reward) / num_active_contacts
        finger_close_fraction = (active_finger_dist < 0.05).float().mean(dim=-1)
        grasp_reward = (finger_close_fraction * self.active_finger_indices.shape[1] + palm_close) / num_active_contacts
                
        # Penalize inactive finger contacts
        inactive_finger_penalty = self.compute_inactive_finger_penalty()

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
            self.pick_reward["reach_weight"] * reaching_reward +
            self.pick_reward["finger_dist_weight"] * combined_dist_reward +
            self.pick_reward["grasp_weight"] * grasp_reward +
            self.pick_reward["inactive_finger_penalty_weight"] * inactive_finger_penalty +
            self.pick_reward["place_weight"] * place_reward +
            self.pick_reward["static_penalty_weight"] * static_penalty +
            self.pick_reward["collision_penalty_weight"] * collision_penalty
        )

        return reward

    def _compute_phase2_reward(self, obs, action, info):

        # # 1. Finger proximity reward (for picking block)
        finger_object_dist = torch.linalg.norm(
            self.agent.tip_poses[..., :3] - self.cube.pose.p[:, None, :], dim=-1
        )  # shape: (B, num_fingers)
        active_finger_dist = torch.gather(finger_object_dist, dim=1, index=self.active_finger_indices)
        grasp_reward = torch.exp(-20 * active_finger_dist).mean(dim=-1)
        grasp_fraction = (active_finger_dist < 0.05).float().mean(dim=-1)

        # Penalize inactive finger contacts
        inactive_finger_penalty = self.compute_inactive_finger_penalty()

        # 2. Press Finger to Button object (reaching reward)
        finger_to_obj_dist = torch.linalg.norm(self.push_block.pose.p[:, None, :] - self.agent.tip_poses[..., :3], dim=-1)
        inactive_finger_dist = torch.gather(finger_to_obj_dist, dim=1, index=self.inactive_finger_indices)
        min_finger_dist = torch.min(inactive_finger_dist, dim=-1)[0]
        reaching_reward = (1.0 - torch.tanh(5.0 * min_finger_dist))

        # 3. Penalize joint velocity to reduce jerky motion
        qvel = self.agent.robot.get_qvel()
        if self.robot_uids in ["panda", "widowxai"]:
            qvel = qvel[..., :-2]
        elif self.robot_uids == "so100":
            qvel = qvel[..., :-1]
        elif self.robot_uids == "xarm7_leap_right":
            qvel = qvel[..., :-16]
        static_penalty = -0.1 * torch.linalg.norm(qvel, axis=1)  # small penalty

        # 4 Contact forces between hand and table/obstacle penalized:
        total_force = torch.zeros(self.num_envs, device=self.device)
        for link in self.agent.robot.get_links():
            table_contacts = self.scene.get_pairwise_contact_forces(link, self.table_scene.table)
            if table_contacts is not None and len(table_contacts) > 0:
                total_force += table_contacts.norm(dim=-1)

            obstacle_contacts = self.scene.get_pairwise_contact_forces(link, self.box)
            if obstacle_contacts is not None and len(obstacle_contacts) > 0:
                total_force += obstacle_contacts.norm(dim=-1)
        
        # penalize between cube and table as well
        contacts = self.scene.get_pairwise_contact_forces(self.cube, self.table_scene.table)
        if contacts is not None and len(contacts) > 0:
                total_force += contacts.norm(dim=-1)

        collision_penalty = - 0.05 * total_force

        # 6. Final reward
        reward = (
            self.press_reward["reach_weight"] * reaching_reward +
            self.press_reward["grasp_weight"] * grasp_reward +
            self.press_reward["inactive_finger_penalty_weight"] * inactive_finger_penalty +
            self.press_reward["static_penalty_weight"] * static_penalty +
            self.press_reward["collision_penalty_weight"] * collision_penalty
        )

        # Bonus for success
        reward[info["success"]] += 5.0
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        
        phase = self._get_current_phase()
        
        # Calculate normalization constants for each phase
        sum_phase1 = sum(self.pick_reward[k] for k in 
                        ["reach_weight", "finger_dist_weight", "grasp_weight", "place_weight"])
        sum_phase2 = 12.0
        
        raw_reward = self.compute_dense_reward(obs=obs, action=action, info=info)
        
        # Hard split normalization based on phase
        normalization = torch.where(
            phase == 1, 
            torch.tensor(sum_phase1, device=self.device),
            torch.tensor(sum_phase2, device=self.device)
        )
        
        return raw_reward / normalization
