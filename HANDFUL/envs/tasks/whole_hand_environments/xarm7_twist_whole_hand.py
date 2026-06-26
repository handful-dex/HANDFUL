from typing import Any, Dict, Union, List

import numpy as np
import sapien
import torch

from mani_skill.agents.robots import XArm7Ability, XArm6Robotiq
from agents.xarm7_allegro import XArm7Allegro
from agents.xarm7_leap import XArm7Leap
from agents.franka_allegro import FrankaAllegro
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.envs.sapien_env import BaseEnv

from envs.config.xarm7_allegro_config import (KNOB_TWIST_CONFIGS, KNOB_TWIST_REWARD_CONFIGS,
                                             MASS_VARIATIONS, FRICTION_VARIATIONS, RESTITUTION_VARIATIONS)
from envs.utils.batched_pose import random_quaternions_batched
from envs.utils.robel import build_robel_valve
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors, articulations
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Actor, Pose
from mani_skill.utils.structs.articulation import Articulation
from mani_skill.utils.structs.link import Link

from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.utils.geometry.rotation_conversions import quaternion_multiply, quaternion_invert, standardize_quaternion, axis_angle_to_quaternion

import sapien.physx as physx
from sapien.physx import PhysxRigidBodyComponent
from sapien.render import RenderBodyComponent

DEFAULT_ACTIVE_FINGER_INDICES = [[0, 1, 2, 3]]
DEFAULT_INACTIVE_FINGER_INDICES = [[0, 1, 2, 3]]

KNOB_TWIST_DOC_STRING = """**Task Description:**
Grasp and twist a valve/knob using specific fingers while keeping other fingers from interfering.
"""

MAX_EPISODE_STEPS = 100
N_MATERIAL_POOL = 1024

WALL_HALF_SIZES = [0.20, 0.20, 0.02]   # [width, depth, height]
WALL_Y_OFFSET = WALL_HALF_SIZES[1]



@register_env("xArm7-v1-knob-twist-whole-hand", max_episode_steps=MAX_EPISODE_STEPS)
class XArm7TableTopKnobTwistWholeHand(BaseEnv):

    SUPPORTED_ROBOTS = [
        "xarm7_ability", "xarm6_robotiq", "xarm7_allegro_right", "xarm7_leap_right"
    ]
    agent: Union[XArm7Ability, XArm6Robotiq, XArm7Allegro, XArm7Leap]

    def __init__(self, *args, state_file_path: str = "",  **kwargs):
        
        known_custom_args = ["robot_uids", "reward_type", "robot_init_qpos_noise", "difficulty"]

        # push_env args
        for arg in known_custom_args:
            setattr(self, arg, kwargs.pop(arg, None))

        self.robot_uids = "xarm7_leap_right" if self.robot_uids is None else self.robot_uids
        self.robot_init_qpos_noise = 0.02 if self.robot_init_qpos_noise is None else self.robot_init_qpos_noise
        self.reward_type = KNOB_TWIST_REWARD_CONFIGS["sac"] if self.reward_type is None else KNOB_TWIST_REWARD_CONFIGS[self.reward_type]
        self.difficulty = 3 if self.difficulty is None else self.difficulty

        if self.robot_uids in KNOB_TWIST_CONFIGS:
            cfg = KNOB_TWIST_CONFIGS[self.robot_uids]
        else:
            cfg = KNOB_TWIST_CONFIGS["xarm7_leap_right"]
        
        self.goal_thresh = cfg["goal_thresh"]
        self.cube_variation_dimensions = cfg["cube_variation_dimensions"]

        self.valve_spawn_pos = cfg["valve_spawn_pos"]
        self.valve_spawn_noise = cfg["valve_spawn_noise"]
        
        self.sensor_cam_eye_pos = cfg["sensor_cam_eye_pos"]
        self.sensor_cam_target_pos = cfg["sensor_cam_target_pos"]
        self.human_cam_eye_pos = cfg["human_cam_eye_pos"]
        self.human_cam_target_pos = cfg["human_cam_target_pos"]

        # Task-specific thresholds based on difficulty
        
        self.success_threshold = 4/3*torch.pi
        
        if self.difficulty <= 1:
            self.valve_spawn_noise = 0
        elif self.difficulty >= 3:
            self.valve_spawn_noise = self.valve_spawn_noise
        else:
            self.valve_spawn_noise /= 2

        self.capsule_offset = 0.01
        self.residual_scale = 0.05

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
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
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
                color=[1, 0, 0, 1],
                name=f"cube_{i}",
                scene_idxs=[i],
                initial_pose=sapien.Pose(p=[0, 0.0, cube_half_sizes[2]]),
            )
            self.remove_from_state_dict_registry(cube)
            self._cubes.append(cube)
        
        self.cube = Actor.merge(self._cubes, name="cube")
        self.add_to_state_dict_registry(self.cube)
        
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[1, 0.3, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(p=[0, 0.0, 0.0]),
        )
        self._hidden_objects.append(self.goal_site)
        
        self.wall = actors.build_box(
            self.scene,
            half_sizes=WALL_HALF_SIZES,
            color=[0.5, 0.5, 0.5, 1],
            name="wall_backplate",
            body_type="kinematic",
            initial_pose=sapien.Pose(p=[
                self.valve_spawn_pos[0],
                self.valve_spawn_pos[1] + WALL_Y_OFFSET,
                self.valve_spawn_pos[2],
            ]),
        )

        self._load_articulations()

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

            for i, obj in enumerate(self.valve_link._objs):
                rb: PhysxRigidBodyComponent = obj.entity.find_component_by_type(PhysxRigidBodyComponent)

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

    def _load_articulations(self):
        # Robel valve
        num_handles_list = self._batched_episode_rng.choice([3, 4], size=(1,))

        # Generate equally spaced angles for each environment
        valve_angles_list = []
        for i in range(self.num_envs):
            num_handles = int(num_handles_list[i])
            angles = np.linspace(0, 2 * np.pi, num_handles, endpoint=False)
            valve_angles_list.append(angles)

        valves: List[Articulation] = []
        capsule_lens = []
        valve_links = []
        for i, valve_angles in enumerate(valve_angles_list):
            scene_idxs = [i]
            if self.difficulty <= 3:
                valve, capsule_len = build_robel_valve(
                    self.scene,
                    valve_angles=valve_angles,
                    scene_idxs=scene_idxs,
                    name=f"valve_station_{i}",
                )
            else:
                scales = self._batched_episode_rng[i].randn(2) * 0.1 + 1
                valve, capsule_len = build_robel_valve(
                    self.scene,
                    valve_angles=valve_angles,
                    scene_idxs=scene_idxs,
                    name=f"valve_station_{i}",
                    radius_scale=scales[0],
                    capsule_radius_scale=scales[1],
                )
            valves.append(valve)
            valve_links.append(valve.links_map["valve"])
            self.remove_from_state_dict_registry(valve)
            capsule_lens.append(capsule_len)
        self.valve = Articulation.merge(valves, "valve_station")
        self.capsule_lens = torch.from_numpy(np.array(capsule_lens)).to(self.device)
        self.valve_link = Link.merge(valve_links, name="valve")

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
            
            if self.intermediate_states:
                self._load_intermediate_states(self.intermediate_states, env_idx)

            # Initialize task related information
            if self.difficulty <=  3:
                self.rotate_direction = torch.ones(b)
            else:
                rand_num = torch.from_numpy(self._batched_episode_rng.randint(0, 2)).float().to(self.device)
                self.rotate_direction = 1 - rand_num * 2

            # Initialize the valve
            xyz = torch.zeros((b, 3))
            random_yz = self._batched_episode_rng.uniform(
                low=-self.valve_spawn_noise,
                high=self.valve_spawn_noise,
                size=(2,)
            )
            random_yz = torch.from_numpy(random_yz).float().to(self.device)

            xyz[:, 0] += self.valve_spawn_pos[0]
            xyz[:, 1] += self.valve_spawn_pos[1] + random_yz[:, 0]
            xyz[:, 2] += self.valve_spawn_pos[2] + random_yz[:, 1]

            random_q = random_quaternions_batched(self._batched_episode_rng, 
                                        device=self.device, lock_x=True, lock_y=True, 
                                        bounds=(-0.261799, 0.261799))

            base_q = torch.tensor([0, -0.7071068, 0, 0.7071068], device=self.device, dtype=torch.float32)
            base_q_batched = base_q.unsqueeze(0).expand(b, -1)
            final_q = quaternion_multiply(random_q, base_q_batched)
            
            pose = Pose.create_from_pq(xyz, final_q)
            self.valve.set_pose(pose)
            self.wall.set_pose(pose)

            random_qpos = self._batched_episode_rng.uniform(
                low=-torch.pi,
                high=torch.pi,
                size=(1,)
            )
            qpos = torch.from_numpy(random_qpos).float().to(self.device)
            self.valve.set_qpos(qpos)
            self.rest_qpos = qpos



    def _get_obs_extra(self, info: Dict):        
        valve_qpos = self.valve.qpos
        valve_qvel = self.valve.qvel
        obs = dict(
            # is_grasped=info["is_grasped"],
            cube_pose=self.cube.pose.raw_pose,
            cube_half_sizes=self.cube_half_sizes_per_env,
            picking_fingers=self.active_finger_indices,
            rotate_dir=self.rotate_direction.to(torch.float32),
            valve_pose=self.valve.pose.raw_pose,
            valve_qpos=valve_qpos,
        )
        if "state" in self.obs_mode:
            
            obs.update(
                tcp_to_valve_pos=self.valve_link.pose.p - self.agent.tcp.pose.p,
            )

        return obs

    def evaluate(self):

        # 2. Finger proximity reward (for picking block)
        finger_object_dist = torch.linalg.norm(
            self.agent.tip_poses[..., :3] - self.cube.pose.p[:, None, :], dim=-1
        )  # shape: (B, num_fingers)

        # Only for pick block do the fingers have to close
        active_finger_dist = torch.gather(finger_object_dist, dim=1, index=self.active_finger_indices)
        finger_close_fraction = (active_finger_dist < 0.07).float().mean(dim=-1)
        is_obj_grasped = (finger_close_fraction >= 0.5)

        # Check valve rotation
        valve_rotation = (self.valve.qpos - self.rest_qpos)[:, 0]
        is_valve_rotated = valve_rotation * self.rotate_direction > self.success_threshold

        # 2. Check cube-table contact
        total_force = torch.zeros(self.num_envs, device=self.device)
        contacts = self.scene.get_pairwise_contact_forces(self.cube, self.table_scene.table)
        if contacts is not None and len(contacts) > 0:
                total_force += contacts.norm(dim=-1)

        has_collision = total_force > 0.01

        is_robot_static = self.agent.is_static(0.2)
        result = {
            "success": is_obj_grasped & is_valve_rotated & ~has_collision,
            "is_obj_grasped": is_obj_grasped,
            "is_valve_rotated": is_valve_rotated,
            "is_robot_static": is_robot_static,
            "valve_rotation": valve_rotation,
        }

        return result


    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):

        # # 1. Finger proximity reward (for picking block)
        finger_object_dist = torch.linalg.norm(
            self.agent.tip_poses[..., :3] - self.cube.pose.p[:, None, :], dim=-1
        )
        active_finger_dist = torch.gather(finger_object_dist, dim=1, index=self.active_finger_indices)
        grasp_reward = torch.exp(-20 * active_finger_dist).mean(dim=-1)
        grasp_fraction = (active_finger_dist < 0.05).float().mean(dim=-1)

        # 2. Reaching reward - TCP to handle
        tcp_to_handle_dist = torch.linalg.norm(self.agent.tcp.pose.p - self.valve.pose.p, dim=-1)
        reaching_reward = 1.0 - torch.tanh(5.0 * tcp_to_handle_dist)

        # 3. Distance between fingertips and valve circle - xy
        finger_object_dist = torch.linalg.norm(
            self.agent.tip_poses[..., :3] - self.valve.pose.p[:, None, :], dim=-1
        )
        inactive_finger_dist = torch.gather(finger_object_dist, dim=1, index=self.inactive_finger_indices)
        finger_reach_reward = torch.exp(-10 * inactive_finger_dist).mean(dim=-1)

        # 4. Valve rotation velocity reward
        qvel = self.valve.qvel
        directed_velocity = qvel[:, 0] * self.rotate_direction
        velocity_reward = torch.tanh(5 * directed_velocity)

        # 5. Valve rotation progress reward
        valve_rotation = info["valve_rotation"]
        motion_reward = torch.clip(valve_rotation / self.success_threshold, -1, 1)

        # 6. Penalize joint velocity to reduce jerky motion
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

            wall_contacts = self.scene.get_pairwise_contact_forces(link, self.wall)
            if wall_contacts is not None and len(wall_contacts) > 0:
                total_force += wall_contacts.norm(dim=-1)

        # penalize between cube and table as well
        contacts = self.scene.get_pairwise_contact_forces(self.cube, self.table_scene.table)
        if contacts is not None and len(contacts) > 0:
                total_force += contacts.norm(dim=-1)

        # penalize between cube and wall
        contacts = self.scene.get_pairwise_contact_forces(self.cube, self.wall)
        if contacts is not None and len(contacts) > 0:
            total_force += contacts.norm(dim=-1)

        cube_valve_contacts = self.scene.get_pairwise_contact_forces(self.cube, self.valve_link)
        if cube_valve_contacts is not None and len(cube_valve_contacts) > 0:
            total_force += cube_valve_contacts.norm(dim=-1)

        collision_penalty = torch.clamp((-0.05 * total_force), min=-5.0, max=0.0)
        
        # 7. Final reward
        reward = (
            self.reward_type["grasp_weight"] * grasp_reward +
            self.reward_type["valve_reach_weight"] * reaching_reward +
            self.reward_type["fingertip_reach_weight"] * finger_reach_reward +
            self.reward_type["velocity_weight"] * velocity_reward +
            self.reward_type["motion_weight"] * motion_reward +
            self.reward_type["static_penalty_weight"] * static_penalty +
            self.reward_type["collision_penalty_weight"] * collision_penalty
        )

        # Bonus for success
        reward[info["success"]] += 5.0
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        # leave out velocity weight, should be relatively transient?
        sum_rewards = 5.0
        for key in ["grasp_weight", "valve_reach_weight", "motion_weight"]:
            sum_rewards += self.reward_type[key]

        return self.compute_dense_reward(obs=obs, action=action, info=info) / sum_rewards