from typing import Any, Dict, Union

import numpy as np
import sapien
import torch
import trimesh

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.robots import XArm7Ability, XArm6Robotiq
from agents.xarm7_allegro import XArm7Allegro
from agents.xarm7_leap import XArm7Leap
from agents.franka_allegro import FrankaAllegro
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.envs.sapien_env import BaseEnv

from envs.config.xarm7_allegro_config import (PICK_CUBE_CONFIGS, PICK_REWARD_CONFIGS, CABINET_CONFIGS, CABINET_REWARD_CONFIGS, 
                                             MASS_VARIATIONS, FRICTION_VARIATIONS, RESTITUTION_VARIATIONS)
from envs.utils.batched_pose import random_quaternions_batched
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils, common
from mani_skill.utils.building import actors, articulations
from mani_skill.utils.geometry.geometry import transform_points
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Articulation, Link, Pose, Actor
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.utils.geometry.rotation_conversions import quaternion_multiply

import sapien.physx as physx
from sapien.physx import PhysxRigidBodyComponent
from sapien.render import RenderBodyComponent

MAX_EPISODE_STEPS = 175
CABINET_COLLISION_BIT = 29
N_MATERIAL_POOL = 1024

OPEN_DRAWER_DOC_STRING = """ Open a drawer while holding an object
TBD
"""

@register_env("xArm7-v1-cabinet-drawer-unified", max_episode_steps=MAX_EPISODE_STEPS)
class XArm7CabinetDrawerEnvUnified(BaseEnv):
    """
    Unified environment combining two sequential picking tasks:
    Phase 1 (0-74 steps): Pick and place first cube to goal
    Phase 2 (75-174 steps): Open drawer
    
    Rewards are automatically switched based on current phase.
    """

    SUPPORTED_ROBOTS = [
        "xarm7_ability", "xarm6_robotiq", "xarm7_allegro_right", "xarm7_leap_right", "franka_allegro_right"
    ]
    agent: Union[XArm7Ability, XArm6Robotiq, XArm7Allegro, XArm7Leap, FrankaAllegro]
    
    handle_types = ["prismatic"]
    TRAIN_JSON = (
        PACKAGE_ASSET_DIR / "partnet_mobility/meta/info_cabinet_drawer_train.json"
    )
    min_open_frac = 0.55

    cabinet_model_ids = [1067]
                        # [1004, 1024, 1076, 1052, 1063, 1067]
                        # [1000, 1004, 1005, 1013, 1016,
                        #  1021, 1024, 1027, 1032, 1033,
                        #  1035, 1038, 1040, 1044, 1045,
                        #  1052, 1054, 1056, 1061, 1063,
                        #  1066, 1067, 1076, 1079, 1082]

    def __init__(self, *args, **kwargs):

        known_custom_args = ["robot_uids", "reward_type", "robot_init_qpos_noise",
                             "finger_selection", "palm_use", "num_active_fingers", "difficulty"]
        for arg in known_custom_args:
            setattr(self, arg, kwargs.pop(arg, None))

        self.phase1_steps = 75

        self.robot_uids = "xarm7_leap_right" if self.robot_uids is None else self.robot_uids
        self.robot_init_qpos_noise = 0.02 if self.robot_init_qpos_noise is None else self.robot_init_qpos_noise
        self.pick_reward = PICK_REWARD_CONFIGS["sac"] if self.reward_type is None else PICK_REWARD_CONFIGS[self.reward_type]
        self.cabinet_reward = CABINET_REWARD_CONFIGS["sac"] if self.reward_type is None else CABINET_REWARD_CONFIGS[self.reward_type]
        self.difficulty = 3 if self.difficulty is None else self.difficulty


        self.finger_selection = "fixed" if self.finger_selection is None else self.finger_selection
        self.palm_use = False if self.palm_use is None else self.palm_use
        self.num_active_fingers = 2 if self.num_active_fingers is None else self.num_active_fingers

        if self.robot_uids in PICK_CUBE_CONFIGS:
            pick_cfg = PICK_CUBE_CONFIGS[self.robot_uids]
        else:
            pick_cfg = PICK_CUBE_CONFIGS["xarm7_leap_right"]

        if self.robot_uids in CABINET_CONFIGS:
            cabinet_cfg = CABINET_CONFIGS[self.robot_uids]
        else:
            cabinet_cfg = CABINET_CONFIGS["xarm7_leap_right"]

        self.cube_variation_dimensions = pick_cfg["cube_variation_dimensions"]
        self.goal_thresh = pick_cfg["goal_thresh"]
        self.cube_spawn_half_size = pick_cfg["cube_spawn_half_size"]
        self.cube_spawn_center = pick_cfg["cube_spawn_center"]

        self.max_cube_goal_height = pick_cfg["max_goal_height"]
        self.sensor_cam_eye_pos = pick_cfg["sensor_cam_eye_pos"]
        self.sensor_cam_target_pos = pick_cfg["sensor_cam_target_pos"]

        self.cabinet_radius = cabinet_cfg["cabinet_radius"]
        self.cabinet_angle_range = cabinet_cfg["cabinet_angle_range"]
        self.cabinet_spawn_noise = cabinet_cfg["cabinet_spawn_noise"]
        self.cabinet_spawn_rot_noise = cabinet_cfg["cabinet_spawn_rot_noise"]

        self.human_cam_eye_pos = cabinet_cfg["human_cam_eye_pos"]
        self.human_cam_target_pos = cabinet_cfg["human_cam_target_pos"]

        if self.difficulty <= 1:
            self.cabinet_angle_range = [0 for angle in self.cabinet_angle_range]
            self.cabinet_spawn_noise = 0
            self.cabinet_spawn_rot_noise = 0
        elif self.difficulty == 2:
            self.cabinet_angle_range = [angle/2 for angle in self.cabinet_angle_range]
            self.cabinet_spawn_noise /= 2
            self.cabinet_spawn_rot_noise /= 2
        else:
            self.cabinet_angle_range = self.cabinet_angle_range
            self.cabinet_spawn_noise = self.cabinet_spawn_noise
            self.cabinet_spawn_rot_noise = self.cabinet_spawn_rot_noise

        # Load cabinet metadata
        train_data = load_json(self.TRAIN_JSON)
        all_ids = np.array(list(train_data.keys()))

        # Filter to specific IDs if provided
        if self.cabinet_model_ids is not None:
            self.all_model_ids = np.array([str(id) for id in self.cabinet_model_ids if str(id) in all_ids])
            print(f"Using {len(self.all_model_ids)} specified cabinet models")
        else:
            self.all_model_ids = all_ids

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

        # Load cabinets
        sapien.set_log_level("off")
        self._load_cabinets(self.handle_types)
        sapien.set_log_level("warn")
        
        self.table_scene.table.set_collision_group_bit(
            group=2, bit_idx=CABINET_COLLISION_BIT, bit=1
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

            
            for i, obj in enumerate(self.handle_link._objs):
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
                    

    def _load_cabinets(self, joint_types: list[str]):
        # from open_cabinet_drawer.py in the mobile manipulation tasks

        model_ids = self._batched_episode_rng.choice(self.all_model_ids)
        link_ids = self._batched_episode_rng.randint(0, 2**31)

        self._cabinets: list[Articulation] = []
        handle_links: list[list[Link]] = []
        handle_links_meshes: list[list[trimesh.Trimesh]] = []
        
        for i, model_id in enumerate(model_ids):
            cabinet_builder = articulations.get_articulation_builder(
                self.scene, f"partnet-mobility:{model_id}"
            )
            cabinet_builder.set_scene_idxs(scene_idxs=[i])
            cabinet_builder.initial_pose = sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0])


            # ── ADD A TOP PANEL TO THE BASE LINK ──────────────────────────────
            # Find the base link builder and get its bounding box first
            base_link_builder = None
            for lb in cabinet_builder.link_builders:
                if lb.name == "base":
                    base_link_builder = lb
                    break

            if base_link_builder is not None:
                # Typical cabinet base is ~0.5m wide, 0.4m deep, 0.8m tall
                # Tune these values or derive from collision mesh bounds if needed
                top_half_size = [0.24, 0.17, 0.01]   # x, y, z half-extents in meters
                top_z_offset  = 0.31                 # height above base origin (~cabinet height)

                # Visual-only box (no collision, so it doesn't affect physics)
                top_visual = base_link_builder.add_box_visual(
                    pose=sapien.Pose(p=[0, 0.01, top_z_offset]),
                    half_size=top_half_size,
                    material=sapien.render.RenderMaterial(
                        base_color=[0.5450980392156862, 0.3764705882352941, 0.2901960784313726, 1.0],   # warm wood color
                        roughness=0.7,
                        metallic=0.0,
                    ),
                    name="cabinet_top_panel",
                )
                # Optional: also add a matching thin collision box if you want the
                # robot to be able to rest objects on top:
                # base_link_builder.add_box_collision(
                #     pose=sapien.Pose(p=[0, 0, top_z_offset]),
                #     half_size=top_half_size,
                # )
            # ──────────────────────────────────────────────────────────────────

            for lb in cabinet_builder.link_builders:
                # The child link of joint_0 is typically named "link_0"
                if lb.name == "link_0":
                    # Remove only visuals whose render shape name contains "handle"
                    lb.visual_records = [
                        vr for vr in lb.visual_records
                        if "handle" not in vr.name.lower()
                    ]
                    break


            cabinet = cabinet_builder.build(name=f"{model_id}-{i}")
            self.remove_from_state_dict_registry(cabinet)
            
            for link in cabinet.links:
                link.set_collision_group_bit(
                    group=2, bit_idx=CABINET_COLLISION_BIT, bit=1
                )
            self._cabinets.append(cabinet)
            handle_links.append([])
            handle_links_meshes.append([])

            for link, joint in zip(cabinet.links, cabinet.joints):
                if joint.type[0] in joint_types:
                    handle_links[-1].append(link)
                    handle_links_meshes[-1].append(
                        link.generate_mesh(
                            filter=lambda _, render_shape: "handle"
                            in render_shape.name,
                            mesh_name="handle",
                        )[0]
                    )

        self.cabinet = Articulation.merge(self._cabinets, name="cabinet")
        self.add_to_state_dict_registry(self.cabinet)
        self.handle_link = Link.merge(
            [links[link_ids[i] % len(links)] for i, links in enumerate(handle_links)],
            name="handle_link",
        )
        self.handle_link_pos = common.to_tensor(
            np.array(
                [
                    meshes[link_ids[i] % len(meshes)].bounding_box.center_mass
                    for i, meshes in enumerate(handle_links_meshes)
                ]
            ),
            device=self.device,
        )

        self.handle_link_goal = actors.build_sphere(
            self.scene,
            radius=0.02,
            color=[0, 1, 0, 1],
            name="handle_link_goal",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),
        )

    def _after_reconfigure(self, options):
        self.cabinet_zs = []
        for cabinet in self._cabinets:
            collision_mesh = cabinet.get_first_collision_mesh()
            self.cabinet_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.cabinet_zs = common.to_tensor(self.cabinet_zs, device=self.device)

        target_qlimits = self.handle_link.joint.limits
        qmin, qmax = target_qlimits[..., 0], target_qlimits[..., 1]
        self.target_qpos = qmin + (qmax - qmin) * self.min_open_frac

    def handle_link_positions(self, env_idx: torch.Tensor = None):
        if env_idx is None:
            return transform_points(
                self.handle_link.pose.to_transformation_matrix().clone(),
                common.to_tensor(self.handle_link_pos, device=self.device),
            )
        return transform_points(
            self.handle_link.pose[env_idx].to_transformation_matrix().clone(),
            common.to_tensor(self.handle_link_pos[env_idx], device=self.device),
        )

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

            # Position cabinets
            xyz = torch.zeros((b, 3))

            # Sample a random base angle per env
            base_angle = torch.tensor(
                self._batched_episode_rng.uniform(
                    low=self.cabinet_angle_range[0],
                    high=self.cabinet_angle_range[1],
                    size=(1,)
                ),
                dtype=torch.float32, device=self.device
            ).squeeze(-1)  # shape (b,)

            # Place cabinet at fixed radius from robot, at that angle
            robot_base = torch.tensor([-0.3, 0.0], dtype=torch.float32, device=self.device)

            xyz[:, 0] = robot_base[0] + self.cabinet_radius * torch.cos(base_angle)
            xyz[:, 1] = robot_base[1] + self.cabinet_radius * torch.sin(base_angle)

            # Small positional noise
            random_xy = self._batched_episode_rng.uniform(
                low=-self.cabinet_spawn_noise,
                high=self.cabinet_spawn_noise,
                size=(2,)
            )
            xyz[:, :2] += torch.from_numpy(random_xy).float().to(self.device)
            xyz[:, 2] += self.cabinet_zs[env_idx] - 0.29

            # Face cabinet inward toward robot, then apply existing rot noise
            half_facing = base_angle / 2.0
            base_q = torch.stack([
                torch.cos(half_facing),
                torch.zeros(b, device=self.device),
                torch.zeros(b, device=self.device),
                torch.sin(half_facing),
            ], dim=-1)  # (b, 4)

            random_q = random_quaternions_batched(self._batched_episode_rng,
                                                device=self.device, lock_x=True, lock_y=True,
                                                bounds=(-self.cabinet_spawn_rot_noise, self.cabinet_spawn_rot_noise))

            final_q = quaternion_multiply(random_q, base_q)

            self.cabinet.set_pose(Pose.create_from_pq(p=xyz, q=final_q))
            
            # Close all cabinets
            qlimits = self.cabinet.get_qlimits()
            self.cabinet.set_qpos(qlimits[env_idx, :, 0])
            self.cabinet.set_qvel(self.cabinet.qpos[env_idx] * 0)


            # GPU sim workaround
            if self.gpu_sim_enabled:
                self.scene._gpu_apply_all()
                self.scene.px.gpu_update_articulation_kinematics()
                self.scene.px.step()
                self.scene._gpu_fetch_all()

            self.handle_link_goal.set_pose(
                Pose.create_from_pq(p=self.handle_link_positions(env_idx))
            )

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

    def _after_control_step(self):
        if self.gpu_sim_enabled:
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene._gpu_fetch_all()
        self.handle_link_goal.set_pose(
            Pose.create_from_pq(p=self.handle_link_positions())
        )
        if self.gpu_sim_enabled:
            self.scene._gpu_apply_all()

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
            picking_fingers=self.active_finger_indices,
            target_handle_pos=info["handle_link_pos"],
            phase=phase,
        )
        if "state" in self.obs_mode:
            
            obs.update(
                tcp_to_cube_pos=self.cube.pose.p - self.agent.tcp.pose.p,
                cube_to_goal_pos=self.goal_site.pose.p - self.cube.pose.p,

                tcp_to_handle_pos=info["handle_link_pos"] - self.agent.tcp.pose.p,
                target_link_qpos=self.handle_link.joint.qpos,
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
                cube_pose=self.cube.pose.raw_pose,
                cube_half_sizes=self.cube_half_sizes_per_env,
                picking_fingers=self.active_finger_indices,
                target_handle_pos=info["handle_link_pos"],
            )
            if "state" in self.obs_mode:
                extra.update(
                    tcp_to_obj_pos=self.cube.pose.p - self.agent.tcp.pose.p,
                    tcp_to_handle_pos=info["handle_link_pos"] - self.agent.tcp.pose.p,
                    target_link_qpos=self.handle_link.joint.qpos,

                )
        
        obs = {
            "agent": self.agent.get_proprioception(),
            "extra": extra,
            }
                
        return obs

    def evaluate(self):

        # active finger dist to cube
        finger_object_dist = torch.linalg.norm(
            self.agent.tip_poses[..., :3] - self.cube.pose.p[:, None, :], dim=-1
        )

        active_finger_dist = torch.gather(finger_object_dist, dim=1, index=self.active_finger_indices)
        finger_close_fraction = (active_finger_dist < 0.07).float().mean(dim=-1)
        is_obj_grasped = (finger_close_fraction > 0.5)

        open_enough = self.handle_link.joint.qpos >= self.target_qpos
        handle_link_pos = self.handle_link_positions()
        
        link_is_static = (
            torch.linalg.norm(self.handle_link.angular_velocity, axis=1) <= 1
        ) & (torch.linalg.norm(self.handle_link.linear_velocity, axis=1) <= 0.1)
        
        # 2. Check cube-table contact
        total_force = torch.zeros(self.num_envs, device=self.device)
        contacts = self.scene.get_pairwise_contact_forces(self.cube, self.table_scene.table)
        if contacts is not None and len(contacts) > 0:
                total_force += contacts.norm(dim=-1)

        has_collision = total_force > 0.01

        result = {
            "success": open_enough & link_is_static & is_obj_grasped & ~has_collision,
            "is_obj_grasped": is_obj_grasped,
            "handle_link_pos": handle_link_pos,
            "open_enough": open_enough,
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

        # 1. Finger proximity reward (for picking block) - only for active fingers
        finger_object_dist = torch.linalg.norm(
            self.agent.tip_poses[..., :3] - self.cube.pose.p[:, None, :], dim=-1
        )
        active_finger_dist = torch.gather(finger_object_dist, dim=1, index=self.active_finger_indices)
        grasp_reward = torch.exp(-20 * active_finger_dist).mean(dim=-1)
        grasp_fraction = (active_finger_dist < 0.07).float().mean(dim=-1)
        
        # Penalize inactive finger contacts
        inactive_finger_penalty = self.compute_inactive_finger_penalty()

        # 2. Reaching reward - TCP to handle
        tcp_to_handle_dist = torch.linalg.norm(
            self.agent.tcp.pose.p - info["handle_link_pos"], axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_handle_dist)
        
        # 3. Inactive finger tips to handle reward (encourage proper grasping posture)
        finger_to_handle_dist = torch.linalg.norm(
            self.agent.tip_poses[..., :3] - info["handle_link_pos"][:, None, :], dim=-1
        )
        inactive_finger_dist = torch.gather(finger_to_handle_dist, dim=1, index=self.inactive_finger_indices)
        finger_reach_reward = (1.0 - torch.tanh(5.0 * inactive_finger_dist)).mean(dim=-1)

        # 4. Opening reward
        amount_to_open_left = torch.div(
            self.target_qpos - self.handle_link.joint.qpos, self.target_qpos
        )
        open_reward = (1 - amount_to_open_left)
        
        # Once joint starts opening, maximize open reward
        reaching_reward[amount_to_open_left < 0.999] = 1.0
        open_reward[info["open_enough"]] = 1.0
        finger_reach_reward[info["open_enough"]] = 1.0

        # 4. Static penalty (reduce jerky motion)
        qvel = self.agent.robot.get_qvel()
        if self.robot_uids == "xarm7_leap_right":
            qvel = qvel[..., :-16]
        static_penalty = -torch.linalg.norm(qvel, axis=1)

        # 5. Collision penalty
        total_force = torch.zeros(self.num_envs, device=self.device)
        for link in self.agent.robot.get_links():
            table_contacts = self.scene.get_pairwise_contact_forces(link, self.table_scene.table)
            if table_contacts is not None and len(table_contacts) > 0:
                total_force += table_contacts.norm(dim=-1)
        
        contacts = self.scene.get_pairwise_contact_forces(self.cube, self.table_scene.table)
        if contacts is not None and len(contacts) > 0:
                total_force += contacts.norm(dim=-1)

        collision_penalty = - 0.05 * total_force

        # Final reward
        reward = (
            self.cabinet_reward["grasp_weight"] * grasp_reward +
            self.cabinet_reward["reach_weight"] * reaching_reward +
            self.cabinet_reward["finger_dist_weight"] * finger_reach_reward +
            self.cabinet_reward["open_weight"] * open_reward +
            self.cabinet_reward["inactive_finger_penalty_weight"] * inactive_finger_penalty +
            self.cabinet_reward["static_penalty_weight"] * static_penalty +
            self.cabinet_reward["collision_penalty_weight"] * collision_penalty
        )
        
        reward[info["success"]] += 5.0
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        
        phase = self._get_current_phase()
        
        # Calculate normalization constants for each phase
        sum_phase1 = sum(self.pick_reward[k] for k in 
                        ["reach_weight", "finger_dist_weight", "grasp_weight", "place_weight"])
        sum_phase2 = 5.0 + sum(self.cabinet_reward[k] for k in 
                            ["grasp_weight", "reach_weight", "finger_dist_weight", "open_weight"])
        
        raw_reward = self.compute_dense_reward(obs=obs, action=action, info=info)
        
        # Hard split normalization based on phase
        normalization = torch.where(
            phase == 1, 
            torch.tensor(sum_phase1, device=self.device),
            torch.tensor(sum_phase2, device=self.device)
        )
        
        return raw_reward / normalization

@register_env("xArm7-v1-cabinet-door-unified", max_episode_steps=MAX_EPISODE_STEPS)
class XArm7CabinetDoorEnvUnified(XArm7CabinetDrawerEnvUnified):
    """Cabinet door opening variant"""
    TRAIN_JSON = (
        PACKAGE_ASSET_DIR / "partnet_mobility/meta/info_cabinet_door_train.json"
    )
    handle_types = ["revolute", "revolute_unwrapped"]

    cabinet_model_ids = [1052, 1063, 1067]
