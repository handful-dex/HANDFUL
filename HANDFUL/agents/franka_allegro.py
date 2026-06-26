from copy import deepcopy
from typing import Dict, Tuple, List

import os
import numpy as np
import sapien
import sapien.physx as physx
import torch

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.utils import sapien_utils
from mani_skill.utils.structs.pose import vectorize_pose

@register_agent()
class FrankaAllegro(BaseAgent):
    uid = "franka_allegro_right"
    urdf_path = f"{os.path.dirname(__file__)}/../robot_assets/franka_allegro/franka_panda_allegro_maniskill.urdf"
    keyframes = dict(
        rest=Keyframe(
            qpos=np.array(
                [
                    0.0,
                    -np.pi / 8,
                    0.0,
                    -np.pi * 3 / 4,
                    0.0,
                    np.pi * 5 / 8,
                    np.pi / 4,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
                
            ),
            pose=sapien.Pose(p=[0, 0, 0]),
        )
    )

    def __init__(self, *args, **kwargs):
        self.arm_joint_names = [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ]
        self.arm_stiffness = 1e3
        self.arm_damping = 1e2
        self.arm_force_limit = 100

        self.hand_joint_names = [
            "joint_0.0",
            "joint_1.0",
            "joint_2.0",
            "joint_3.0",
            "joint_4.0",
            "joint_5.0",
            "joint_6.0",
            "joint_7.0",
            "joint_8.0",
            "joint_9.0",
            "joint_10.0",
            "joint_11.0",
            "joint_12.0",
            "joint_13.0",
            "joint_14.0",
            "joint_15.0",
        ]
        self.joint_stiffness = 4e2
        self.joint_damping = 1e1
        self.joint_force_limit = 5e1

        # Order: thumb finger, index finger, middle finger, ring finger
        self.tip_link_names = [
            "link_15.0_tip",
            "link_3.0_tip",
            "link_7.0_tip",
            "link_11.0_tip",
        ]
        self.front_link_names = [
            "link_15.0",
            "link_3.0",
            "link_7.0",
            "link_11.0",
        ]

        self.palm_link_name = "palm"
        self.ee_link_name = "panda_link8"
        super().__init__(*args, **kwargs)
    
    @property
    def _controller_configs(self):
        # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=True,
        )
        arm_pd_joint_target_delta_pos = deepcopy(arm_pd_joint_delta_pos)
        arm_pd_joint_target_delta_pos.use_target = True

        # PD ee position
        arm_pd_ee_delta_pos = PDEEPosControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            rot_lower=-0.1,
            rot_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )
        arm_pd_ee_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=None,
            pos_upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
            use_delta=False,
            normalize_action=False,
        )

        arm_pd_ee_target_delta_pos = deepcopy(arm_pd_ee_delta_pos)
        arm_pd_ee_target_delta_pos.use_target = True
        arm_pd_ee_target_delta_pose = deepcopy(arm_pd_ee_delta_pose)
        arm_pd_ee_target_delta_pose.use_target = True

        # PD joint velocity
        arm_pd_joint_vel = PDJointVelControllerConfig(
            self.arm_joint_names,
            -1.0,
            1.0,
            self.arm_damping,  # this might need to be tuned separately
            self.arm_force_limit,
        )

        # PD joint position and velocity
        arm_pd_joint_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            None,
            None,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            use_delta=True,
        )
        # -------------------------------------------------------------------------- #
        # Hand
        # -------------------------------------------------------------------------- #
                # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
        hand_joint_pos = PDJointPosControllerConfig(
            self.hand_joint_names,
            None,
            None,
            self.joint_stiffness,
            self.joint_damping,
            self.joint_force_limit,
            normalize_action=False,
        )
        hand_joint_delta_pos = PDJointPosControllerConfig(
            self.hand_joint_names,
            -0.1,
            0.1,
            self.joint_stiffness,
            self.joint_damping,
            self.joint_force_limit,
            use_delta=True,
        )
        hand_joint_target_delta_pos = deepcopy(hand_joint_delta_pos)
        hand_joint_target_delta_pos.use_target = True

        controller_configs = dict(
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos, gripper=hand_joint_delta_pos
            ),
            pd_joint_pos=dict(arm=arm_pd_joint_pos, gripper=hand_joint_pos),
            pd_ee_delta_pos=dict(arm=arm_pd_ee_delta_pos, gripper=hand_joint_delta_pos),
            pd_ee_delta_pose=dict(
                arm=arm_pd_ee_delta_pose, gripper=hand_joint_delta_pos
            ),
            pd_ee_pose=dict(arm=arm_pd_ee_pose, gripper=hand_joint_pos),
            # TODO(jigu): how to add boundaries for the following controllers
            pd_joint_target_delta_pos=dict(
                arm=arm_pd_joint_target_delta_pos, gripper=hand_joint_target_delta_pos
            ),
            pd_ee_target_delta_pos=dict(
                arm=arm_pd_ee_target_delta_pos, gripper=hand_joint_target_delta_pos
            ),
            pd_ee_target_delta_pose=dict(
                arm=arm_pd_ee_target_delta_pose, gripper=hand_joint_target_delta_pos
            ),
            # Caution to use the following controllers
            pd_joint_vel=dict(arm=arm_pd_joint_vel, gripper=hand_joint_pos),
            pd_joint_pos_vel=dict(
                arm=arm_pd_joint_pos_vel, gripper=hand_joint_pos
            ),
            pd_joint_delta_pos_vel=dict(
                arm=arm_pd_joint_delta_pos_vel, gripper=hand_joint_pos
            ),
        )

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)


    def _after_init(self):
        self.hand_front_links = sapien_utils.get_objs_by_names(
            self.robot.get_links(), self.front_link_names
        )

        self.tip_links: List[sapien.Entity] = sapien_utils.get_objs_by_names(
            self.robot.get_links(), self.tip_link_names
        )

        self.palm_link: sapien.Entity = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.palm_link_name
        )

        self.tcp = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.ee_link_name
        )

        self.queries: Dict[str, Tuple[physx.PhysxGpuContactQuery, Tuple[int]]] = dict()

    def get_proprioception(self):
        """
        Get the proprioceptive state of the agent.
        """
        obs = super().get_proprioception()
        obs.update(
            {
                "palm_pose": self.palm_pose,
                "tip_poses": self.tip_poses.reshape(-1, len(self.tip_links) * 7),
            }
        )

        return obs

    @property
    def tip_poses(self):
        """
        Get the tip pose for each of the finger, four fingers in total
        """
        tip_poses = [
            vectorize_pose(link.pose, device=self.device) for link in self.tip_links
        ]
        return torch.stack(tip_poses, dim=-2)

    @property
    def palm_pose(self):
        """
        Get the palm pose for allegro hand
        """
        return vectorize_pose(self.palm_link.pose, device=self.device)

    @property
    def tcp_pos(self):
        return self.tcp.pose.p

    @property
    def tcp_pose(self):
        return self.tcp.pose
    
    def is_static(self, threshold: float = 0.2):
        qvel = self.robot.get_qvel()[..., :]
        return torch.max(torch.abs(qvel), 1)[0] <= threshold