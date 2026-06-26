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
class XArm7Allegro(BaseAgent):
    uid = "xarm7_allegro_right"
    urdf_path = f"{os.path.dirname(__file__)}/../robot_assets/xarm7/xarm7_allegro_right.urdf"
    keyframes = dict(
        rest=Keyframe(
            qpos=np.array(
                [
                    0.0,
                    -0.4,
                    0.0,
                    0.5,
                    0.0,
                    0.9,
                    -3.0,
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
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]
        self.arm_stiffness = 1e3
        self.arm_damping = 1e2
        self.arm_force_limit = 500

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
        self.hand_stiffness = 4e2
        self.hand_damping = 1e1
        self.hand_force_limit = 5e1

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
        self.ee_link_name = "base_link"
        super().__init__(*args, **kwargs)
    
    @property
    def _controller_configs(self):
        # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            None,
            None,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            use_delta=True,
        )
        arm_pd_joint_target_delta_pos = deepcopy(arm_pd_joint_delta_pos)
        arm_pd_joint_target_delta_pos.use_target = True

        # PD ee position
        # TODO: Doesn't work yet
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            self.arm_joint_names,
            -1.0,
            1.0,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
            rot_lower= -2*np.pi,
            rot_upper= 2*np.pi,
            use_delta=True,
        )

        arm_pd_ee_target_delta_pose = deepcopy(arm_pd_ee_delta_pose)
        arm_pd_ee_target_delta_pose.use_target = True

        # -------------------------------------------------------------------------- #
        # Hand
        # -------------------------------------------------------------------------- #
        hand_target_delta_pos = PDJointPosControllerConfig(
            self.hand_joint_names,
            -0.1,
            0.1,
            self.hand_stiffness,
            self.hand_damping,
            self.hand_force_limit,
            use_delta=True,
        )
        hand_target_delta_pos.use_target = True

        controller_configs = dict(
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos, gripper=hand_target_delta_pos
            ),
            pd_joint_pos=dict(arm=arm_pd_joint_pos, gripper=hand_target_delta_pos),
            pd_ee_delta_pose=dict(
                arm=arm_pd_ee_delta_pose, gripper=hand_target_delta_pos
            ),
            pd_ee_target_delta_pose=dict(
                arm=arm_pd_ee_target_delta_pose, gripper=hand_target_delta_pos
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