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
from mani_skill.utils.geometry.rotation_conversions import quaternion_apply


@register_agent()
class XArm7Leap(BaseAgent):
    uid = "xarm7_leap_right"
    urdf_path = f"{os.path.dirname(__file__)}/../robot_assets/xarm7/xarm7_leap_right_alt.urdf"
    keyframes = dict(
        rest=Keyframe(
            qpos=np.array(
                [
                    0.0,
                    0.1,
                    0.0,
                    0.8,
                    0.0,
                    0.7,
                    -3.14,
                    0.0, # index mcp
                    0.0, # middle mcp
                    0.0, # ring mcp
                    0.8, # thumb mcp
                    0.0, # index pip
                    0.0,
                    0.0, 
                    0.0, # thumb pip 
                    0.0, # index dip
                    0.0,
                    0.0,
                    0.0, # thumb dip 
                    0.0, # index tip
                    0.0,
                    0.0,
                    0.0, # thumb tip
                ]
            ),
            pose=sapien.Pose(p=[0, 0, 0]),
        ),
        rest_palm=Keyframe(
            qpos=np.array(
                [
                    0.0,
                    0.6,
                    0.0,
                    0.4,
                    0.0,
                    -1.5,
                    -3.14,
                    0.0, # index mcp
                    0.0, # middle mcp
                    0.0, # ring mcp
                    0.8, # thumb mcp
                    0.0, # index pip
                    0.0,
                    0.0, 
                    0.0, # thumb pip 
                    0.5, # index dip
                    0.5,
                    0.5,
                    0.0, # thumb dip 
                    0.0, # index tip
                    0.0,
                    0.0,
                    0.0, # thumb tip
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
        self.arm_stiffness = 30
        self.arm_damping = 15
        self.arm_force_limit = 800
        self.arm_friction = 10

        self.hand_joint_names = [
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
        ]
        self.hand_stiffness = 0.5
        self.hand_damping = 0.05
        self.hand_friction = 0.0
        self.hand_force_limit = 5e2

        # Order: index finger, middle finger, ring finger, thumb finger
        self.tip_link_names = [
            "fingertip",
            "fingertip_2",
            "fingertip_3",
            "thumb_fingertip",
        ]
        self.front_link_names = [
            "dip",
            "dip_2",
            "dip_3",
            "thumb_dip",
        ]
        self.pip_link_names = [
            "pip",
            "pip_2",
            "pip_3",
            "thumb_pip",
        ]
        self.mcp_link_names = [
            "mcp_joint",
            "mcp_joint_2",
            "mcp_joint_3"
        ]
        

        self.palm_link_name = "palm_lower"
        self.ee_link_name = "link7"

        super().__init__(*args, **kwargs)

        # Compensate for offset in fingertip frame
        self.offsets = torch.tensor([
            [-0.005, -0.0345, 0.0145],  # Finger 1
            [-0.005, -0.0345, 0.0145],  # Finger 2
            [-0.005, -0.0345, 0.0145],  # Finger 3
            [-0.005, -0.0470, -0.0145], # Thumb
        ], device=self.device)

        # TCP offset
        self.tcp_offset = sapien.Pose(p=[0, 0, 0.125])

        # Palm center offset from palm_link origin
        self.palm_center_offset = torch.tensor([-0.05, -0.0385, -0.02], device=self.device)

        # Palm contact points as offsets from palm center
        self.palm_contact_offsets = torch.tensor([
            [0.0, 0.046, 0.0],      # right (towards pinky side)
            [0.0, 0.023, 0.0],
            [0.0, 0.0, 0.0],       # center
            [0.0, -0.023, 0.0],
            [0.0, -0.046, 0.0],     # left (towards thumb side)
        ], device=self.device)


    
    def reset(self, init_qpos = keyframes["rest"].qpos):
        """
        Reset the robot to a clean state with zero velocity and forces.

        Args:
            init_qpos (torch.Tensor): The initial qpos to set the robot to. If None, the robot's qpos is not changed.
        """
        if init_qpos is not None:
            self.robot.set_qpos(init_qpos)
        self.robot.set_qvel(torch.zeros(self.robot.max_dof, device=self.device))
        self.robot.set_qf(torch.zeros(self.robot.max_dof, device=self.device))

        self.controller.reset()
                
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
            -0.2,
            0.2,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            use_delta=True,
        )
        arm_pd_joint_target_delta_pos = deepcopy(arm_pd_joint_delta_pos)
        arm_pd_joint_target_delta_pos.use_target = True

        # PD ee position
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-1,
            pos_upper=1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
            rot_lower= -np.pi,
            rot_upper= np.pi,
            normalize_action=False,
        )

        arm_pd_ee_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=None,
            pos_upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            friction=self.arm_friction,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
            use_delta=False,
            normalize_action=False,
        )

        arm_pd_ee_target_delta_pose = deepcopy(arm_pd_ee_delta_pose)
        arm_pd_ee_target_delta_pose.use_target = True

        # -------------------------------------------------------------------------- #
        # Hand
        # -------------------------------------------------------------------------- #
        hand_target_pos = PDJointPosControllerConfig(
            self.hand_joint_names,
            lower=None, # -0.5
            upper=None, # 0.5
            stiffness=self.hand_stiffness,
            damping=self.hand_damping,
            force_limit=self.hand_force_limit,
            friction=self.hand_friction,
            normalize_action=False,
            use_delta=False,
        )
        hand_target_pos.use_target = False

        hand_delta_pos = PDJointPosControllerConfig(
            self.hand_joint_names,
            -0.2,
            0.2,
            stiffness=self.hand_stiffness,
            damping=self.hand_damping,
            force_limit=self.hand_force_limit,
            friction=self.hand_friction,
            use_delta=True,
        )

        controller_configs = dict(
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos, gripper=hand_delta_pos
            ),
            pd_joint_pos=dict(arm=arm_pd_joint_pos, gripper=hand_target_pos),
            pd_ee_delta_pose=dict(
                arm=arm_pd_ee_delta_pose, gripper=hand_delta_pos
            ),
            pd_ee_target_delta_pose=dict(
                arm=arm_pd_ee_target_delta_pose, gripper=hand_delta_pos
            ),
            pd_ee_pose=dict(
                arm=arm_pd_ee_pose, gripper=hand_target_pos
            ),
        )

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)

    def _after_init(self):
        self.hand_front_links = sapien_utils.get_objs_by_names(
            self.robot.get_links(), self.front_link_names
        )

        self.mcp_links = sapien_utils.get_objs_by_names(
            self.robot.get_links(), self.mcp_link_names
        )

        self.pip_links = sapien_utils.get_objs_by_names(
            self.robot.get_links(), self.pip_link_names
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
                "tcp_pose": self.tcp_pose.raw_pose,
            }
        )

        return obs

    @property
    def tip_poses(self):
        """
        Get the tip pose for each of the finger, four fingers in total
        """
        adjusted_poses = []
        for i, link in enumerate(self.tip_links):
            # Apply offset to each tip link
            adjusted_pose = link.pose * self.offsets[i]
            adjusted_poses.append(vectorize_pose(adjusted_pose, device=self.device))
        
        # Stack into (batch, num_fingers, 7) tensor
        tip_poses = torch.stack(adjusted_poses, dim=-2)
        
        return tip_poses

    @property
    def palm_pose(self):
        """
        Get the palm pose for allegro hand
        """
        adjusted_pose = self.palm_link.pose * self.palm_center_offset
        return vectorize_pose(adjusted_pose, device=self.device)

    @property
    def palm_contact_points(self):
        """
        Get the palm contact points in world frame
        """
        # Get palm center pose
        palm_center_pose = self.palm_link.pose * self.palm_center_offset
        
        # Transform each contact offset
        contact_points = []
        for offset in self.palm_contact_offsets:
            point_pose = palm_center_pose * offset
            contact_points.append(vectorize_pose(point_pose, device=self.device))
        
        # Stack into (batch, num_points, 3) tensor
        return torch.stack(contact_points, dim=-2)

    @property
    def tcp_pose(self):
        """
        Get the adjusted tcp pose
        """

        adjusted_pose = self.tcp.pose * self.tcp_offset
        
        return adjusted_pose

    @property
    def tcp_pos(self):
        """
        Get the adjusted tcp position
        """
        
        return self.tcp_pose.p
    
    def is_static(self, threshold: float = 0.2):
        qvel = self.robot.get_qvel()[..., :]
        return torch.max(torch.abs(qvel), 1)[0] <= threshold

