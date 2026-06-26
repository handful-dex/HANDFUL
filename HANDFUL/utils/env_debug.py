import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import gymnasium as gym
import mani_skill.envs
import numpy as np

from envs.tasks.xarm7_leap_pick_env import XArm7TableTop
from envs.tasks.xarm7_leap_push_env import XArm7TableTopPush
from envs.tasks.xarm7_press_button import XArm7TableTopPress
from envs.tasks.xarm7_pick_randomized import XArm7TableTopPickRandomized
from envs.tasks.xarm7_pick_all import XArm7TableTopPickAll
from envs.tasks.xarm7_two_pick import XArm7TableTopTwoPick
from envs.tasks.xarm7_twist import XArm7TableTopKnobTwist
from envs.tasks.xarm7_open_drawer import XArm7CabinetDrawerEnv, XArm7CabinetDoorEnv

from envs.tasks.unified_environments.xarm7_two_pick_unified import XArm7TableTopTwoPickUnified
from envs.tasks.unified_environments.xarm7_push_unified import XArm7TableTopPushUnified
from envs.tasks.unified_environments.xarm7_open_drawer_unified import XArm7CabinetDrawerEnvUnified
from envs.tasks.unified_environments.xarm7_press_button_unified import XArm7TableTopPressUnified
from envs.tasks.unified_environments.xarm7_twist_unified import XArm7TableTopKnobTwistUnified

from envs.tasks.whole_hand_environments.xarm7_push_whole_hand import XArm7TableTopPushWholeHand
from envs.tasks.whole_hand_environments.xarm7_open_drawer_whole_hand import XArm7CabinetDrawerWholeHand
from envs.tasks.whole_hand_environments.xarm7_press_button_whole_hand import XArm7TableTopPressWholeHand
from envs.tasks.whole_hand_environments.xarm7_twist_whole_hand import XArm7TableTopKnobTwistWholeHand
from envs.tasks.whole_hand_environments.xarm7_two_pick_whole_hand import XArm7TableTopTwoPickWholeHand


from mani_skill.utils import gym_utils
from mani_skill.utils.structs.pose import Pose


num_envs = 1
env_kwargs = dict(robot_uids="xarm7_leap_right", 
                #   reconfiguration_freq=1,
                sim_backend="gpu",
                # difficulty=3,
                  )

env = gym.make(
    "xArm7-v1-pick-randomized", # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
    num_envs=num_envs,
    obs_mode="state", # there is also "state_dict", "rgbd", ...
    control_mode="pd_joint_delta_pos", # there is also "pd_joint_delta_pos", ...
    render_mode="human",
    viewer_camera_configs=dict(fov=60),
    **env_kwargs
)

obs, _ = env.reset(seed=5)
done = False

while not done:
    for i in range(1):
        action = env.action_space.sample()

        # print(env.single_action_space.low)
        # print(env.single_action_space.high)
        # action = np.zeros((num_envs, 23))

        obs, reward, terminated, truncated, info = env.step(action)
        

        # print(env.unwrapped.cube.linear_velocity)
        # for i in range(3):
        #     print(env.unwrapped.agent.palm_contact_points)
        #     env.unwrapped.goal_site.set_pose(env.unwrapped.agent.palm_contact_points[0][i])
        #     env.render()
        
        env.render()  # a display is required to render


        # while True:
        #     env.render()
    env.reset()
    env.render()


env.close()