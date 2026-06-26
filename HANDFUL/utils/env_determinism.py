import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import gymnasium as gym
import mani_skill.envs
import numpy as np
import time

from envs.tasks.xarm7_leap_pick_env import XArm7TableTop
from envs.tasks.xarm7_leap_push_env import XArm7TableTopPush
from envs.tasks.xarm7_press_button import XArm7TableTopPress
from envs.tasks.xarm7_pick_randomized import XArm7TableTopPickRandomized
from envs.tasks.xarm7_pick_all import XArm7TableTopPickAll
from envs.tasks.xarm7_two_pick import XArm7TableTopTwoPick
from envs.tasks.xarm7_twist import XArm7TableTopKnobTwist
from envs.tasks.xarm7_open_drawer import XArm7CabinetDrawerEnv, XArm7CabinetDoorEnv


from mani_skill.utils import gym_utils
from mani_skill.utils.structs.pose import Pose

from transition_feasibility.load_trajectories import load_trajectory_from_h5


num_envs = 1
env_kwargs = dict(robot_uids="xarm7_leap_right", 
                  reconfiguration_freq=1,
                sim_backend="gpu",
                reward_mode="dense"
                  )

env_id = "xArm7-v1-cabinet-drawer"

env = gym.make(
    env_id, # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
    num_envs=num_envs,
    obs_mode="state", # there is also "state_dict", "rgbd", ...
    control_mode="pd_joint_delta_pos", # there is also "pd_joint_delta_pos", ...
    render_mode="human",
    viewer_camera_configs=dict(fov=60),
    **env_kwargs
)

env2 = gym.make(
    env_id, # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
    num_envs=num_envs,
    obs_mode="state", # there is also "state_dict", "rgbd", ...
    control_mode="pd_joint_delta_pos", # there is also "pd_joint_delta_pos", ...
    render_mode="human",
    viewer_camera_configs=dict(fov=60),
    **env_kwargs
)

obs, _ = env.reset(seed=1)
obs, _ = env2.reset(seed=1)
done = False

while not done:
    for i in range(100):
        action = env.action_space.sample()
        action = env2.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        obs2, reward, terminated, truncated, info = env2.step(action)
        env.reset()
        env2.reset()
        
        print(env.get_state_dict())
        print(env2.get_state_dict())



        print(f"Reset matches: {torch.allclose(obs, obs2)}")
        time.sleep(1)



        # print(env.unwrapped.cube.linear_velocity)
        env.render()  # a display is required to render
        env2.render()  # a display is required to render



    env.reset()
    env.render()


env.close()
env2.close()