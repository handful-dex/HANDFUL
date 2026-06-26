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


from mani_skill.utils import gym_utils
from mani_skill.utils.structs.pose import Pose

from transition_feasibility.load_trajectories import load_trajectory_from_h5


replay_mode = "action" # Can be: "state", "state_action", "action"


num_envs = 1
env_kwargs = dict(robot_uids="xarm7_leap_right", 
                #   reconfiguration_freq=1,
                sim_backend="gpu",
                  )

env = gym.make(
    "xArm7-v1-two-pick", # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
    num_envs=num_envs,
    obs_mode="state", # there is also "state_dict", "rgbd", ...
    control_mode="pd_joint_pos", # there is also "pd_joint_delta_pos", ...
    render_mode="human",
    viewer_camera_configs=dict(fov=60),
    **env_kwargs
)
env2 = gym.make(
    "xArm7-v1-two-pick", # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
    num_envs=num_envs,
    obs_mode="state", # there is also "state_dict", "rgbd", ...
    control_mode="pd_joint_pos", # there is also "pd_joint_delta_pos", ...
    render_mode="human",
    viewer_camera_configs=dict(fov=60),
    **env_kwargs
)

env3 = gym.make(
    "xArm7-v1-two-pick", # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
    num_envs=num_envs,
    obs_mode="state", # there is also "state_dict", "rgbd", ...
    control_mode="pd_joint_delta_pos", # there is also "pd_joint_delta_pos", ...
    render_mode="human",
    viewer_camera_configs=dict(fov=60),
    **env_kwargs
)

obs, _ = env.reset(seed=1)
done = False


traj = load_trajectory_from_h5(
    "multi_task_trajectories/pick_then_pick/trajectories.h5",
    trajectory_idx=4
)

qpos_traj = traj["qpos"]
qvel_traj = traj["qvel"]
actions = traj["actions"]
cube = traj["cube"]
push_block = traj["push_block"]


while True:
    obs, _ = env.reset()
    obs, _ = env2.reset()
    obs, _ = env3.reset()
    
    # Convert cube pose to tensor on correct device
    cube_pose = torch.from_numpy(cube[0]).to(env.unwrapped.agent.robot.device)
    cube_pose = cube_pose.unsqueeze(0)
    env.unwrapped.cube.set_pose(Pose.create(cube_pose))
    env2.unwrapped.cube.set_pose(Pose.create(cube_pose))
    env3.unwrapped.cube.set_pose(Pose.create(cube_pose))


    push_block_pose = torch.from_numpy(push_block[80]).to(env.unwrapped.agent.robot.device)
    push_block_pose = push_block_pose.unsqueeze(0)
    env.unwrapped.push_block.set_pose(Pose.create(push_block_pose))
    env2.unwrapped.push_block.set_pose(Pose.create(push_block_pose))
    env3.unwrapped.push_block.set_pose(Pose.create(push_block_pose))

    qpos_tensor = torch.from_numpy(qpos_traj[0]).to(env.unwrapped.agent.robot.device).unsqueeze(0)
    qvel_tensor = torch.from_numpy(qvel_traj[0]).to(env.unwrapped.agent.robot.device).unsqueeze(0)
    env.unwrapped.agent.robot.set_qpos(qpos_tensor)
    env.unwrapped.agent.robot.set_qvel(qvel_tensor)
    env2.unwrapped.agent.robot.set_qpos(qpos_tensor)
    env2.unwrapped.agent.robot.set_qvel(qvel_tensor)
    env3.unwrapped.agent.robot.set_qpos(qpos_tensor)
    env3.unwrapped.agent.robot.set_qvel(qvel_tensor)
    
    if env.unwrapped.gpu_sim_enabled:
        env.unwrapped.scene._gpu_apply_all()
        env.unwrapped.scene.px.gpu_update_articulation_kinematics()
        env.unwrapped.scene._gpu_fetch_all()

        env2.unwrapped.scene._gpu_apply_all()
        env2.unwrapped.scene.px.gpu_update_articulation_kinematics()
        env2.unwrapped.scene._gpu_fetch_all()

        env3.unwrapped.scene._gpu_apply_all()
        env3.unwrapped.scene.px.gpu_update_articulation_kinematics()
        env3.unwrapped.scene._gpu_fetch_all()
    
    for t in range(len(qpos_traj)):
        # If you want to set robot state directly:
        qpos_tensor = torch.from_numpy(qpos_traj[t]).to(env.unwrapped.agent.robot.device).unsqueeze(0)
        qvel_tensor = torch.from_numpy(qvel_traj[t]).to(env.unwrapped.agent.robot.device).unsqueeze(0)
        env.unwrapped.agent.robot.set_qpos(qpos_tensor)
        env.unwrapped.agent.robot.set_qvel(qvel_tensor)
        
        if env.unwrapped.gpu_sim_enabled:
            env.unwrapped.scene._gpu_apply_all()
            env.unwrapped.scene.px.gpu_update_articulation_kinematics()
            env.unwrapped.scene._gpu_fetch_all()

            env2.unwrapped.scene._gpu_apply_all()
            env2.unwrapped.scene.px.gpu_update_articulation_kinematics()
            env2.unwrapped.scene._gpu_fetch_all()

            env3.unwrapped.scene._gpu_apply_all()
            env3.unwrapped.scene.px.gpu_update_articulation_kinematics()
            env3.unwrapped.scene._gpu_fetch_all()

        # env.step(None)

        qpos_target = torch.from_numpy(qpos_traj[t]).to(env.unwrapped.agent.robot.device)
        action = qpos_target.unsqueeze(0)
        obs, reward, terminated, truncated, info = env2.step(action)

        obs, reward, terminated, truncated, info = env3.step(actions[t])
            
        env.render()
        env2.render()
        env3.render()