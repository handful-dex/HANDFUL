"""
Sequential policy rollout in unified environment.
Demonstrates how to run task 1 policy for 75 steps, then task 2 policy for remaining steps.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import gymnasium as gym
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

from mani_skill.utils.wrappers.record import RecordEpisode

from train import Actor  # Your policy class


def sequential_policy_rollout(
    env_id: str = "xArm7-v1-two-pick-unified",
    task1_checkpoint: str = "runs/xArm7-v1-pick-randomized__train__8__1769321319/final_ckpt.pt",
    task2_checkpoint: str = "runs/xArm7-v1-two-pick__train__8__1769748840/final_ckpt.pt",
    num_envs: int = 1,
    num_episodes: int = 10,
    task1_steps: int = 75,
    total_steps: int = 175,
    record: bool = False,
    video_dir: str = "videos/sequential_rollout",
):
    """
    Run two different policies sequentially in a unified environment.
    
    Args:
        env_id: Unified environment ID
        task1_checkpoint: Path to task 1 policy checkpoint
        task2_checkpoint: Path to task 2 policy checkpoint
        num_envs: Number of parallel environments
        num_episodes: Number of episodes to run
        task1_steps: Number of steps to run task 1 policy
        total_steps: Total episode length
        record: Whether to record videos
        video_dir: Directory to save recorded videos
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    env_kwargs = dict(
                    difficulty=1,
                    )

    # Create unified environment
    env = gym.make(
        env_id,
        num_envs=num_envs,
        obs_mode="state_dict",
        control_mode="pd_joint_delta_pos",
        render_mode="rgb_array",  # rgb_array is required for RecordEpisode
        sim_backend="gpu",
        robot_uids="xarm7_leap_right",
        human_render_camera_configs=dict(shader_pack="default"),
        **env_kwargs
    )

    env1 = gym.make(
        "xArm7-v1-pick-randomized",
        num_envs=num_envs,
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        render_mode="rgb_array",
        sim_backend="gpu",
        robot_uids="xarm7_leap_right",
    )

    env2 = gym.make(
        "xArm7-v1-cabinet-drawer",
        num_envs=num_envs,
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        render_mode="rgb_array",
        sim_backend="gpu",
        robot_uids="xarm7_leap_right",
    )

    if record:
        finger_str = os.path.basename(os.path.dirname(task1_checkpoint))
        video_dir = os.path.join(video_dir, env_id, finger_str)
        os.makedirs(video_dir, exist_ok=True)
        env = RecordEpisode(
            env,
            output_dir=video_dir,
            save_trajectory=False,
            save_video=True,
            video_fps=60,
            max_steps_per_video=total_steps,
            info_on_video=False,
        )


    # Load task 1 policy
    print(f"Loading task 1 policy from {task1_checkpoint}")
    task1_actor = Actor(env1).to(device)
    task1_ckpt = torch.load(task1_checkpoint)
    task1_actor.load_state_dict(task1_ckpt['actor'])
    task1_actor.eval()
    
    # Load task 2 policy  
    print(f"Loading task 2 policy from {task2_checkpoint}")
    task2_actor = Actor(env2).to(device)
    task2_ckpt = torch.load(task2_checkpoint)
    task2_actor.load_state_dict(task2_ckpt['actor'])
    task2_actor.eval()
    
    success_count = 0
    
    for episode in range(num_episodes):
        obs, info = env.reset(seed=episode)
        obs_tensor = env.get_split_observation(info)

        episode_reward = 0
        
        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'='*60}")
        
        for t in range(total_steps):
            
            # Switch between task policies based on step count
            if t < task1_steps:
                if t == 0:
                    print(f"Running task 1 policy (steps 0-{task1_steps-1})")
                
                # Filter and flatten observation for task 1
                flat_obs = flatten_state_dict(obs_tensor)
                
                with torch.no_grad():
                    action = task1_actor.get_eval_action(flat_obs)
            
            else:
                if t == task1_steps:
                    print(f"Switching to task 2 policy (steps {task1_steps}-{total_steps-1})")
                
                # Filter and flatten observation for task 2
                flat_obs = flatten_state_dict(obs_tensor)
                
                with torch.no_grad():
                    action = task2_actor.get_eval_action(flat_obs)
            
            # Execute action
            obs, reward, terminated, truncated, info = env.step(action)
            obs_tensor = env.get_split_observation(info)
            episode_reward += reward.mean().item()
            
            # Print progress every 25 steps
            if (t + 1) % 25 == 0:
                print(f"  Step {t+1}/{total_steps}, Reward: {reward.mean().item():.3f}")
        
        # Batched success tensor [num_envs]
        success_tensor = info.get(
            "success",
            torch.zeros(num_envs, dtype=torch.bool, device=device)
        )

        # Count successes in this batch
        batch_successes = success_tensor.sum().item()
        success_count += batch_successes

        print(
            f"Episode {episode + 1}: "
            f"{batch_successes}/{num_envs} envs successful"
        )

    
    total_trials = num_episodes * num_envs

    print(f"\n{'='*60}")
    print(
        f"RESULTS: {success_count}/{total_trials} successes "
        f"({success_count / total_trials * 100:.1f}%)"
    )
    print(f"{'='*60}\n")

    if record:
        print(f"Videos saved to: {video_dir}")

    env.close()
    return success_count / total_trials



def flatten_state_dict(state_dict):
    """
    Flatten a nested dict of tensors into a single concatenated tensor.
    Handles nested dicts, and casts integer tensors to float before concat.
    """
    tensors = []
    
    def collect(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                collect(v)
        elif isinstance(obj, torch.Tensor):
            t = obj.float()          # cast int tensors (e.g. picking_fingers, phase)
            tensors.append(t.reshape(t.shape[0], -1))  # (batch, -1)
    
    collect(state_dict)
    return torch.cat(tensors, dim=-1)


if __name__ == "__main__":

    success_rate = sequential_policy_rollout(
        env_id="xArm7-v1-cabinet-drawer-unified",
        task1_checkpoint="runs/fingers_0_1_2_3_active_1_palm_True_seed_10/ckpt_5900288.pt",
        task2_checkpoint="runs/video_data/xArm7-v1-cabinet-drawer__stage2_diff3__seed26__fingers_0_1_2_3_active_1_palm_True_seed_10/ckpt_5400576.pt",
        num_envs=1,
        num_episodes=3,
        task1_steps=75,
        total_steps=175,
        record=True,
        video_dir="videos/sequential_rollout",
    )