import os, sys
import torch
import numpy as np
import tqdm
import gymnasium as gym
from typing import Optional, List, Dict, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mani_skill.utils.wrappers.record import RecordEpisode


from train import Actor

def filter_state_keys(state: Dict, include_keys: Optional[List[str]] = None, 
                     exclude_keys: Optional[List[str]] = None) -> Dict:
    """Filter state dictionary to only include/exclude specified keys."""
    if include_keys is not None:
        return {k: v for k, v in state.items() if k in include_keys}
    elif exclude_keys is not None:
        return {k: v for k, v in state.items() if k not in exclude_keys}
    return state

def label_intermediate_states(
    task2_env_id: str,
    task2_checkpoint: str,
    task1_states_file: str,
    num_envs: int,
    control_mode: str,
    output_dir: str,
    video_output_dir: Optional[str] = None,
    include_keys: Optional[List[str]] = None,
    exclude_keys: Optional[List[str]] = None,
) -> Tuple[List[List[Dict]], List[List[Dict]]]:
    """
    Label Task 1 intermediate states by running Task 2 policy from each state.
        
    Args:
        task2_env_id: Task 2 environment ID
        task2_checkpoint: Path to trained Task 2 policy
        task1_states_file: Path to Task 1 intermediate states
        num_envs: Number of parallel environments
        control_mode: Control mode
        output_dir: Where to save labeled states
        include_keys: If provided, only save these state keys
        exclude_keys: If provided, exclude these state keys
        
    Returns:
        Tuple of (success_episodes, failure_episodes)
    """
    
    os.makedirs(output_dir, exist_ok=True)

    if video_output_dir is not None:
        os.makedirs(video_output_dir, exist_ok=True)
        print(f"Videos will be saved to: {video_output_dir}")
    
    # Load Task 1 states (each episode is a list of timesteps)
    print(f"Loading Task 1 states from {task1_states_file}")
    episodes = torch.load(task1_states_file)
    print(f"Loaded {len(episodes)} episodes")
    
    # Extract final timestep from each episode
    final_states = [episode[-1] for episode in episodes]
    print(f"Extracted {len(final_states)} final states to label")
    
    # Create Task 2 environment
    env_kwargs = dict(
        obs_mode="state",
        render_mode="rgb_array", 
        sim_backend="gpu",
        control_mode=control_mode,
        robot_uids="xarm7_leap_right",
        state_file_path=task1_states_file, 
    )
    
    env = gym.make(task2_env_id, num_envs=num_envs, reconfiguration_freq=1, **env_kwargs)
    
    num_rollout_steps = gym_utils.find_max_episode_steps_value(env)

    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    
    # Wrap with video recording if requested
    if video_output_dir is not None:
        env = RecordEpisode(
            env,
            output_dir=video_output_dir,
            save_trajectory=False,
            save_video=True,
            video_fps=30,
            max_steps_per_video=num_rollout_steps,
        )

    env = ManiSkillVectorEnv(env, num_envs, ignore_terminations=True, record_metrics=True)

    # Load Task 2 policy
    print(f"Loading Task 2 policy from {task2_checkpoint}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = Actor(env).to(device)
    ckpt = torch.load(task2_checkpoint)
    actor.load_state_dict(ckpt['actor'])
    actor.eval()
    
    # Label states
    success_episodes = []
    failure_episodes = []
    
    num_batches = (len(final_states) + num_envs - 1) // num_envs
    pbar = tqdm.tqdm(total=len(final_states), desc="Labeling states")
    
    current_idx = 0
    
    for batch_idx in range(num_batches):
        # Prepare batch
        batch_states = []
        batch_indices = []
        
        for i in range(num_envs):
            if current_idx >= len(final_states):
                break
            batch_states.append(final_states[current_idx])
            batch_indices.append(current_idx)
            current_idx += 1
        
        if len(batch_states) == 0:
            break
        
        num_valid = len(batch_states)
        
        # Pad batch if needed
        while len(batch_states) < num_envs:
            batch_states.append(batch_states[0])
        
        obs, info = env.reset(options={"intermediate_states": batch_states})
        
        # Rollout Task 2 policy
        episode_successes = torch.zeros(num_envs, dtype=torch.bool, device=device)
        
        for t in range(num_rollout_steps):
            with torch.no_grad():
                actions = actor.get_eval_action(obs)
            
            obs, rewards, terminated, truncated, infos = env.step(actions)
            
            # Check success on final step
            if t == num_rollout_steps - 1:
                top_level_success = infos.get("success", None)
                final_info_success = infos.get("final_info", {}).get("success", None)
                
                if top_level_success is not None and final_info_success is not None:
                    episode_successes = top_level_success | final_info_success
                elif top_level_success is not None:
                    episode_successes = top_level_success
                elif final_info_success is not None:
                    episode_successes = final_info_success
        
        # Categorize full epsiodes
        for i in range(num_valid):
            full_episode = episodes[batch_indices[i]]  # Get the full trajectory
            
            # Filter each timestep for t value function
            filtered_episode = [filter_state_keys(state, include_keys, exclude_keys) 
                               for state in full_episode]
            
            if episode_successes[i]:
                success_episodes.append(filtered_episode)
            else:
                failure_episodes.append(filtered_episode)
            
            pbar.update(1)
        
        # Print progress
        num_success = episode_successes[:num_valid].sum().item()
        print(f"Batch {batch_idx+1}/{num_batches}: {num_success}/{num_valid} successful")
    
    pbar.close()
    
    # Save labeled episodes (full trajectories)
    success_path = os.path.join(output_dir, "success_states.pt")
    failure_path = os.path.join(output_dir, "failure_states.pt")
    
    torch.save(success_episodes, success_path)
    torch.save(failure_episodes, failure_path)
    
    print("\n" + "="*60)
    print("STATE LABELING SUMMARY")
    print("="*60)
    print(f"Total episodes labeled: {len(episodes)}")
    print(f"Success episodes: {len(success_episodes)} ({len(success_episodes)/len(episodes)*100:.1f}%)")
    print(f"Failure episodes: {len(failure_episodes)} ({len(failure_episodes)/len(episodes)*100:.1f}%)")
    print(f"Saved to: {output_dir}")
    
    if success_episodes and success_episodes[0]:
        print(f"Saved state keys: {list(success_episodes[0][0].keys())}")
    
    print("="*60 + "\n")
    
    env.close()
    
    return success_episodes, failure_episodes

if __name__ == "__main__":
    
    SAVE_STATES = True

    task_name = "xArm7-v1-push"
    
    # List of (task2_checkpoint, task1_states_file) tuples
    labeling_configs = [
        (
            "runs/xArm7-v1-push__stage2_diff3__seed25__fingers_3_2_0_1_active_2_palm_False_seed_9/ckpt_4800512.pt",
            "intermediate_states/fingers_3_2_0_1_active_2_palm_False_seed_9/grasp_to_push.pt",
        )

    ]
    
    for task2_checkpoint, task1_states_file in labeling_configs:
        
        # Extract names from paths
        task2_dir = os.path.dirname(task2_checkpoint)
        task2_name = os.path.basename(task2_dir)
        
        task1_dir = os.path.dirname(task1_states_file)
        task1_name = os.path.basename(task1_dir)
        
        # Output directory: {intermediate_states_name}/{task2_checkpoint_name}
        output_dir = f"./relabeled_states/{task1_name}/{task2_name}"
        video_output_dir = f"./relabeled_states/{task1_name}/{task2_name}/videos"
        
        print(f"\n{'='*60}")
        print(f"Labeling configuration:")
        print(f"  Task 1 states: {task1_states_file}")
        print(f"  Task 2 checkpoint: {task2_checkpoint}")
        print(f"  Output directory: {output_dir}")
        print(f"{'='*60}\n")
        
        if not SAVE_STATES:
            print("NOT SAVING STATES")
        
        # Label states
        success_states, failure_states = label_intermediate_states(
            task2_env_id=task_name,
            task2_checkpoint=task2_checkpoint,
            task1_states_file=task1_states_file,
            num_envs=512,
            control_mode="pd_joint_delta_pos",
            output_dir=output_dir if SAVE_STATES else None,
            video_output_dir=None,
            include_keys=['cube', 'goal_site', 'xarm7_leap_right', 
                          'active_finger_indices', 'inactive_finger_indices',
                          'cube_half_sizes'],
            # exclude_keys=['tcp_pose'],
        )
    
    print("\nAll labeling complete!")
