import os, sys
import torch
import numpy as np
import tqdm
import gymnasium as gym
from typing import Optional, List, Dict

import h5py

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.utils import gym_utils

from load_trajectories import print_h5_summary
from train import Actor, Args


def collect_trajectories_with_contact_monitoring(
    env_id: str,
    checkpoint: str,
    num_envs: int = 32,
    num_trajectories: int = 20,
    control_mode: str = "pd_joint_delta_pos",
    output_dir: str = "trajectory_collection",
    hand_table_force_threshold: float = 1.0,
    track_objects: List[str] = ['cube'],
    record: bool = True,
    finger_selection: Optional[str] = None,
    num_active_fingers: Optional[int] = None,
    palm_use: Optional[bool] = None,
):
    """
    Collect trajectories with contact monitoring and video recording.
    Flags trajectories that exceed hand-table contact threshold.

    Args:
        env_id: Environment ID
        checkpoint: Path to trained model
        num_envs: Number of parallel environments
        num_trajectories: Number of successful trajectories to collect
        control_mode: Control mode
        output_dir: Directory to save videos and trajectory data
        hand_table_force_threshold: Max allowed hand-table force (N)
        track_objects: List of object names to track (e.g., ['cube', 'push_block', 'button']).
            Objects are tracked if they exist in the environment.
        record: Whether to record videos
        extra_kwargs: Optional dict of extra kwargs for gym.make
        finger_selection: Optional list of finger indices for success criteria (e.g., [3,0,1,2])
        num_active_fingers: Optional number of active fingers required for success
        palm_use: Optional boolean for whether palm contact is required for success
    """

    video_dir = os.path.join(output_dir, "videos")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    
    # Setup environment
    env_kwargs = dict(
        obs_mode="state",
        render_mode="rgb_array",
        sim_backend="gpu",
        control_mode=control_mode,
        robot_uids="xarm7_leap_right",
    )
    
    # Add finger config args if provided
    if finger_selection is not None:
        if ',' in finger_selection:
            env_kwargs["finger_selection"] = [int(x) for x in finger_selection.split(',')]
        else:
            env_kwargs["finger_selection"] = finger_selection

        env_kwargs["num_active_fingers"] = num_active_fingers
        env_kwargs["palm_use"] = palm_use
    
    envs = gym.make(
        env_id,
        num_envs=num_envs,
        reconfiguration_freq=1,
        human_render_camera_configs=dict(shader_pack="default"),
        **env_kwargs
    )

    num_eval_steps = gym_utils.find_max_episode_steps_value(envs)

    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)

    if record:
        # Use ManiSkill's RecordEpisode wrapper
        envs = RecordEpisode(
            envs,
            output_dir=video_dir,
            save_trajectory=False,
            save_video=True,
            video_fps=60,
            max_steps_per_video=num_eval_steps,
            info_on_video=True
        )

    envs = ManiSkillVectorEnv(envs, num_envs, ignore_terminations=True, record_metrics=True)

    assert isinstance(envs.get_wrapper_attr('single_action_space'), gym.spaces.Box), "only continuous action space is supported"

    # Load actor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = Actor(envs).to(device)
    ckpt = torch.load(checkpoint)
    actor.load_state_dict(ckpt['actor'])
    actor.eval()
    
    # Collection loop
    trajectories_saved = 0
    total_episodes = 0
    all_trajectories = []
    problematic_trajectories = []
    
    pbar = tqdm.tqdm(total=num_trajectories, desc="Collecting trajectories")
    
    while trajectories_saved < num_trajectories:
        obs, infos = envs.reset()
        
        # Track data for this batch
        batch_trajectories = {
            i: {
                'trajectory': [],
                'max_hand_table_force': 0.0,
            }
            for i in range(num_envs)
        }
        successful_envs = [False] * num_envs

        for t in range(num_eval_steps):
            with torch.no_grad():
                actions = actor.get_eval_action(obs)
            
            # Store trajectory data for all environments
            for i in range(num_envs):
                timestep_data = {
                    'timestep': t,
                    'qpos': envs.unwrapped.agent.robot.get_qpos()[i].clone().cpu(),
                    'qvel': envs.unwrapped.agent.robot.get_qvel()[i].clone().cpu(),
                    'action': actions[i].clone().cpu(),
                    'tcp_pose': envs.unwrapped.agent.tcp_pose.raw_pose[i].clone().cpu(),
                    'hand_table_force': infos.get('hand_table_force', torch.zeros(num_envs, device=device))[i].item(),
                    'cube_half_sizes': envs.unwrapped.cube_half_sizes_per_env[i].clone().cpu(),
                }
                
                # Track all specified objects if they exist in the environment
                for obj_name in track_objects:
                    if hasattr(envs.unwrapped, obj_name):
                        obj = getattr(envs.unwrapped, obj_name)
                        timestep_data[f'{obj_name}'] = obj.pose.raw_pose[i].clone().cpu()
                
                batch_trajectories[i]['trajectory'].append(timestep_data)
                
                # Track max force
                batch_trajectories[i]['max_hand_table_force'] = max(
                    batch_trajectories[i]['max_hand_table_force'],
                    timestep_data['hand_table_force']
                )
            
            obs, rewards, terminated, truncated, infos = envs.step(actions)
            
            # Check success at the end
            if t == num_eval_steps - 1:
                success_mask = _get_success_mask(infos)
                
                if success_mask is not None:
                    for i in range(num_envs):
                        if success_mask[i]:
                            successful_envs[i] = True
        
        # Save successful trajectories
        for i, success in enumerate(successful_envs):
            if success and trajectories_saved < num_trajectories:
                max_force = batch_trajectories[i]['max_hand_table_force']
                
                # FILTER: skip high-contact trajectories
                if max_force > hand_table_force_threshold:
                    problematic_trajectories.append({
                        'video_index': total_episodes + i,
                        'max_force': max_force
                    })
                    continue 
                
                traj_dict = {
                    'trajectory': batch_trajectories[i]['trajectory'],
                    'trajectory_length': len(batch_trajectories[i]['trajectory']),
                    'max_hand_table_force': max_force,
                    'video_index': total_episodes + i,  # For finding the video file
                }
                
                all_trajectories.append(traj_dict)
                trajectories_saved += 1
                pbar.update(1)
        
        total_episodes += num_envs
    
    pbar.close()
    
    # Save trajectory data to HDF5 - extract run name from checkpoint for naming
    run_name = os.path.basename(os.path.dirname(checkpoint))
    h5_filename = f"{run_name}.h5"
    h5_path = os.path.join(output_dir, h5_filename)
    _save_trajectories_to_h5(
        h5_path,
        all_trajectories,
        env_id,
        control_mode,
        hand_table_force_threshold,
        problematic_trajectories,
        track_objects
    )
    
    # Print summary
    _print_summary(
        total_episodes,
        all_trajectories,
        problematic_trajectories,
        hand_table_force_threshold,
        output_dir,
        h5_path
    )
    
    envs.close()
    return all_trajectories, problematic_trajectories


def _get_success_mask(infos: Dict) -> Optional[torch.Tensor]:
    """Extract success mask from info dict."""
    top_level_success = infos.get("success", None)
    final_info_success = infos.get("final_info", {}).get("success", None)
    
    if top_level_success is not None and final_info_success is not None:
        return top_level_success | final_info_success
    elif top_level_success is not None:
        return top_level_success
    elif final_info_success is not None:
        return final_info_success

    return None


def _save_trajectories_to_h5(
    h5_path: str,
    trajectories: List,
    env_id: str,
    control_mode: str,
    force_threshold: float,
    problematic: List,
    track_objects: List[str],
):
    """Save trajectories to HDF5 file."""
    with h5py.File(h5_path, 'w') as f:
        # Global metadata
        f.attrs['num_trajectories'] = len(trajectories)
        f.attrs['env_id'] = env_id
        f.attrs['control_mode'] = control_mode
        f.attrs['hand_table_force_threshold'] = force_threshold
        f.attrs['num_problematic'] = len(problematic)
        
        for idx, traj_dict in enumerate(trajectories):
            # Create group for this trajectory
            traj_group = f.create_group(f'trajectory_{idx:04d}')
            
            # Stack all timesteps into arrays
            trajectory = traj_dict['trajectory']
            traj_group.create_dataset('qpos', data=np.stack([t['qpos'].numpy() for t in trajectory]), compression='gzip')
            traj_group.create_dataset('qvel', data=np.stack([t['qvel'].numpy() for t in trajectory]), compression='gzip')
            traj_group.create_dataset('tcp_pose', data=np.stack([t['tcp_pose'].numpy() for t in trajectory]), compression='gzip')
            traj_group.create_dataset('actions', data=np.stack([t['action'].numpy() for t in trajectory]), compression='gzip')
            traj_group.create_dataset('hand_table_force', data=np.array([t['hand_table_force'] for t in trajectory]), compression='gzip')
            traj_group.create_dataset('cube_half_sizes', data=np.array([t['cube_half_sizes'].numpy() for t in trajectory]), compression='gzip')
            
            # Automatically save any tracked object positions
            # Check what object keys exist in the first timestep
            first_timestep = trajectory[0]
            object_keys = [key for key in track_objects]
            
            # Save each object's positions
            for key in object_keys:
                # Collect positions where they exist, use NaN for timesteps where object doesn't exist
                positions = []
                for t in trajectory:
                    if key in t:
                        positions.append(t[key])
                    else:
                        # Object doesn't exist at this timestep, use NaN
                        positions.append(np.full(7, np.nan))
                
                traj_group.create_dataset(key, data=np.stack(positions), compression='gzip')
            
            # Save metadata as attributes
            traj_group.attrs['trajectory_length'] = traj_dict['trajectory_length']
            traj_group.attrs['max_hand_table_force'] = traj_dict['max_hand_table_force']
            traj_group.attrs['video_index'] = traj_dict['video_index']


def _print_summary(
    total_episodes: int,
    successful_trajectories: List,
    problematic: List,
    force_threshold: float,
    output_dir: str,
    h5_path: str
):
    """Print collection summary."""
    print("\n" + "="*60)
    print("COLLECTION SUMMARY")
    print("="*60)
    print(f"Total episodes run: {total_episodes}")
    print(f"Successful trajectories collected: {len(successful_trajectories)}")
    print(f"Trajectories exceeding force threshold ({force_threshold} N): {len(problematic)}")
    
    if problematic:
        print(f"\nPROBLEMATIC TRAJECTORIES (High hand-table contact):")
        print("-" * 60)
        for p in problematic:
            print(f"  Trajectory {p['video_index']}: Max force = {p['max_force']:.3f} N")
    
    # Contact statistics
    if successful_trajectories:
        all_forces = [t['max_hand_table_force'] for t in successful_trajectories]
        print(f"\nHand-Table Contact Statistics (successful trajectories):")
        print(f"  Max force: {max(all_forces):.3f} N")
        print(f"  Mean max force: {sum(all_forces)/len(all_forces):.3f} N")
        print(f"  Min force: {min(all_forces):.3f} N")
    
    print(f"\nData saved to: {h5_path}")
    print(f"Videos saved to: {output_dir}/videos/")
    print("="*60 + "\n")


if __name__ == "__main__":
    args = Args()
    
    env_id = "xArm7-v1-pick-randomized"

    checkpoints = [
                ("")
                # ("runs/fingers_0_1_2_3_active_1_palm_True_seed_10/ckpt_5801984.pt", [0,1,2,3], 1, True),
                # ("runs/fingers_0_1_2_3_active_2_palm_True_seed_9/ckpt_5101568.pt", [0,1,2,3], 2, True),
                # ("runs/fingers_0_2_1_3_active_2_palm_False_seed_9/ckpt_5900288.pt", [0,2,1,3], 2, False),
                # ("runs/fingers_1_0_2_3_active_1_palm_True_seed_9/ckpt_4900864.pt", [1,0,2,3], 1, True),
                # ("runs/fingers_1_2_0_3_active_2_palm_True_seed_9/ckpt_5601280.pt", [1,2,0,3], 2, True),
                # ("runs/fingers_2_0_1_3_active_1_palm_True_seed_9/ckpt_5701632.pt", [2,0,1,3], 1, True),
                # ("runs/fingers_3_0_1_2_active_2_palm_False_seed_9/ckpt_5001216.pt", [3,0,1,2], 2, False),
                # ("runs/fingers_3_1_0_2_active_2_palm_False_seed_9/ckpt_5601280.pt", [3,1,0,2], 2, False),
                # ("runs/fingers_3_2_0_1_active_2_palm_False_seed_9/ckpt_5900288.pt", [3,2,0,1], 2, False),
                ]


    for i, (checkpoint, fingers, num_active_fingers, palm_use) in enumerate(checkpoints, 1):
        
        if checkpoint is None:
            raise ValueError("Please provide a checkpoint path")
        
        # Extract run name from checkpoint path
        dir_name = os.path.dirname(checkpoint)
        run_name = os.path.basename(dir_name)
        output_dir = f"sim2real_trajectories/{run_name}"
        
        print(f"Saving trajectories to: {output_dir}")

        # Collect trajectories
        trajectories, problematic = collect_trajectories_with_contact_monitoring(
            env_id=env_id,
            checkpoint=checkpoint,
            num_envs=100,
            num_trajectories=50,
            control_mode=args.control_mode,
            output_dir=output_dir,
            hand_table_force_threshold=5,
            track_objects=['cube'],
            record=False,
            finger_selection=fingers,
            num_active_fingers=num_active_fingers, 
            palm_use=palm_use, 
        )

        # Print HDF5 file summary
        run_name = os.path.basename(os.path.dirname(checkpoint))
        h5_path = os.path.join(output_dir, f"{run_name}.h5")
        print_h5_summary(h5_path)