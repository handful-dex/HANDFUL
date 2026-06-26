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


def _make_envs(config, num_envs, num_rollout_steps, video_dir, task_idx, record):
    """Helper to build (and optionally wrap with RecordEpisode) a vectorised env."""
    env_kwargs = dict(
        obs_mode="state",
        render_mode="rgb_array",
        sim_backend="gpu",
        control_mode=config['control_mode'],
        robot_uids="xarm7_leap_right",
    )
    if 'extra_kwargs' in config:
        env_kwargs.update(config['extra_kwargs'])

    envs = gym.make(
        config['env_id'],
        num_envs=num_envs,
        reconfiguration_freq=1,
        human_render_camera_configs=dict(shader_pack="default"),
        **env_kwargs
    )

    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)

    envs = ManiSkillVectorEnv(envs, num_envs, ignore_terminations=True, record_metrics=True)

    if record:
        task_video_dir = os.path.join(video_dir, f"task_{task_idx}")
        os.makedirs(task_video_dir, exist_ok=True)
        envs = RecordEpisode(
            envs,
            output_dir=task_video_dir,
            save_trajectory=False,
            save_video=True,
            video_fps=60,
            max_steps_per_video=num_rollout_steps,
            info_on_video=False
        )

    return envs


def collect_multi_task_trajectories(
    env_configs: List[Dict],
    checkpoints: List[str],
    num_envs: int = 512,
    num_trajectories: int = 100,
    task_name: str = "two_pick",
    output_dir: str = "multi_task_trajectories",
    hand_table_force_threshold: float = 1.0,
    track_objects: List[str] = ['cube'],
    stable_success_steps: int = 3,
    record: bool = True,
):
    """
    Collect multi-task trajectories by chaining multiple tasks together.

    Loops over batches of parallel environments until `num_trajectories`
    successful (all-task) trajectories have been collected.

    Args:
        env_configs: List of dicts, each containing:
            - env_id: Environment ID
            - control_mode: Control mode
            - extra_kwargs: Optional dict of extra kwargs for gym.make
              For tasks after the first you can include a 'difficulty' key
              here and it will be forwarded to gym.make as well.
        checkpoints: List of checkpoint paths (one per task)
        num_envs: Number of parallel environments per batch
        num_trajectories: Target number of successful full-chain trajectories
        task_name: Used for the output HDF5 filename
        output_dir: Directory to save videos and trajectory data
        hand_table_force_threshold: Max allowed hand-table force (N)
        track_objects: Object names to track (must exist as env attributes)
        stable_success_steps: Consecutive success steps before marking done
        record: Whether to record videos
    """

    assert len(env_configs) == len(checkpoints), "Must have one checkpoint per task"
    num_tasks = len(env_configs)

    video_dir = os.path.join(output_dir, "videos")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pre-load all actors once — they don't change between batches
    actors = []
    for task_idx, (cfg, ckpt_path) in enumerate(zip(env_configs, checkpoints)):
        print(f"Loading policy {task_idx + 1}/{num_tasks} from {ckpt_path}")
        # We need an env to infer the action-space shape; build a throw-away one.
        _tmp_kwargs = dict(
            obs_mode="state",
            render_mode="rgb_array",
            sim_backend="gpu",
            control_mode=cfg['control_mode'],
            robot_uids="xarm7_leap_right",
        )
        if 'extra_kwargs' in cfg:
            _tmp_kwargs.update(cfg['extra_kwargs'])
        _tmp_env = gym.make(cfg['env_id'], num_envs=1, **_tmp_kwargs)
        if isinstance(_tmp_env.action_space, gym.spaces.Dict):
            _tmp_env = FlattenActionSpaceWrapper(_tmp_env)
        _tmp_env = ManiSkillVectorEnv(_tmp_env, 1, ignore_terminations=True, record_metrics=True)
        actor = Actor(_tmp_env).to(device)
        _tmp_env.close()

        ckpt = torch.load(ckpt_path)
        actor.load_state_dict(ckpt['actor'])
        actor.eval()
        actors.append(actor)

    # -----------------------------------------------------------------------
    # Collection loop — keeps going until we have enough good trajectories
    # -----------------------------------------------------------------------
    all_successful_trajectories = []
    all_problematic_trajectories = []
    total_batches = 0

    outer_pbar = tqdm.tqdm(total=num_trajectories, desc="Collecting trajectories")

    while len(all_successful_trajectories) < num_trajectories:
        total_batches += 1
        print(f"\n{'='*60}")
        print(f"Batch {total_batches}  |  collected so far: "
              f"{len(all_successful_trajectories)}/{num_trajectories}")
        print(f"{'='*60}")

        # Per-env bookkeeping for this batch
        batch_trajectories = {
            i: {
                'tasks': [[] for _ in range(num_tasks)],
                'max_hand_table_force': 0.0,
                'task_success': [False] * num_tasks,
            }
            for i in range(num_envs)
        }

        # ------------------------------------------------------------------
        # Run each task stage
        # ------------------------------------------------------------------
        envs = None
        intermediate_states = None  # list[dict | None], length num_envs

        for task_idx in range(num_tasks):
            current_config = env_configs[task_idx]
            print(f"\n  Task {task_idx + 1}/{num_tasks}: {current_config['env_id']}")

            if envs is not None:
                envs.close()

            # Build envs for this stage
            _tmp_kwargs = dict(
                obs_mode="state",
                render_mode="rgb_array",
                sim_backend="gpu",
                control_mode=current_config['control_mode'],
                robot_uids="xarm7_leap_right",
            )
            if 'extra_kwargs' in current_config:
                _tmp_kwargs.update(current_config['extra_kwargs'])

            envs = gym.make(
                current_config['env_id'],
                num_envs=num_envs,
                reconfiguration_freq=1,
                human_render_camera_configs=dict(shader_pack="default"),
                **_tmp_kwargs
            )

            num_rollout_steps = gym_utils.find_max_episode_steps_value(envs)

            if isinstance(envs.action_space, gym.spaces.Dict):
                envs = FlattenActionSpaceWrapper(envs)

            if record:
                task_video_dir = os.path.join(video_dir, f"task_{task_idx}")
                os.makedirs(task_video_dir, exist_ok=True)
                envs = RecordEpisode(
                    envs,
                    output_dir=task_video_dir,
                    save_trajectory=False,
                    save_video=True,
                    video_fps=60,
                    max_steps_per_video=num_rollout_steps,
                    info_on_video=True
                )

            envs = ManiSkillVectorEnv(envs, num_envs, ignore_terminations=True, record_metrics=True)

            if task_idx == 0:
                obs, infos = envs.reset(seed=total_batches)
            else:
                print("  Loading end states from previous task as initial states...")
                obs, infos = envs.reset(seed=total_batches, options={"intermediate_states": intermediate_states})

            actor = actors[task_idx]

            pbar = tqdm.tqdm(total=num_rollout_steps, desc=f"  Task {task_idx + 1}", leave=False)

            env_completed = torch.zeros(num_envs, dtype=torch.bool, device=device)
            consecutive_success_steps = torch.zeros(num_envs, dtype=torch.int32, device=device)
            intermediate_states = [None] * num_envs

            for t in range(num_rollout_steps - 1):
                with torch.no_grad():
                    actions = actor.get_eval_action(obs)

                # Store trajectory data for environments not yet done
                for i in range(num_envs):
                    if env_completed[i]:
                        continue

                    timestep_data = {
                        'task_idx': task_idx,
                        'timestep': t,
                        'qpos': envs.unwrapped.agent.robot.get_qpos()[i].clone().cpu(),
                        'qvel': envs.unwrapped.agent.robot.get_qvel()[i].clone().cpu(),
                        'action': actions[i].clone().cpu(),
                        'tcp_pose': envs.unwrapped.agent.tcp_pose.raw_pose[i].clone().cpu(),
                        'hand_table_force': infos.get(
                            'hand_table_force', torch.zeros(num_envs, device=device)
                        )[i].item(),
                        'cube_half_sizes': envs.unwrapped.cube_half_sizes_per_env[i].clone().cpu(),
                    }

                    for obj_name in track_objects:
                        if hasattr(envs.unwrapped, obj_name):
                            obj = getattr(envs.unwrapped, obj_name)
                            timestep_data[obj_name] = obj.pose.raw_pose[i].clone().cpu()

                    batch_trajectories[i]['tasks'][task_idx].append(timestep_data)
                    batch_trajectories[i]['max_hand_table_force'] = max(
                        batch_trajectories[i]['max_hand_table_force'],
                        timestep_data['hand_table_force']
                    )

                obs, rewards, terminated, truncated, infos = envs.step(actions)
                pbar.update(1)

                success_mask = _get_success_mask(infos)
                if success_mask is not None:
                    consecutive_success_steps = torch.where(
                        success_mask,
                        consecutive_success_steps + 1,
                        torch.zeros_like(consecutive_success_steps)
                    )

                    newly_stable = (consecutive_success_steps >= stable_success_steps) & ~env_completed

                    if newly_stable.any():
                        num_newly = newly_stable.sum().item()
                        for i in range(num_envs):
                            if newly_stable[i] and intermediate_states[i] is None:
                                intermediate_states[i] = _collect_single_env_state(
                                    envs, i, track_objects
                                )
                                batch_trajectories[i]['task_success'][task_idx] = True

                        env_completed = env_completed | newly_stable
                        num_done = env_completed.sum().item()
                        print(f"\n  [Step {t+1}] +{num_newly} stable completions → "
                              f"{num_done}/{num_envs} done")

                        if env_completed.all():
                            print(f"  All envs finished early at step {t+1}!")
                            break

                # Collect final state for any env that never succeeded
                if t >= num_rollout_steps - 2:
                    for i in range(num_envs):
                        if intermediate_states[i] is None:
                            intermediate_states[i] = _collect_single_env_state(
                                envs, i, track_objects
                            )

            pbar.close()

            task_successes = sum(
                batch_trajectories[i]['task_success'][task_idx] for i in range(num_envs)
            )
            print(f"  Task {task_idx + 1}: {task_successes}/{num_envs} "
                  f"succeeded ({task_successes / num_envs * 100:.1f}%)")

        if envs is not None:
            envs.close()

        # ------------------------------------------------------------------
        # Filter batch results and accumulate
        # ------------------------------------------------------------------
        for i in range(num_envs):
            traj_data = batch_trajectories[i]

            if not all(traj_data['task_success']):
                continue  # didn't complete every stage

            max_force = traj_data['max_hand_table_force']
            if max_force > hand_table_force_threshold:
                all_problematic_trajectories.append({
                    'env_idx': i,
                    'batch': total_batches,
                    'max_force': max_force,
                })
                continue

            full_trajectory = []
            for task_traj in traj_data['tasks']:
                full_trajectory.extend(task_traj)

            all_successful_trajectories.append({
                'trajectory': full_trajectory,
                'trajectory_length': len(full_trajectory),
                'max_hand_table_force': max_force,
                'video_index': (total_batches - 1) * num_envs + i,
            })
            outer_pbar.update(1)

            if len(all_successful_trajectories) >= num_trajectories:
                break  # got what we need; don't overshoot

    outer_pbar.close()

    # Trim to exactly num_trajectories in case we slightly overshot
    successful_trajectories = all_successful_trajectories[:num_trajectories]

    # Save
    h5_path = os.path.join(output_dir, f"{task_name}.h5")
    _save_trajectories_to_h5(
        h5_path,
        successful_trajectories,
        env_configs,
        hand_table_force_threshold,
        all_problematic_trajectories,
        track_objects,
    )

    _print_summary(
        total_batches * num_envs,
        successful_trajectories,
        all_problematic_trajectories,
        hand_table_force_threshold,
        output_dir,
        h5_path,
    )

    return successful_trajectories, all_problematic_trajectories


# ---------------------------------------------------------------------------
# Helpers (unchanged from original)
# ---------------------------------------------------------------------------

def _collect_single_env_state(envs, env_idx: int, track_objects: List[str]) -> Dict:
    """Collect end states from a single environment for the next task."""
    current_states = envs.unwrapped.get_state_dict()
    tcp_poses = envs.unwrapped.agent.tcp_pose.raw_pose
    cube_half_sizes = envs.unwrapped.cube_half_sizes_per_env

    active_finger_indices = None
    inactive_finger_indices = None
    if hasattr(envs.unwrapped, "active_finger_indices"):
        active_finger_indices = envs.unwrapped.active_finger_indices
        inactive_finger_indices = envs.unwrapped.inactive_finger_indices

    env_state = {}

    for object_name, tensor in current_states.get('actors', {}).items():
        env_state[object_name] = tensor[env_idx].clone()

    for object_name, tensor in current_states.get('articulations', {}).items():
        env_state[object_name] = tensor[env_idx].clone()

    env_state['tcp_pose'] = tcp_poses[env_idx].clone()
    env_state['cube_half_sizes'] = cube_half_sizes[env_idx].clone()

    if active_finger_indices is not None:
        env_state['active_finger_indices'] = active_finger_indices[env_idx].clone()
        env_state['inactive_finger_indices'] = inactive_finger_indices[env_idx].clone()

    return env_state


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
    env_configs: List,
    force_threshold: float,
    problematic: List,
    track_objects: List[str],
):
    """Save trajectories to HDF5 file (same format as single-task collector)."""
    with h5py.File(h5_path, 'w') as f:
        f.attrs['num_trajectories'] = len(trajectories)
        f.attrs['env_id'] = '+'.join([cfg['env_id'] for cfg in env_configs])
        f.attrs['control_mode'] = env_configs[0]['control_mode']
        f.attrs['hand_table_force_threshold'] = force_threshold
        f.attrs['num_problematic'] = len(problematic)

        for idx, traj_dict in enumerate(trajectories):
            traj_group = f.create_group(f'trajectory_{idx:04d}')
            trajectory = traj_dict['trajectory']

            traj_group.create_dataset('qpos',
                data=np.stack([t['qpos'].numpy() for t in trajectory]), compression='gzip')
            traj_group.create_dataset('qvel',
                data=np.stack([t['qvel'].numpy() for t in trajectory]), compression='gzip')
            traj_group.create_dataset('tcp_pose',
                data=np.stack([t['tcp_pose'].numpy() for t in trajectory]), compression='gzip')
            traj_group.create_dataset('actions',
                data=np.stack([t['action'].numpy() for t in trajectory]), compression='gzip')
            traj_group.create_dataset('hand_table_force',
                data=np.array([t['hand_table_force'] for t in trajectory]), compression='gzip')
            traj_group.create_dataset('cube_half_sizes',
                data=np.array([t['cube_half_sizes'].numpy() for t in trajectory]), compression='gzip')

            for key in track_objects:
                positions = []
                for t in trajectory:
                    positions.append(t[key] if key in t else np.full(7, np.nan))
                traj_group.create_dataset(key,
                    data=np.stack(positions), compression='gzip')

            traj_group.attrs['trajectory_length'] = traj_dict['trajectory_length']
            traj_group.attrs['max_hand_table_force'] = traj_dict['max_hand_table_force']
            traj_group.attrs['video_index'] = traj_dict['video_index']


def _print_summary(
    total_episodes: int,
    successful_trajectories: List,
    problematic: List,
    force_threshold: float,
    output_dir: str,
    h5_path: str,
):
    """Print collection summary."""
    print("\n" + "="*60)
    print("MULTI-TASK COLLECTION SUMMARY")
    print("="*60)
    print(f"Total episodes run:              {total_episodes}")
    print(f"Successful trajectories saved:   {len(successful_trajectories)}")
    print(f"Overall success rate:            "
          f"{len(successful_trajectories)/max(total_episodes,1)*100:.1f}%")
    print(f"Trajectories exceeding force threshold ({force_threshold} N): {len(problematic)}")

    if problematic:
        print(f"\nPROBLEMATIC TRAJECTORIES (high hand-table contact):")
        print("-" * 60)
        for p in problematic:
            print(f"  Batch {p['batch']} Env {p['env_idx']}: max force = {p['max_force']:.3f} N")

    if successful_trajectories:
        all_forces = [t['max_hand_table_force'] for t in successful_trajectories]
        print(f"\nHand-Table Contact Statistics (successful trajectories):")
        print(f"  Max:  {max(all_forces):.3f} N")
        print(f"  Mean: {sum(all_forces)/len(all_forces):.3f} N")
        print(f"  Min:  {min(all_forces):.3f} N")

    print(f"\nData saved to: {h5_path}")
    print(f"Videos saved to: {output_dir}/videos/")
    print("="*60 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = Args()

    SEED         = 9
    TASK1_ENV_ID = "xArm7-v1-pick-all"
    TASK2_ENV_ID = "xArm7-v1-two-pick-whole-hand"
    TASK_NAME    = TASK2_ENV_ID

    runs = [

        dict(
            fingers=[3,2,0,1], num_active=2, palm_use=True,
            task1_ckpt="runs/xArm7-v1-pick-all__train__9__1771746645/ckpt_5900288.pt",
            task2_ckpt="runs/video_data/two_pick_whole_hand_seed10/xArm7-v1-two-pick-whole-hand__stage2_diff3__seed10__xArm7-v1-pick-all__train__9__1771746645/ckpt_5300224.pt",
            task2_difficulty=1,
        ),
    ]

    for run in runs:
        fingers    = run['fingers']
        num_active = run['num_active']
        palm_use   = run['palm_use']

        # Derive the canonical grasp name the same way train_pick.py does
        grasp_name = (
            f"fingers_{'_'.join(map(str, fingers))}"
            f"_active_{num_active}"
            f"_palm_{palm_use}"
            f"_seed_{SEED}"
        )

        env_configs = [
            {
                # ----------------------------------------------------------
                # Task 1: pick with this grasp configuration
                # ----------------------------------------------------------
                'env_id': TASK1_ENV_ID,
                'control_mode': args.control_mode,
                # 'extra_kwargs': {
                #     'finger_selection': fingers,
                #     'num_active_fingers': num_active,
                #     'palm_use': palm_use,
                # },
            },
            {
                # ----------------------------------------------------------
                # Task 2: two-pick, initialised from task 1 live end-states
                # ----------------------------------------------------------
                'env_id': TASK2_ENV_ID,
                'control_mode': args.control_mode,
                # 'extra_kwargs': {
                #     'difficulty': run['task2_difficulty'],
                # },
            },
        ]

        checkpoints = [run['task1_ckpt'], run['task2_ckpt']]
        output_dir  = f"multi_task_trajectories/{TASK_NAME}/{grasp_name}"

        print(f"\n{'#'*70}")
        print(f"  Grasp : {grasp_name}")
        print(f"  Output: {output_dir}")
        print(f"{'#'*70}")

        trajectories, problematic = collect_multi_task_trajectories(
            env_configs=env_configs,
            checkpoints=checkpoints,
            num_envs=10,
            num_trajectories=10,
            task_name=TASK_NAME,
            output_dir=output_dir,
            hand_table_force_threshold=10.0,
            track_objects=['cube', 'push_block'],
            stable_success_steps=5,
            record=True,
        )

        print_h5_summary(h5_path=os.path.join(output_dir, f"{TASK_NAME}.h5"))