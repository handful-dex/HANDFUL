import subprocess
import os
from pathlib import Path
import argparse


ENV_ID = "xArm7-v1-two-pick"

STATE_CONFIGS = [
    # "intermediate_states/fingers_0_1_2_3_active_1_palm_True_seed_10/grasp_to_push.pt",
    # "intermediate_states/fingers_0_1_2_3_active_2_palm_True_seed_9/grasp_to_push.pt",
    # "intermediate_states/fingers_0_2_1_3_active_2_palm_False_seed_9/grasp_to_push.pt",
    # "intermediate_states/fingers_1_0_2_3_active_1_palm_True_seed_9/grasp_to_push.pt",
    # "intermediate_states/fingers_1_2_0_3_active_2_palm_True_seed_9/grasp_to_push.pt",
    # "intermediate_states/fingers_2_0_1_3_active_1_palm_True_seed_9/grasp_to_push.pt",
    # "intermediate_states/fingers_3_0_1_2_active_2_palm_False_seed_9/grasp_to_push.pt",
    "intermediate_states/fingers_3_1_0_2_active_2_palm_False_seed_9/grasp_to_push.pt",
    "intermediate_states/fingers_3_2_0_1_active_2_palm_False_seed_9/grasp_to_push.pt",
    # "intermediate_states/xArm7-v1-pick-all__train__9__1771746645/grasp_to_push.pt"
]

TOTAL_TIMESTEPS = "10_000_000"
ROBOT_UIDS = "xarm7_leap_right"


def extract_config_from_path(state_file_path):
    return Path(state_file_path).parent.name


def run_training(env_id, state_file_path, seed, total_timesteps):
    if not Path(state_file_path).exists():
        print(f"ERROR: State file not found: {state_file_path}")
        return None

    config_name = extract_config_from_path(state_file_path)
    exp_name = f"{env_id}_seed_{seed}_{config_name}"
    run_dir = f"runs/{exp_name}"

    cmd = [
        "python3", "train_cudagraphs.py",
        "--exp_name", exp_name,
        "--seed", str(seed),
        "--env_id", env_id,
        "--state_file_path", state_file_path,
        "--total_timesteps", str(total_timesteps),
        "--checkpoint", "None",
        "--no-evaluate",
    ]

    result = subprocess.run(cmd)

    final_ckpt = Path(run_dir) / "final_ckpt.pt"
    return run_dir if final_ckpt.exists() else None


def get_final_metrics(run_dir):
    if not run_dir:
        return {}
    try:
        from tensorboard.backend.event_processing import event_accumulator
        events_file = list(Path(run_dir).glob("events.out.tfevents.*"))[0]
        ea = event_accumulator.EventAccumulator(str(events_file))
        ea.Reload()

        metrics = {}
        for tag in ['eval/success_once', 'eval/return']:
            if tag in ea.Tags()['scalars']:
                metrics[tag] = ea.Scalars(tag)[-1].value
        return metrics
    except Exception as e:
        print(f"Failed to load metrics: {e}")
        return {}


def run_second_task_pipeline(state_configs, env_id, seed, total_timesteps):
    results = []

    print(f"Training task: {env_id}")
    print(f"Seed: {seed}")
    print(f"Number of configurations: {len(state_configs)}\n")

    for i, state_file_path in enumerate(state_configs, 1):
        config_name = extract_config_from_path(state_file_path)
        print(f"\nTraining config {i}/{len(state_configs)}: {config_name}")
        print(f"  State file: {state_file_path}")

        run_dir = run_training(env_id, state_file_path, seed, total_timesteps)
        metrics = get_final_metrics(run_dir)

        results.append({
            'config': config_name,
            'state_file': state_file_path,
            'status': 'success',
            'metrics': metrics,
            'run_dir': run_dir,
        })

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=10)
    args = parser.parse_args()
    seed = args.seed

    results = run_second_task_pipeline(STATE_CONFIGS, ENV_ID, seed, TOTAL_TIMESTEPS)

    print(f"\n{'='*100}")
    print(f"TRAINING COMPLETE - {ENV_ID}")
    print('='*100)
    print(f"{'Configuration':<60} {'Success':<10} {'Return':<10}")
    print('-'*100)
    for r in results:
        success = r.get('metrics', {}).get('eval/success_once', 0)
        ret = r.get('metrics', {}).get('eval/return', 0)
        print(f"{r['config']:<60} {success:<10.3f} {ret:<10.2f}")
    print('='*100)


if __name__ == "__main__":
    main()