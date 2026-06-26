import subprocess
import os
from pathlib import Path
import re

# Configuration
PALM_CONFIG = [
    # (fingers, num_active, palm_use)
    ([3,0,1,2], 2, False),
    ([3,1,0,2], 2, False),
    ([3,2,0,1], 2, False),
    ([0,1,2,3], 2, True),
    ([0,2,1,3], 2, False),
    ([0,2,1,3], 2, True),
    ([1,2,0,3], 2, True),
    ([1,0,2,3], 1, True),
    ([2,0,1,3], 1, True),
    ([0,1,2,3], 1, True),
]
SEED = 10
TOTAL_TIMESTEPS = "6_000_000"
ENV_ID = "xArm7-v1-pick-randomized"


def load_events_scalars(run_dir):
    """Load all tfevents files in a directory and merge their scalar series."""
    try:
        from tensorboard.backend.event_processing import event_accumulator
        events_files = sorted(Path(run_dir).glob("events.out.tfevents.*"))
        if not events_files:
            print(f"  [warn] No tfevents file in {run_dir}")
            return None
        
        merged = {}
        for f in events_files:
            try:
                ea = event_accumulator.EventAccumulator(str(f))
                ea.Reload()
                tags = ea.Tags().get("scalars", [])
                for tag in tags:
                    if tag not in merged:
                        merged[tag] = {}
                    # Later files overwrite values for the same step from earlier files
                    for e in ea.Scalars(tag):
                        merged[tag][e.step] = e.value
            except Exception as e:
                print(f"  [warn] Could not load events from {f}: {e}")
                
        # Convert to list of (step, value) sorted by step
        for tag in merged:
            merged[tag] = sorted(merged[tag].items(), key=lambda x: x[0])
            
        return merged
    except Exception as e:
        print(f"  [warn] Could not process events in {run_dir}: {e}")
        return None


def find_best_picking_checkpoint(run_dir):
    """Select the best checkpoint by evaluation success metrics."""
    rank_metrics = [
        ("eval/success_at_end", True),
        ("eval/success_once",   True),
        ("eval/return",         True),
    ]

    run_path = Path(run_dir)
    final = run_path / "final_ckpt.pt"

    ckpts = sorted(run_path.glob("ckpt_*.pt"),
                   key=lambda p: int(p.stem.split("_")[1]))
    if not ckpts:
        return str(final) if final.exists() else None

    ckpt_steps = [int(p.stem.split("_")[1]) for p in ckpts]

    scalars_dict = load_events_scalars(run_dir)
    if scalars_dict is None:
        return str(final) if final.exists() else str(ckpts[-1])

    # Pre-load all series
    all_series = {tag: scalars_dict.get(tag, []) for tag, _ in rank_metrics}

    def ckpt_score(ckpt_step):
        score = []
        for tag, higher_is_better in rank_metrics:
            series = all_series[tag]
            if not series:
                val = float("-inf")
            else:
                nearest_val = min(series, key=lambda x: abs(x[0] - ckpt_step))[1]
                val = nearest_val if higher_is_better else -nearest_val
            score.append(val)
        return tuple(score)

    best_idx = max(range(len(ckpts)), key=lambda i: ckpt_score(ckpt_steps[i]))
    best_ckpt = ckpts[best_idx]
    
    print(f"  [ckpt] Selected {best_ckpt.name} (step {ckpt_steps[best_idx]}) for {run_path.name}")
    return str(best_ckpt)


def run_training(fingers, num_active_fingers, palm_use, seed, total_timesteps, env_id):
    """Run training for a finger configuration."""
    exp_name = f"fingers_{'_'.join(map(str, fingers))}_active_{num_active_fingers}_palm_{palm_use}_seed_{seed}"
    run_dir = f"runs/{exp_name}"
    
    cmd = [
        "python", "train_cudagraphs.py",
        "--exp_name", exp_name,
        "--seed", str(seed),
        "--env_id", env_id,
        "--finger_selection", ','.join(map(str, fingers)),
        "--num_active_fingers", str(num_active_fingers),
        "--total_timesteps", str(total_timesteps),
        "--checkpoint", "None",
        "--no-evaluate",
    ]
    cmd.append("--palm_use" if palm_use else "--no-palm-use")
    subprocess.run(cmd)
    
    final_ckpt = Path(run_dir) / "final_ckpt.pt"
    if final_ckpt.exists():
        best_ckpt = find_best_picking_checkpoint(run_dir)
        return run_dir, best_ckpt
    else:
        return None, None


def get_final_metrics(run_dir):
    """Get final eval metrics from tensorboard."""
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


def run_picking_pipeline(palm_configs, seed, total_timesteps, env_id):
    """Run the picking training pipeline on a list of configurations."""
    results = []
    
    for i, (fingers, num_active_fingers, palm_use) in enumerate(palm_configs, 1):
        print(f"\nTraining config {i}/{len(palm_configs)}: fingers={fingers}, active={num_active_fingers}, palm={palm_use}")
        
        run_dir, final_ckpt = run_training(fingers, num_active_fingers, palm_use, seed, total_timesteps, env_id)
        
        if run_dir:
            metrics = get_final_metrics(run_dir)
        else:
            metrics = {}
            
        results.append({
            'fingers': fingers,
            'num_active': num_active_fingers,
            'palm': palm_use,
            'status': 'success' if final_ckpt else 'failed',
            'metrics': metrics,
            'run_dir': run_dir,
            'final_ckpt': final_ckpt
        })
        
    return results


def main():
    results = run_picking_pipeline(PALM_CONFIG, SEED, TOTAL_TIMESTEPS, ENV_ID)
    
    # Print summary
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print('='*80)
    print(f"{'Fingers':<20} {'Active':<8} {'Palm':<8} {'Success':<10} {'Return':<10}")
    print('-'*80)
    for r in results:
        fingers_str = '_'.join(map(str, r['fingers']))
        success = r.get('metrics', {}).get('eval/success_once', 0)
        ret = r.get('metrics', {}).get('eval/return', 0)
        print(f"{fingers_str:<20} {r['num_active']:<8} {str(r['palm']):<8} {success:<10.3f} {ret:<10.2f}")
    print('='*80)


if __name__ == "__main__":
    main()