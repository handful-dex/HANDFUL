"""
Curriculum training script for grasp selection across difficulty stages.
"""

import subprocess
import json
import statistics
from pathlib import Path


# ---------------------------------------------------------------------------
# Pruning strategies
# ---------------------------------------------------------------------------

def prune_to_n(n):
    def fn(metrics):
        ranked = sorted(metrics, key=lambda i: metrics[i], reverse=True)
        return ranked[:n]
    return fn


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ENV_ID = "xArm7-v1-two-pick"

STATE_CONFIGS = [
    "intermediate_states/fingers_0_1_2_3_active_1_palm_True_seed_10/grasp_to_push.pt",
    "intermediate_states/fingers_0_1_2_3_active_2_palm_True_seed_10/grasp_to_push.pt",
    "intermediate_states/fingers_0_2_1_3_active_2_palm_False_seed_10/grasp_to_push.pt",
    "intermediate_states/fingers_1_0_2_3_active_1_palm_True_seed_10/grasp_to_push.pt",
    "intermediate_states/fingers_1_2_0_3_active_2_palm_True_seed_10/grasp_to_push.pt",
    "intermediate_states/fingers_2_0_1_3_active_1_palm_True_seed_10/grasp_to_push.pt",
    "intermediate_states/fingers_3_0_1_2_active_2_palm_False_seed_10/grasp_to_push.pt",
    "intermediate_states/fingers_3_1_0_2_active_2_palm_False_seed_10/grasp_to_push.pt",
    "intermediate_states/fingers_3_2_0_1_active_2_palm_False_seed_10/grasp_to_push.pt",

]

ROBOT_UIDS = "xarm7_leap_right"

# Ordered list of (tensorboard_tag, higher_is_better) for ranking.
# First metric is primary decider, subsequent metrics break ties.
RANK_METRICS = [
    ("train/success_at_end", True),
    ("train/success_once",   True),
    ("train/return",         True),
]

STAGES = [
    dict(difficulty=1, total_timesteps="3_000_000",  learning_starts=51_200,  prune_fn=prune_to_n(6)),
    dict(difficulty=2, total_timesteps="2_409_600",  learning_starts=409_600, prune_fn=prune_to_n(3)),
    dict(difficulty=3, total_timesteps="5_409_600",  learning_starts=409_600, prune_fn=None),
]


# ---------------------------------------------------------------------------
# Metric + checkpoint helpers
# ---------------------------------------------------------------------------

def grasp_name_from_path(state_file_path):
    return Path(state_file_path).parent.name


def exp_name(env_id, stage_idx, difficulty, state_file_path, seed):
    grasp = grasp_name_from_path(state_file_path)
    return f"{env_id}__stage{stage_idx}_diff{difficulty}__seed{seed}__{grasp}"

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


def get_scalar_series(scalars_dict, tag):
    """Return list of (step, value) for a tag, or [] if missing."""
    if not scalars_dict:
        return []
    return scalars_dict.get(tag, [])


def get_best_metric_value(scalars_dict, tag, higher_is_better=True):
    """Return the best scalar value seen for a tag."""
    series = get_scalar_series(scalars_dict, tag)
    if not series:
        return None
    values = [v for _, v in series]
    return max(values) if higher_is_better else min(values)


def find_best_checkpoint(run_dir, rank_metrics=None):
    """
    Select the best checkpoint by the ordered RANK_METRICS list.

    Strategy:
      Score each checkpoint by its metric values at that step, using
      lexicographic tuple comparison so secondary metrics break ties in
      the primary metric, tertiary breaks ties in secondary, etc.

    Falls back to the final checkpoint if anything goes wrong.
    """
    if rank_metrics is None:
        rank_metrics = RANK_METRICS

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
    all_series = {tag: get_scalar_series(scalars_dict, tag) for tag, _ in rank_metrics}

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
    print(f"  [ckpt] Selected {best_ckpt.name} (step {ckpt_steps[best_idx]})")
    return str(best_ckpt)


def get_ranking_tuple(run_dir, rank_metrics=None):
    """
    Return a tuple of best metric values for ranking/pruning grasps.
    Higher is always better in the tuple (signs are flipped for lower_is_better metrics).
    Missing metrics default to -inf.
    """
    if rank_metrics is None:
        rank_metrics = RANK_METRICS

    scalars_dict = load_events_scalars(run_dir)
    result = []
    for tag, higher_is_better in rank_metrics:
        val = get_best_metric_value(scalars_dict, tag, higher_is_better)
        if val is None:
            val = float("-inf")
        # Normalise so that larger tuple value always means better
        result.append(val if higher_is_better else -val)
    return tuple(result)


# ---------------------------------------------------------------------------
# Training runner
# ---------------------------------------------------------------------------

def run_training(env_id, state_file_path, stage_idx, difficulty,
                 total_timesteps, learning_starts, seed, checkpoint=None):
    if not Path(state_file_path).exists():
        print(f"  ERROR: state file not found: {state_file_path}")
        return None

    name = exp_name(env_id, stage_idx, difficulty, state_file_path, seed)
    run_dir = Path("runs") / name

    if run_dir.exists():
        import shutil
        print(f"  [warn] Cleaning up aborted run directory: {run_dir}")
        shutil.rmtree(run_dir)

    cmd = [
        "python3", "train_cudagraphs.py",
        "--exp_name", name,
        "--seed", str(seed),
        "--env_id", env_id,
        "--state_file_path", state_file_path,
        "--total_timesteps", total_timesteps,
        "--learning_starts", str(learning_starts),
        "--difficulty", str(difficulty),
        "--no-evaluate",
    ]

    if checkpoint is not None:
        cmd += ["--checkpoint", checkpoint]

    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"  WARNING: training subprocess exited with code {result.returncode}")
        if find_best_checkpoint(str(run_dir)):
            print(f"  Checkpoint found despite non-zero exit — treating as successful.")
        else:
            print(f"  ERROR: No checkpoint found. Skipping grasp.")
            return None

    return str(run_dir)


# ---------------------------------------------------------------------------
# Main curriculum loop
# ---------------------------------------------------------------------------

def run_curriculum_pipeline(state_configs, env_id, seed, no_cull=False):
    """
    Orchestrates the multi-stage training curriculum.
    
    The curriculum trains multiple grasp configurations across a sequence of stages
    with increasing difficulty. Between stages, underperforming grasp configurations
    are culled (pruned) based on metrics extracted from TensorBoard logs.
    
    Args:
        state_configs: List of file paths to the intermediate state files (.pt)
        env_id: Target task environment identifier (e.g. xArm7-v1-push)
        seed: Execution random seed
        no_cull: If True, disables pruning and retains all configurations throughout
    """
    output_dir = f"curriculum_runs/{env_id}_seed{seed}"

    active = list(range(len(state_configs)))
    checkpoints = {i: None for i in range(len(state_configs))}
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results_path = Path(output_dir) / "curriculum_results.json"

    # Resumption logic: Load previous curriculum progression if a crash/restart occurred
    if results_path.exists():
        with open(results_path) as f:
            results_summary = json.load(f)
        print(f"  [resume] Loaded existing results from {results_path}")
    else:
        results_summary = {"env_id": env_id, "seed": seed, "no_cull": no_cull, "stages": []}

    completed_stages = len(results_summary.get("stages", []))
    if completed_stages > 0:
        print(f"  [resume] Found {completed_stages} completed stage(s), resuming...")
        for s in results_summary["stages"]:
            surviving_names = s["survivors"]
            active = [i for i in range(len(state_configs))
                      if grasp_name_from_path(state_configs[i]) in surviving_names]
        # Recover best checkpoints from disk for all surviving grasps
        for i in range(len(state_configs)):
            for stage_j in range(completed_stages):
                stage_cfg = STAGES[stage_j]
                run_dir = Path("runs") / exp_name(env_id, stage_j, stage_cfg["difficulty"], state_configs[i], seed)
                if run_dir.exists():
                    ckpt = find_best_checkpoint(str(run_dir))
                    if ckpt:
                        checkpoints[i] = ckpt

    # Main curriculum stage loop
    for stage_idx, stage in enumerate(STAGES):
        if stage_idx < completed_stages:
            continue

        difficulty   = stage["difficulty"]
        timesteps    = stage["total_timesteps"]
        learn_starts = stage["learning_starts"]
        # Default pruning function retains all if culling is disabled
        prune_fn     = (lambda m: list(m.keys())) if (no_cull or stage["prune_fn"] is None) else stage["prune_fn"]

        print(f"\n{'#'*70}")
        print(f"  STAGE {stage_idx+1}/{len(STAGES)}  |  difficulty={difficulty}  |  steps={timesteps}")
        print(f"  Active grasps ({len(active)}): {[grasp_name_from_path(state_configs[i]) for i in active]}")
        if no_cull:
            print("  [no_cull: pruning disabled]")
        print(f"{'#'*70}\n")

        # Map run directory paths to retrieve metrics later
        run_dirs = {}

        # Train each active configuration concurrently/sequentially
        for i in active:
            state_file = state_configs[i]
            grasp = grasp_name_from_path(state_file)
            print(f"\nTraining grasp: {grasp}")

            run_dir = run_training(
                env_id=env_id,
                state_file_path=state_file,
                stage_idx=stage_idx,
                difficulty=difficulty,
                total_timesteps=timesteps,
                learning_starts=learn_starts,
                seed=seed,
                checkpoint=checkpoints[i],
            )
            run_dirs[i] = run_dir

            # Select best checkpoint by RANK_METRICS (usually success rate and return)
            ckpt = find_best_checkpoint(run_dir) if run_dir else None
            if ckpt:
                checkpoints[i] = ckpt
                print(f"  --> {grasp}: best checkpoint = {Path(ckpt).name}")
            else:
                print(f"  [warn] No checkpoint found for {grasp}")

        # Build ranking tuples for pruning. The tuples contain metric values in the order
        # defined in RANK_METRICS, allowing lexicographical comparison of performance.
        stage_ranking = {i: get_ranking_tuple(run_dirs[i]) for i in active if run_dirs.get(i)}

        # For display and JSON storage, retrieve the primary metric (e.g. eval/success)
        primary_tag, primary_higher = RANK_METRICS[0]
        stage_metrics_display = {}
        for i in active:
            if run_dirs.get(i):
                scalars_dict = load_events_scalars(run_dirs[i])
                val = get_best_metric_value(scalars_dict, primary_tag, primary_higher)
                stage_metrics_display[i] = val if val is not None else 0.0
            else:
                stage_metrics_display[i] = 0.0

        # Execute pruning: survivors are determined by the stage's prune_fn 
        # based on their full lexicographical ranking tuples.
        active_primary = {i: stage_metrics_display[i] for i in active}
        survivors = prune_fn(stage_ranking)
        pruned    = [i for i in active if i not in survivors]

        # Sort surviving configs by full ranking tuple (highest performance first)
        survivors = sorted(survivors,
                           key=lambda i: stage_ranking.get(i, (float("-inf"),) * len(RANK_METRICS)),
                           reverse=True)
        survivors_set = set(survivors)

        print(f"\n--- Stage {stage_idx+1} Results (primary: {primary_tag}) ---")
        for i in sorted(active,
                        key=lambda i: stage_ranking.get(i, (float("-inf"),) * len(RANK_METRICS)),
                        reverse=True):
            marker = "KEEP " if i in survivors_set else "PRUNE"
            rank_str = ", ".join(f"{v:.4f}" for v in stage_ranking.get(i, ()))
            print(f"  [{marker}] {grasp_name_from_path(state_configs[i])}: ({rank_str})")

        results_summary["stages"].append({
            "stage":      stage_idx + 1,
            "difficulty": difficulty,
            "metrics": {
                grasp_name_from_path(state_configs[i]): {
                    tag: get_best_metric_value(load_events_scalars(run_dirs[i]), tag, h)
                    for tag, h in RANK_METRICS
                }
                for i in active if run_dirs.get(i)
            },
            "survivors": [grasp_name_from_path(state_configs[i]) for i in survivors],
            "pruned":    [grasp_name_from_path(state_configs[i]) for i in pruned],
            "best_checkpoints": {
                grasp_name_from_path(state_configs[i]): checkpoints[i]
                for i in active
            },
        })

        active = survivors

        with open(results_path, "w") as f:
            json.dump(results_summary, f, indent=2)

        if not active:
            print("[error] All grasps pruned. Stopping early.")
            break

    # Final summary
    print(f"\n{'#'*70}")
    print("  CURRICULUM COMPLETE")
    print(f"{'#'*70}")
    last_stage = results_summary["stages"][-1]
    header = f"{'Configuration':<60} " + "  ".join(tag.split("/")[-1] for tag, _ in RANK_METRICS)
    print(header)
    print('-' * len(header))
    for grasp, metric_dict in sorted(last_stage["metrics"].items(),
                                     key=lambda x: x[1].get(RANK_METRICS[0][0], 0),
                                     reverse=True):
        vals = "  ".join(f"{metric_dict.get(tag, 0.0):<10.4f}" for tag, _ in RANK_METRICS)
        print(f"{grasp:<60} {vals}")

    results_summary["final_survivors"]   = [grasp_name_from_path(state_configs[i]) for i in active]
    results_summary["final_checkpoints"] = {grasp_name_from_path(state_configs[i]): checkpoints[i] for i in active}

    with open(results_path, "w") as f:
        json.dump(results_summary, f, indent=2)

    print(f"\nResults saved to: {results_path}")

    try:
        import matplotlib
        matplotlib.use('Agg')
        try:
            from curriculum.visualize_curriculum import visualize
        except ImportError:
            from visualize_curriculum import visualize
        visualize(str(results_path), show=False)
    except Exception as e:
        print(f"  [warn] Visualization failed: {e}")


def main(seed, no_cull=False):
    run_curriculum_pipeline(STATE_CONFIGS, ENV_ID, seed, no_cull)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--no_cull", action="store_true")
    args = parser.parse_args()
    main(seed=args.seed, no_cull=args.no_cull)