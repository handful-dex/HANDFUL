import argparse
import sys
import subprocess
import os
from pathlib import Path

# Force headless execution by unsetting DISPLAY to bypass X11 "Maximum clients" limit
os.environ.pop("DISPLAY", None)

from train_picking import run_picking_pipeline, find_best_picking_checkpoint
from collect_intermediate_states_grouped import run_collection_pipeline
from curriculum.curriculum_train import run_curriculum_pipeline
from train_second_task import run_second_task_pipeline

# Here, specify which fingers and palm configurations you want to use.
# Format: (fingers, num_active, palm)
# fingers: list of finger indices (0 is index, 1 is middle, 2 is ring, 3 is thumb)
# num_active: number of active fingers - (indexes 0 to num_active are active)
# palm: whether palm is active

# For example: ([3,0,1,2], 2, False) corresponds to thumb and index finger active (grasping) 
# with other fingers inactive, and no palm use.
DEFAULT_PALM_CONFIGS = [
    ([3,0,1,2], 2, False),
    ([3,1,0,2], 2, False),
    ([3,2,0,1], 2, False),
    ([0,1,2,3], 2, True),
    ([0,2,1,3], 2, False),
    ([1,2,0,3], 2, True),
    ([1,0,2,3], 1, True),
    ([2,0,1,3], 1, True),
    ([0,1,2,3], 1, True),
]

def main():
    parser = argparse.ArgumentParser(description="Unified Training Pipeline")
    parser.add_argument("--mode", type=str, default="curriculum", 
                        choices=["curriculum", "whole_hand", "unified", "no_curriculum"],
                        help="The training pipeline mode to run.")
    parser.add_argument("--seed", type=int, default=10, help="Random seed for the pipeline.")
    
    # Picking Training Options
    parser.add_argument("--grasps", type=int, nargs="+", default=None,
                        help="Space-separated indices of grasp configs to train (0-9). Defaults to all 9.")
    parser.add_argument("--picking_env", type=str, default="xArm7-v1-pick-randomized", help="Environment for picking.")
    parser.add_argument("--picking_timesteps", type=str, default="6_000_000", help="Total timesteps for picking.")
    
    # State Collection Options
    parser.add_argument("--states_to_save", type=int, default=16384)
    parser.add_argument("--steps_to_save", type=int, default=10)
    parser.add_argument("--save_videos", action="store_true")
    parser.add_argument("--min_episodes_for_cutoff", type=int, default=5000,
                        help="Minimum number of episodes run before checking for success rate cutoff.")
    parser.add_argument("--success_rate_cutoff", type=float, default=15.0,
                        help="Success rate threshold (in %%) below which state collection is terminated.")
    
    # Second Task Options
    parser.add_argument("--second_task_env", type=str, default="xArm7-v1-push", help="Environment for second task / curriculum.")
    parser.add_argument("--second_task_timesteps", type=str, default="10_000_000", help="Total timesteps for no_curriculum mode.")
    parser.add_argument("--no_cull", action="store_true", help="Disable pruning in curriculum mode.")
    parser.add_argument("--skip_existing", action="store_true", help="Skip training and collection if files already exist on disk.")
    
    args = parser.parse_args()

    print(f"{'='*80}")
    print(f"STARTING PIPELINE: Mode = {args.mode.upper()}")
    print(f"Seed: {args.seed}")
    print(f"{'='*80}\n")

    # Determine which grasps to use
    if args.grasps is not None:
        try:
            palm_configs = [DEFAULT_PALM_CONFIGS[i] for i in args.grasps]
        except IndexError:
            print("ERROR: Invalid grasp indices. Must be between 0 and 9.")
            sys.exit(1)
    else:
        palm_configs = DEFAULT_PALM_CONFIGS

    if args.mode == "unified":
        print(f"--- PHASE 1: UNIFIED TRAINING ---")
        print(f"Training {len(palm_configs)} configurations on unified env {args.second_task_env}-unified...")
        # For unified mode, we run train_cudagraphs directly and we are done.
        for i, (fingers, num_active, palm) in enumerate(palm_configs, 1):
            print(f"\nUnified config {i}/{len(palm_configs)}: fingers={fingers}, active={num_active}, palm={palm}")
            exp_name = f"unified_fingers_{'_'.join(map(str, fingers))}_active_{num_active}_palm_{palm}_seed_{args.seed}"
            cmd = [
                "python", "train_cudagraphs.py",
                "--exp_name", exp_name,
                "--seed", str(args.seed),
                "--env_id", args.second_task_env + "-unified" if not args.second_task_env.endswith("-unified") else args.second_task_env,
                "--finger_selection", ','.join(map(str, fingers)),
                "--num_active_fingers", str(num_active),
                "--total_timesteps", "16_000_000",
                "--checkpoint", "None",
                "--no-evaluate",
            ]
            cmd.append("--palm_use" if palm else "--no-palm-use")
            subprocess.run(cmd)
        
        print(f"\n{'='*80}")
        print("PIPELINE COMPLETE (Unified Mode)")
        print(f"{'='*80}")
        sys.exit(0)

    collection_configs = []
    already_collected_paths = []

    # 1. Grasp Training
    print("\n--- PHASE 1: GRASP TRAINING ---")
    if args.mode == "whole_hand":
        env_id = "xArm7-v1-pick-all"
        exp_name = f"whole_hand_seed_{args.seed}"
        state_file = f"intermediate_states/{exp_name}/grasp_to_push.pt"
        
        if args.skip_existing and Path(state_file).exists():
            print(f"State file {state_file} already exists. Skipping training and collection.")
            already_collected_paths.append(state_file)
        else:
            print(f"Training whole hand picking on {env_id}...")
            run_dir = f"runs/{exp_name}"
            cmd = [
                "python", "train_cudagraphs.py",
                "--exp_name", exp_name,
                "--seed", str(args.seed),
                "--env_id", env_id,
                "--total_timesteps", "6_000_000",
                "--checkpoint", "None",
                "--no-evaluate",
            ]
            subprocess.run(cmd)
            
            final_ckpt = Path(run_dir) / "final_ckpt.pt"
            if final_ckpt.exists():
                best_ckpt = find_best_picking_checkpoint(run_dir)
                collection_configs.append((best_ckpt, [0,1,2,3], 4, True))
    else:
        configs_to_train = []
        for fingers, num_active, palm in palm_configs:
            exp_name = f"fingers_{'_'.join(map(str, fingers))}_active_{num_active}_palm_{palm}_seed_{args.seed}"
            state_file = f"intermediate_states/{exp_name}/grasp_to_push.pt"
            if args.skip_existing and Path(state_file).exists():
                print(f"State file {state_file} already exists. Skipping training and collection.")
                already_collected_paths.append(state_file)
            else:
                configs_to_train.append((fingers, num_active, palm))
                
        if configs_to_train:
            print(f"Training {len(configs_to_train)} grasp configurations on {args.picking_env}...")
            picking_results = run_picking_pipeline(configs_to_train, args.seed, args.picking_timesteps, args.picking_env)
            for r in picking_results:
                if r['status'] == 'success' and r.get('final_ckpt'):
                    collection_configs.append((r['final_ckpt'], r['fingers'], r['num_active'], r['palm']))
            
    if not collection_configs and not already_collected_paths:
        print("\nERROR: No grasps trained successfully and no pre-existing states found. Pipeline terminating.")
        sys.exit(1)
        
    if collection_configs:
        print(f"\n{len(collection_configs)} checkpoints ready for state collection.")

    # 2. State Collection
    print("\n--- PHASE 2: STATE COLLECTION ---")
    if collection_configs:
        print(f"Collecting intermediate states for {len(collection_configs)} checkpoints...")
        picking_env = "xArm7-v1-pick-all" if args.mode == "whole_hand" else args.picking_env
        state_file_paths = run_collection_pipeline(
            collection_configs=collection_configs,
            env_id=picking_env,
            states_to_save=args.states_to_save,
            steps_to_save=args.steps_to_save,
            save_videos=args.save_videos,
            min_episodes_for_cutoff=args.min_episodes_for_cutoff,
            success_rate_cutoff=args.success_rate_cutoff
        )
        newly_collected_paths = [
            f"{r['output_dir']}/grasp_to_push.pt"
            for r in state_file_paths
            if Path(f"{r['output_dir']}/grasp_to_push.pt").exists()
        ]
    else:
        newly_collected_paths = []
        
    successful_state_paths = already_collected_paths + newly_collected_paths

    if not successful_state_paths:
        print("\nERROR: No intermediate states collected successfully. Pipeline terminating.")
        sys.exit(1)

    print(f"\nGenerated {len(successful_state_paths)} total state files.")

    # 3. Second Task Training
    print(f"\n--- PHASE 3: SECOND TASK ({args.mode}) ---")
    second_task_env = args.second_task_env
    if args.mode == "whole_hand":
        second_task_env = second_task_env + "-whole-hand" if not second_task_env.endswith("-whole-hand") else second_task_env
    print(f"Target Environment: {second_task_env}")
    
    if args.mode == "no_curriculum":
        run_second_task_pipeline(
            state_configs=successful_state_paths,
            env_id=second_task_env,
            seed=args.seed,
            total_timesteps=args.second_task_timesteps
        )
    elif args.mode in ["curriculum", "whole_hand"]:
        run_curriculum_pipeline(
            state_configs=successful_state_paths,
            env_id=second_task_env,
            seed=args.seed,
            no_cull=args.no_cull
        )

    print(f"\n{'='*80}")
    print("PIPELINE COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
