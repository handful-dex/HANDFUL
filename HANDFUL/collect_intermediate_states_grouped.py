import subprocess
import os
from pathlib import Path

COLLECTION_CONFIGS = [
    # (checkpoint_path, fingers, num_active_fingers, palm_use)
    ("runs/fingers_0_1_2_3_active_1_palm_True_seed_15/ckpt_5601280.pt", [0,1,2,3], 1, True),
    ("runs/fingers_0_1_2_3_active_2_palm_True_seed_15/ckpt_5601280.pt", [0,1,2,3], 2, True),
    ("runs/fingers_0_2_1_3_active_2_palm_False_seed_15/ckpt_5601280.pt", [0,2,1,3], 2, False),
    ("runs/fingers_1_0_2_3_active_1_palm_True_seed_15/ckpt_5601280.pt", [1,0,2,3], 1, True),
    ("runs/fingers_1_2_0_3_active_2_palm_True_seed_15/ckpt_5500928.pt", [1,2,0,3], 2, True),
    ("runs/fingers_2_0_1_3_active_1_palm_True_seed_15/ckpt_5601280.pt", [2,0,1,3], 1, True),
    ("runs/fingers_3_0_1_2_active_2_palm_False_seed_15/ckpt_5601280.pt", [3,0,1,2], 2, False),
    ("runs/fingers_3_1_0_2_active_2_palm_False_seed_15/ckpt_5601280.pt", [3,1,0,2], 2, False),
    ("runs/fingers_3_2_0_1_active_2_palm_False_seed_15/ckpt_5500928.pt", [3,2,0,1], 2, False),
]

# Collection settings
ENV_ID = "xArm7-v1-pick-randomized"
NUM_STATES_TO_SAVE = 16384
NUM_STEPS_TO_SAVE = 10
SAVE_VIDEOS = True


def collect_states(checkpoint, fingers, num_active_fingers, palm_use, env_id, states_to_save, steps_to_save, save_videos,
                   min_episodes_for_cutoff=5000, success_rate_cutoff=15.0):
    """Collect intermediate states from trained model."""
    
    if not Path(checkpoint).exists():
        print(f"ERROR: Checkpoint not found at {checkpoint}")
        return False
    
    cmd = [
        "python", "transition_feasibility/collect_intermediate_states.py",
        "--checkpoint", checkpoint,
        "--env_id", env_id,
        "--finger_selection", ','.join(map(str, fingers)),
        "--num_active_fingers", str(num_active_fingers),
        "--num_states_to_save", str(states_to_save),
        "--num_steps_to_save", str(steps_to_save),
        "--min-episodes-for-cutoff", str(min_episodes_for_cutoff),
        "--success-rate-cutoff", str(success_rate_cutoff),
    ]
    
    if palm_use:
        cmd.append("--palm_use")
    else:
        cmd.append("--no-palm-use")
    
    if save_videos:
        cmd.append("--save_videos")
    else:
        cmd.append("--no-save_videos")
    
    print(f"Collecting from: {Path(checkpoint).parent.name}")
    
    result = subprocess.run(cmd)
    
    # SAPIEN/Vulkan drivers often segfault at Python interpreter exit (exit code 139 / signal 11).
    # Since this happens after all files have been saved, we check if the output file exists
    # to determine success, rather than relying solely on the exit code.
    run_name = Path(checkpoint).parent.name
    output_file = Path(f"intermediate_states/{run_name}/grasp_to_push.pt")
    return output_file.exists()


def run_collection_pipeline(collection_configs, env_id, states_to_save, steps_to_save, save_videos,
                            min_episodes_for_cutoff=5000, success_rate_cutoff=15.0):
    results = []
    
    for i, (checkpoint, fingers, num_active_fingers, palm_use) in enumerate(collection_configs, 1):
        print(f"\nCollection {i}/{len(collection_configs)}: fingers={fingers}, active={num_active_fingers}, palm={palm_use}")
        
        success = collect_states(
            checkpoint, fingers, num_active_fingers, palm_use, env_id, states_to_save, steps_to_save, save_videos,
            min_episodes_for_cutoff=min_episodes_for_cutoff, success_rate_cutoff=success_rate_cutoff
        )
        
        run_name = Path(checkpoint).parent.name
        output_dir = f"intermediate_states/{run_name}"
        
        results.append({
            'checkpoint': checkpoint,
            'fingers': fingers,
            'num_active': num_active_fingers,
            'palm': palm_use,
            'success': success,
            'output_dir': output_dir
        })
        
    return results


def main():
    results = run_collection_pipeline(COLLECTION_CONFIGS, ENV_ID, NUM_STATES_TO_SAVE, NUM_STEPS_TO_SAVE, SAVE_VIDEOS)
    
    # Print summary
    print(f"\n{'='*80}")
    print("STATE COLLECTION COMPLETE")
    print('='*80)
    print(f"{'Fingers':<20} {'Active':<8} {'Palm':<8} {'Status':<10}")
    print('-'*80)
    for r in results:
        fingers_str = '_'.join(map(str, r['fingers']))
        status = 'success' if r['success'] else 'failed'
        print(f"{fingers_str:<20} {r['num_active']:<8} {str(r['palm']):<8} {status:<10}")
    print('='*80)


if __name__ == "__main__":
    main()