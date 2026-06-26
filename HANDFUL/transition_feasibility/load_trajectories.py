import h5py
import numpy as np

def load_trajectory_from_h5(h5_path, trajectory_idx):
    """
    Load a single trajectory from HDF5 file.
    
    Args:
        h5_path: Path to HDF5 file
        trajectory_idx: Index of trajectory to load
        
    Returns:
        Dictionary with trajectory data
    """
    with h5py.File(h5_path, 'r') as f:
        traj_name = f'trajectory_{trajectory_idx:04d}'
        if traj_name not in f:
            raise ValueError(f"Trajectory {trajectory_idx} not found in {h5_path}")
        
        traj_group = f[traj_name]
        
        trajectory_data = {}

        # Load Datasets
        for key in traj_group.keys():
            trajectory_data[key] = np.array(traj_group[key])

        # Load metadata
        trajectory_data['metadata'] = {}
        for attr in traj_group.attrs:
            trajectory_data['metadata'][attr] = traj_group.attrs[attr]
        
        return trajectory_data


def print_h5_summary(h5_path):
    """Print summary of trajectories in HDF5 file."""
    with h5py.File(h5_path, 'r') as f:
        print("\n" + "="*60)
        print("HDF5 TRAJECTORY FILE SUMMARY")
        print("="*60)
        print(f"File: {h5_path}")
        print(f"Number of trajectories: {f.attrs['num_trajectories']}")
        print(f"Environment: {f.attrs['env_id']}")
        print(f"Control mode: {f.attrs['control_mode']}")
        print(f"Force threshold: {f.attrs['hand_table_force_threshold']} N")
        print(f"Problematic trajectories: {f.attrs['num_problematic']}")
        
        # Sample first trajectory to show structure
        if 'trajectory_0000' in f:
            traj = f['trajectory_0000']
            print(f"\nTrajectory structure (example from trajectory_0000):")
            for key in traj.keys():
                dataset = traj[key]
                print(f"  {key}: shape={dataset.shape}, dtype={dataset.dtype}")
        
        print("="*60 + "\n")
