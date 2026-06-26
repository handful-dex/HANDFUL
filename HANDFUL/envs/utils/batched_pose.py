import numpy as np
import torch
from mani_skill.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
)


def random_quaternions_batched(
    batched_rng,
    device: torch.device,
    lock_x: bool = False,
    lock_y: bool = False,
    lock_z: bool = False,
    bounds=(0, np.pi * 2),
):
    """
    Generates random quaternions using a BatchedRNG for deterministic, per-environment randomization.
    Adapted from maniskill.envs.utils.randomization.pose.py

    Args:
        batched_rng: BatchedRNG object (e.g., self._batched_episode_rng)
        device: torch device to place tensors on
        lock_x: If True, lock X rotation to 0
        lock_y: If True, lock Y rotation to 0
        lock_z: If True, lock Z rotation to 0
        bounds: Tuple of (min, max) angles in radians
        
    Returns:
        Quaternions of shape (n, 4) in [w, x, y, z] format
    """
    dist = bounds[1] - bounds[0]
    xyz_angles = batched_rng.uniform(bounds[0], bounds[1], size=(3,))
    xyz_angles = torch.from_numpy(xyz_angles).float().to(device)
    
    if lock_x:
        xyz_angles[:, 0] *= 0
    if lock_y:
        xyz_angles[:, 1] *= 0
    if lock_z:
        xyz_angles[:, 2] *= 0
    
    return matrix_to_quaternion(euler_angles_to_matrix(xyz_angles, convention="XYZ"))