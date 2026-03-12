"""Generate realistic noisy measurements from EuRoC MAV dataset.

This script loads ground truth rotations and simulates realistic IMU measurements
with appropriate noise characteristics, rather than using ground truth as measurements.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Tuple

import numpy as np

import sys
sys.path.insert(0, 'src')

from so3 import exp_so3, log_so3


def quaternion_to_rotation_matrix(q: Tuple[float, float, float, float]) -> np.ndarray:
    """Convert quaternion (qw, qx, qy, qz) to 3x3 rotation matrix."""
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])


def load_euroc_ground_truth(gt_path: str, max_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Load ground truth rotations from EuRoC dataset.

    Args:
        gt_path: Path to state_groundtruth_estimate0/data.csv
        max_samples: Maximum number of samples to load

    Returns:
        R_gt: Ground truth rotations, shape (N, 3, 3)
        timestamps: Timestamps in seconds, shape (N,)
    """
    R_gt = []
    timestamps = []

    with open(gt_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for i, row in enumerate(reader):
            if i >= max_samples:
                break
            ts = float(row[0])
            qw, qx, qy, qz = map(float, row[4:8])
            R_gt.append(quaternion_to_rotation_matrix([qw, qx, qy, qz]))
            timestamps.append(ts)

    return np.array(R_gt, dtype=np.float64), np.array(timestamps)


def simulate_imu_measurements(
    R_gt: np.ndarray,
    timestamps: np.ndarray,
    gyro_std: float = 0.01,  # rad/s (EuRoC specification)
    gyro_bias_std: float = 0.005,  # rad/s (EuRoC specification)
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate IMU measurements with realistic noise model.

    Args:
        R_gt: Ground truth rotations, shape (N, 3, 3)
        timestamps: Timestamps in seconds, shape (N,)
        gyro_std: Gyroscope noise standard deviation (rad/s)
        gyro_bias_std: Gyroscope bias stability (rad/s)
        seed: Random seed for reproducibility

    Returns:
        R_meas: Noisy rotation measurements, shape (N, 3, 3)
        eps: Tube radii derived from noise characteristics, shape (N,)

    Notes:
        - Model: R_meas[i] = R_gt[i-1] @ exp_so3(noise_i)
        - Noise_i represents gyroscope integration error over dt
        - eps = 2*sqrt(dt*gyro_std^2 + dt^2*gyro_bias_std^2) for 95% confidence
    """
    rng = np.random.default_rng(seed)
    N = len(R_gt)

    if N == 0:
        return np.array([]), np.array([])

    # Compute time steps
    dt = np.diff(timestamps, prepend=0)
    dt[0] = timestamps[1] - timestamps[0] if N > 1 else 0.01

    # Simulate gyroscope noise and bias
    # Bias drift over time: modeled as random walk
    bias_drift = np.cumsum(rng.normal(0, gyro_bias_std * dt, size=(N, 3)), axis=0)
    gyro_noise = rng.normal(0, gyro_std, size=(N, 3))

    # Total angular noise (integration of gyro error)
    angular_noise = (gyro_noise + bias_drift) * dt[:, None]

    # Apply noise to previous rotation (right-multiplication)
    # R_meas[i] = R_gt[i] @ exp_so3(noise) is NOT correct
    # We need to propagate from R_gt[i-1] with noise
    R_meas = np.zeros_like(R_gt)
    R_meas[0] = R_gt[0] @ exp_so3(angular_noise[0])  # First measurement from gt[0]
    for i in range(1, N):
        # Angular velocity from gt difference
        phi_diff = log_so3(R_gt[i-1].T @ R_gt[i])
        # Add noise to this difference
        phi_noisy = phi_diff + angular_noise[i]
        # Propagate with noisy increment
        R_meas[i] = R_gt[i-1] @ exp_so3(phi_noisy)

    # Compute tube radii from noise characteristics (95% confidence)
    # Assumes noise is Gaussian with std approx gyro_std * sqrt(dt)
    noise_std_per_sample = np.sqrt(gyro_std**2 * dt + (gyro_bias_std * dt)**2)
    eps = 2.0 * noise_std_per_sample  # 2-sigma for 95% confidence

    return R_meas, eps


def main() -> None:
    """Generate EuRoC subset with simulated IMU measurements."""
    # Configuration
    dataset_path = Path('data/machine_hall/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv')
    output_path = Path('data/euroc_mav_subset.npz')
    max_samples = 1000

    # EuRoC gyroscope specifications (from dataset documentation)
    gyro_std = 0.01  # rad/s noise standard deviation
    gyro_bias_std = 0.005  # rad/s bias stability

    # Load ground truth
    R_gt, timestamps = load_euroc_ground_truth(str(dataset_path), max_samples)
    print(f"Loaded {len(R_gt)} ground truth samples from {dataset_path}")

    # Simulate IMU measurements with realistic noise
    R_meas, eps = simulate_imu_measurements(
        R_gt,
        timestamps,
        gyro_std=gyro_std,
        gyro_bias_std=gyro_bias_std,
        seed=42,  # Fixed seed for reproducibility
    )
    print(f"Generated {len(R_meas)} noisy measurements")
    print(f"Tube radii: eps_min={eps.min():.4f}, eps_max={eps.max():.4f}, eps_mean={eps.mean():.4f} rad")

    # Save to NPZ file with all arrays
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        R_gt=R_gt,  # Ground truth for validation
        R_meas=R_meas,  # Noisy measurements
        eps=eps,  # Tube radii derived from noise model
        timestamps=timestamps,
    )
    print(f"Saved to {output_path}")

    # Summary statistics
    geodesic_errors = np.array([np.linalg.norm(log_so3(R_gt[i].T @ R_meas[i])) for i in range(len(R_gt))])
    print(f"\nMeasurement noise statistics:")
    print(f"  Mean geodesic error: {geodesic_errors.mean():.4f} rad")
    print(f"  Median geodesic error: {np.median(geodesic_errors):.4f} rad")
    print(f"  Max geodesic error: {geodesic_errors.max():.4f} rad")


if __name__ == '__main__':
    main()
