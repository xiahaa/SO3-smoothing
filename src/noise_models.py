"""Unified noise models for SO(3) tube smoothing experiments.

This module provides consistent noise generation across all scripts, ensuring
reproducibility and clear semantics for different experimental scenarios.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional

from so3 import exp_so3, batch_log


def add_gaussian_rotation_noise(
    R_seq: np.ndarray,
    sigma: float,
    seed: int = 0,
    add_outliers: bool = False,
    outlier_rate: float = 0.05,
    outlier_scale: float = 5.0,
) -> np.ndarray:
    """Right-multiply isotropic Gaussian perturbation in tangent space.

    Args:
        R_seq: Original rotations, shape (N, 3, 3).
        sigma: Standard deviation of noise (radians).
        seed: Random seed for reproducibility.
        add_outliers: If True, add outliers to subset of samples.
        outlier_rate: Fraction of samples to corrupt with outliers (0.05).
        outlier_scale: Scale factor for outliers (5x sigma).

    Returns:
        Noisy rotations R_meas, shape (N, 3, 3).

    Notes:
        - Uses right-multiplication: R_meas[i] = R_seq[i] @ exp_so3(noise)
        - Equivalent to: phi_meas = log(R_seq) + noise
        - Preserves trajectory structure while adding measurement noise.
        - Optionally adds outliers to subset (controlled corruption study).
    """
    rng = np.random.default_rng(seed)
    N = R_seq.shape[0]

    R_noisy = R_seq.copy()

    # Start with clean measurements
    if add_outliers:
        n_outliers = int(outlier_rate * N)
        if n_outliers > 0:
            outlier_indices = rng.choice(N, size=n_outliers, replace=False)
            outlier_noise = rng.normal(scale=outlier_scale * sigma, size=(n_outliers, 3))
            # Apply large outliers by right-multiplication
            R_noisy[outlier_indices] = R_noisy[outlier_indices] @ exp_so3(outlier_noise)

    # Add isotropic Gaussian noise to all samples
    noise = rng.normal(scale=sigma, size=(N, 3))
    for i in range(N):
        R_noisy[i] = R_noisy[i] @ exp_so3(noise[i])

    return R_noisy


def add_log_space_noise(
    R_seq: np.ndarray,
    sigma: float,
    seed: int = 0,
) -> np.ndarray:
    """Add isotropic Gaussian noise in Lie algebra (tangent space).

    Args:
        R_seq: Original rotations, shape (N, 3, 3).
        sigma: Standard deviation of noise (radians).
        seed: Random seed for reproducibility.

    Returns:
        Noisy rotations R_meas, shape (N, 3, 3).

    Notes:
        - Uses log-addition: phi_meas = log(R_seq) + noise
        - Equivalent to: R_meas[i] = exp_so3(phi_meas[i]) @ R_seq[i]
        - Breaks right-invariance but is mathematically simple.
    """
    rng = np.random.default_rng(seed)
    N = R_seq.shape[0]

    # Add noise in tangent space
    noise = rng.normal(scale=sigma, size=(N, 3))
    phi_meas = batch_log(R_seq) + noise

    # Convert back to rotations: R_meas[i] = exp_so3(phi_meas[i]) @ R_seq[i]
    R_noisy = np.array([exp_so3(phi_meas[i]) @ R_seq[i] for i in range(N)])

    return R_noisy


def set_bounded_noise(
    R_seq: np.ndarray,
    eps: Optional[np.ndarray] = None,
    seed: int = 0,
    noise_sigma: float = 0.08,
    eps_factor: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Set tube radii for bounded-noise experiments.

    Args:
        R_seq: Original rotations, shape (N, 3, 3).
        eps: Tube radii per sample in radians, shape (N,).
            If None, computed as eps_factor * noise_sigma.
        seed: Random seed for reproducibility.
        noise_sigma: Standard deviation of noise to add.
        eps_factor: Factor to multiply noise_sigma by to compute eps if not provided.
            Default 2.0 corresponds to 95% confidence (2-sigma).

    Returns:
        Noisy rotations R_meas with added controlled noise.
        eps: Array of eps values for validation.

    Notes:
        - Creates synthetic bounded-noise study where tube radii are
          derived from known noise characteristics.
        - Adds controlled noise using right-multiplication.
        - Returns eps array for downstream constraint validation.
    """
    rng = np.random.default_rng(seed)
    N = R_seq.shape[0]

    # Compute eps if not provided
    if eps is None:
        eps = eps_factor * noise_sigma * np.ones(N)
    else:
        eps = np.asarray(eps, dtype=float).reshape(-1)

    # Add controlled noise (right-multiplication in tangent space)
    controlled_noise = rng.normal(scale=noise_sigma, size=(N, 3))
    R_noisy = np.array([R_seq[i] @ exp_so3(controlled_noise[i]) for i in range(N)], dtype=np.float64)

    return R_noisy, eps


def generate_controlled_synthetic(
    M: int,
    tau: float = 0.1,
    noise_sigma: float = 0.05,
    seed: int = 0,
    noise_type: str = "gaussian_rotation",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic trajectory for controlled experiments.

    Args:
        M: Number of samples.
        tau: Time step.
        noise_sigma: Standard deviation of noise.
        seed: Random seed.
        noise_type: Type of noise ("gaussian_rotation" or "log_space").

    Returns:
        R_true: Ground truth rotations.
        R_meas: Noisy measurements.
        eps: Tube radii (derived from noise_sigma).

    Notes:
        - Creates realistic synthetic data with known noise characteristics.
        - Returns eps as factor * noise_sigma for validation.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(M) * tau

    # Generate smooth trajectory
    w = np.stack([
        0.7 * np.sin(0.8 * t),
        0.5 * np.sin(0.5 * t + 0.3),
        0.6 * np.cos(0.6 * t - 0.2),
    ], axis=1)

    phi_true = np.cumsum(w * tau, axis=0)
    R_true = np.array([exp_so3(phi) for phi in phi_true], axis=0)

    # Add noise
    if noise_type == "gaussian_rotation":
        noise = rng.normal(scale=noise_sigma, size=(M, 3))
        R_meas = [R_true[i] @ exp_so3(noise[i]) for i in range(M)]
        eps = 2.0 * noise_sigma * np.ones(M)  # eps = 2*sigma for 95% constraint satisfaction
    elif noise_type == "log_space":
        noise = rng.normal(scale=noise_sigma, size=(M, 3))
        phi_meas = batch_log(R_true) + noise
        R_meas = [exp_so3(phi_meas[i]) @ R_true[i] for i in range(M)]
        eps = 2.0 * noise_sigma * np.ones(M)
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")

    return R_true, R_meas, eps
