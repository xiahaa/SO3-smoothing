"""Unconstrained SO(3) smoothing baseline.

This module provides a simple unconstrained smoother without tube constraints,
serving as a baseline to demonstrate the value of bounded-noise formulation.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from hessian import build_H
from so3 import exp_so3, log_so3


def tube_smooth_unconstrained(
    R_meas: np.ndarray,
    lam: float,
    mu: float,
    tau: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Smooth rotations WITHOUT tube constraints (unconstrained baseline).

    This solves:
        min_phi 0.5 * phi^T H phi + phi^T g

    where g = H @ phi_meas and phi_meas = log(R_meas).

    No tube constraints: ||log(R_meas[i].T @ exp(phi_i))|| <= eps_i

    Args:
        R_meas: Measured rotations, shape (M, 3, 3).
        lam: First-order smoothness weight.
        mu: Second-order smoothness weight.
        tau: Time step.

    Returns:
        R_hat: Smoothed rotations (M, 3, 3).
        info: Diagnostic dictionary.

    Notes:
        - Provides unconstrained baseline for comparison with tube-constrained methods
        - Direct linear solve: phi* = -H \\ (H @ phi_meas)
        - Equivalent to regularized least squares without constraints
        - Significantly faster than constrained methods
    """
    R_meas = np.asarray(R_meas, dtype=np.float64)
    M = R_meas.shape[0]
    if R_meas.shape != (M, 3, 3):
        raise ValueError("R_meas must have shape (M,3,3)")

    start_t = time.perf_counter()

    # Build Hessian
    H = build_H(M, lam, mu, tau, damping=1e-9)

    # Convert to log space (tangent space)
    phi_meas = np.vstack([log_so3(R_meas[i]) for i in range(M)])

    # Stack as vector: phi ∈ R^{3M}
    phi_meas_vec = phi_meas.reshape(-1)

    # Build gradient: g = H @ phi_meas
    g = H @ phi_meas_vec

    # Solve unconstrained QP: phi* = -H \\ g
    # Equivalent to: phi* = argmin 0.5 phi^T H phi + phi^T g
    phi_hat_vec = -spla.spsolve(H, g)

    # Unstack back to (M, 3)
    phi_hat = phi_hat_vec.reshape(-1, 3)

    # Convert back to rotations
    R_hat = np.stack([exp_so3(phi) for phi in phi_hat], axis=0)

    elapsed = time.perf_counter() - start_t

    # Compute metrics for comparison
    phi_hat_vec_stack = phi_hat.reshape(-1)
    obj_val = float(0.5 * phi_hat_vec_stack @ (H @ phi_hat_vec_stack))

    info: Dict[str, Any] = {
        "elapsed_sec": elapsed,
        "objective": obj_val,
        "outer_iter": 1,  # Single solve (no outer iterations)
        "method": "unconstrained",
    }

    return R_hat, info


if __name__ == "__main__":
    # Quick demo of unconstrained smoothing
    print("=== Unconstrained Smoother Demo ===")

    # Generate synthetic data
    import sys
    sys.path.insert(0, 'src')
    from noise_models import add_gaussian_rotation_noise

    N = 100
    tau = 0.1
    lam = 1.0
    mu = 0.2

    # Generate ground truth
    t = np.arange(N) * tau
    w = np.stack([
        0.7 * np.sin(0.8 * t),
        0.5 * np.sin(0.5 * t + 0.3),
        0.6 * np.cos(0.6 * t - 0.2),
    ], axis=1)
    phi_true = np.cumsum(w * tau, axis=0)
    R_gt = np.stack([exp_so3(phi) for phi in phi_true], axis=0)

    # Add noise
    R_meas = add_gaussian_rotation_noise(R_gt, sigma=0.08, seed=42)

    # Smooth unconstrained
    R_hat, info = tube_smooth_unconstrained(R_meas, lam, mu, tau)

    # Compute errors
    geodesic_errors = np.array([
        np.linalg.norm(log_so3(R_gt[i].T @ R_hat[i]))
        for i in range(N)
    ])
    noise_errors = np.array([
        np.linalg.norm(log_so3(R_meas[i].T @ R_hat[i]))
        for i in range(N)
    ])

    print(f"Runtime: {info['elapsed_sec']:.4f} s")
    print(f"GT error RMS: {np.sqrt(np.mean(geodesic_errors**2)):.4f} rad")
    print(f"Noise error RMS: {np.sqrt(np.mean(noise_errors**2)):.4f} rad")
    print(f"Noise reduction: {100*(1 - np.sqrt(np.mean(noise_errors**2))/0.08):.1f}%")
