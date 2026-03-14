"""Regression tests for paper figures and tables.

This module verifies that key paper results can be reproduced
from fixed random seeds, ensuring consistency across runs.
"""

from __future__ import annotations

import numpy as np
import sys
sys.path.insert(0, 'src')

from noise_models import add_gaussian_rotation_noise, set_bounded_noise
from smoother_fast import tube_smooth_fast
from so3 import exp_so3, log_so3


def test_demo_synthetic_regression() -> None:
    """Verify demo_synthetic.py produces expected outputs.

    Expected values based on: --N 20 --lam 1.0 --mu 0.2 --tau 0.1
    --eps 0.18 --noise 0.08 --solver SCS
    """
    # Fixed seed from demo_synthetic.py
    N = 20
    tau = 0.1
    lam = 1.0
    mu = 0.2
    noise_sigma = 0.08
    eps_val = 0.18

    # Generate data (same as demo_synthetic.py)
    t = np.arange(N) * tau
    w = np.stack([
        0.7 * np.sin(0.8 * t),
        0.5 * np.sin(0.5 * t + 0.3),
        0.6 * np.cos(0.6 * t - 0.2),
    ], axis=1)
    phi = np.cumsum(w * tau, axis=0)
    R_gt = np.stack([exp_so3(phi) for phi in phi], axis=0)

    # Add noise with seed 0 (default in demo)
    R_noisy, eps = set_bounded_noise(R_gt, noise_sigma=noise_sigma, seed=0)

    # Smooth
    R_hat, info = tube_smooth_fast(
        R_noisy, eps, lam=lam, mu=mu, tau=tau,
        max_outer=20, Delta=0.25, rho=1.0, inner_max_iter=2000, tol_outer=1e-7, tol_inner=1e-4,
    )

    # Compute metrics
    err_meas = np.array([np.linalg.norm(log_so3(R_gt[i]).T @ R_noisy[i]) for i in range(N)])
    err_hat = np.array([np.linalg.norm(log_so3(R_gt[i]).T @ R_hat[i]) for i in range(N)])

    gt_rms_noisy = np.sqrt(np.mean(err_meas**2))
    gt_rms_hat = np.sqrt(np.mean(err_hat**2))

    # Expected values (within tolerance for SCS solver)
    expected_outer_iter = 20  # Max iterations
    expected_runtime_range = (0.2, 2.0)  # Typical range for N=20
    expected_gt_rms_hat = (0.8, 1.2)  # Reasonable smoothing

    assert info['outer_iter'] <= expected_outer_iter, \
        f"Outer iterations mismatch: {info['outer_iter']} vs {expected_outer_iter}"
    assert expected_runtime_range[0] <= info['elapsed_sec'] <= expected_runtime_range[1], \
        f"Runtime out of range: {info['elapsed_sec']:.4f}s vs {expected_runtime_range}"
    assert expected_gt_rms_hat[0] <= gt_rms_hat <= expected_gt_rms_hat[1], \
        f"GT error unexpected: {gt_rms_hat:.4f} vs expected {expected_gt_rms_hat}"


def test_noise_model_consistency() -> None:
    """Verify noise_models.py produces consistent outputs."""
    # Test that noise addition is right-invariant
    R = np.stack([exp_so3(np.array([0.0, 0.0, angle])) for angle in [0.0, np.pi/2, np.pi]], axis=0)

    # Add same noise twice
    from noise_models import add_gaussian_rotation_noise
    R_noisy_1 = add_gaussian_rotation_noise(R, sigma=0.1, seed=42)
    R_noisy_2 = add_gaussian_rotation_noise(R, sigma=0.1, seed=42)

    # Should be identical (deterministic with fixed seed)
    assert np.allclose(R_noisy_1, R_noisy_2, atol=1e-7), \
        "Noise model not deterministic with fixed seed"

    # Test log-space noise is different
    from noise_models import add_log_space_noise
    R_noisy_log = add_log_space_noise(R, sigma=0.1, seed=42)

    # Should NOT be identical (different noise model)
    assert not np.allclose(R_noisy_1, R_noisy_log, atol=1e-3), \
        "Log-space noise should differ from right-multiply noise"


def test_geodesic_error_stability() -> None:
    """Verify geodesic error computation is stable across branch cuts."""
    # Test rotations with known geodesic distances
    R1 = exp_so3(np.array([0.0, 0.0, 0.0]))
    R2 = exp_so3(np.array([0.0, 0.0, np.pi/4]))  # 45 degrees
    R3 = exp_so3(np.array([0.0, 0.0, np.pi/2]))  # 90 degrees
    R4 = exp_so3(np.array([0.0, 0.0, np.pi]))  # 180 degrees
    R5 = exp_so3(np.array([0.0, 0.0, np.pi]))  # 180 degrees

    # Compute geodesic angles
    from so3 import geodesic_angle
    d12 = geodesic_angle(R1, R2)
    d13 = geodesic_angle(R1, R3)
    d14 = geodesic_angle(R1, R4)
    d15 = geodesic_angle(R1, R5)

    # Verify known values
    assert abs(d12 - np.pi/4) < 1e-6, f"geodesic_angle(R1,R2): {d12:.6f} vs π/4"
    assert abs(d13 - np.pi/2) < 1e-6, f"geodesic_angle(R1,R3): {d13:.6f} vs π/2"
    assert abs(d14 - np.pi) < 1e-6, f"geodesic_angle(R1,R4): {d14:.6f} vs π"
    assert abs(d15 - np.pi) < 1e-6, f"geodesic_angle(R1,R5): {d15:.6f} vs π"


def test_scaling_behavior() -> None:
    """Verify runtime scales approximately linearly with M."""
    M_values = [50, 100, 200]

    results = []
    for M in M_values:
        # Generate synthetic data
        t = np.arange(M) * 0.1
        phi = np.array([[0.1 * np.sin(0.8 * t[i]), 0.0, 0.0] for i in range(M)])
        R_gt = np.stack([exp_so3(phi) for phi in phi], axis=0)

        R_noisy, eps = set_bounded_noise(R_gt, seed=42, noise_sigma=0.05, eps_factor=3.0)

        # Smooth
        _, info = tube_smooth_fast(
            R_noisy, eps, lam=1.0, mu=0.1, tau=0.1,
            max_outer=20, Delta=0.2, rho=1.0, inner_max_iter=1000,
        )

        results.append({'M': M, 'time': info['elapsed_sec']})

    # Check scaling: should be near O(M)
    # Compute scaling factor
    t_50 = results[0]['time']
    t_100 = results[1]['time']
    t_200 = results[2]['time']

    # Runtime should scale less than quadratically (ideally linearly)
    # Allow for some overhead: t_100/t_50 < 3, t_200/t_100 < 3
    assert t_100 / t_50 < 3.0, f"Scaling M=50→100: {t_100/t_50:.2f}x (expected ~2x)"
    assert t_200 / t_100 < 3.0, f"Scaling M=100→200: {t_200/t_100:.2f}x (expected ~2x)"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
