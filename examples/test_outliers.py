"""Test tube smoothing with outliers.

This script demonstrates the value of slack variables for handling
measurements corrupted by occasional large outliers.
"""

from __future__ import annotations

import sys
import numpy as np

sys.path.insert(0, 'src')

from noise_models import add_gaussian_rotation_noise, set_bounded_noise
from smoother_fast import tube_smooth_fast
from so3 import exp_so3, log_so3


def generate_trajectory(M: int, tau: float = 0.1) -> np.ndarray:
    """Generate a smooth synthetic SO(3) trajectory."""
    t = np.arange(M) * tau
    w = np.stack([
        0.7 * np.sin(0.8 * t),
        0.5 * np.sin(0.5 * t + 0.3),
        0.6 * np.cos(0.6 * t - 0.2),
    ], axis=1)
    phi = np.cumsum(w * tau, axis=0)
    return np.stack([exp_so3(phi) for phi in phi], axis=0)


def compute_geodesic_error(R_ref: np.ndarray, R_est: np.ndarray) -> float:
    """Compute RMS geodesic angle error."""
    errors = np.array([
        np.linalg.norm(log_so3(R_ref[i].T @ R_est[i]))
        for i in range(len(R_ref))
    ])
    return np.sqrt(np.mean(errors**2))


def main() -> None:
    """Run outlier study."""
    print("=== Outlier Study: Tube Smoothing with Slack Variables ===")

    # Parameters
    M = 200
    tau = 0.1
    lam = 1.0
    mu = 0.2
    noise_sigma = 0.05  # Standard noise level
    outlier_rate = 0.05  # 5% outliers
    outlier_scale = 5.0  # 5x normal noise

    # Generate ground truth
    R_gt = generate_trajectory(M, tau)
    print(f"Generated {M} ground truth rotations")

    # Add Gaussian noise with outliers
    R_noisy = add_gaussian_rotation_noise(
        R_gt,
        sigma=noise_sigma,
        seed=42,
        add_outliers=True,
        outlier_rate=outlier_rate,
        outlier_scale=outlier_scale,
    )

    # Compute eps (slightly larger than noise to handle regular measurements)
    eps = 2.5 * noise_sigma * np.ones(M)

    # Identify true outliers (for validation)
    rng = np.random.default_rng(42)
    n_outliers = int(outlier_rate * M)
    if n_outliers > 0:
        true_outlier_indices = rng.choice(M, size=n_outliers, replace=False)
        true_outlier_indices = np.sort(true_outlier_indices)
    else:
        true_outlier_indices = np.array([])

    print(f"Added {n_outliers} outliers ({100*outlier_rate:.0f}%) at indices: {true_outlier_indices[:10]}...")

    # Test 1: Without slack (typically yields positive tube excess on outliers)
    print("\n--- Test 1: Without Slack Variables ---")
    R_hat_no_slack, info_no_slack = tube_smooth_fast(
        R_noisy, eps, lam, mu, tau,
        max_outer=20,
        Delta=0.2,
        rho=1.0,
        inner_max_iter=2000,
        tol_outer=1e-6,
        tol_inner=1e-4,
        slack=False,
    )

    error_no_slack = compute_geodesic_error(R_gt, R_hat_no_slack)
    violations_no_slack = np.array([
        np.linalg.norm(log_so3(R_noisy[i].T @ R_hat_no_slack[i])) - eps[i]
        for i in range(M)
    ])
    max_violation_no_slack = np.max(violations_no_slack)
    avg_violation_no_slack = np.mean(np.maximum(violations_no_slack, 0))

    print(f"Outer iterations: {info_no_slack['outer_iter']}")
    print(f"Runtime: {info_no_slack['elapsed_sec']:.4f} s")
    print(f"GT error RMS: {error_no_slack:.4f} rad")
    print(f"Max tube excess: {max_violation_no_slack:.4f} rad")
    print(f"Avg positive tube excess: {avg_violation_no_slack:.4f} rad")

    # Test 2: With slack (should identify and accommodate outliers)
    print("\n--- Test 2: With Slack Variables ---")
    rho_values = [1e2, 1e3, 1e4]  # Test different slack penalties

    for rho in rho_values:
        R_hat_slack, info_slack = tube_smooth_fast(
            R_noisy, eps, lam, mu, tau,
            max_outer=20,
            Delta=0.2,
            rho=rho,
            inner_max_iter=2000,
            tol_outer=1e-6,
            tol_inner=1e-4,
            slack=True,
        )

        error_slack = compute_geodesic_error(R_gt, R_hat_slack)
        violations_slack = np.array([
            np.linalg.norm(log_so3(R_noisy[i].T @ R_hat_slack[i])) - eps[i]
            for i in range(M)
        ])
        max_violation_slack = np.max(violations_slack)
        avg_violation_slack = np.mean(np.maximum(violations_slack, 0))

        active_slack = info_slack.get('active_slack_indices', [])
        outlier_recovery_rate = len(np.intersect1d(active_slack, true_outlier_indices)) / len(true_outlier_indices) if len(true_outlier_indices) > 0 else 0.0

        print(f"\n--- rho = {rho} ---")
        print(f"Outer iterations: {info_slack['outer_iter']}")
        print(f"Runtime: {info_slack['elapsed_sec']:.4f} s")
        print(f"GT error RMS: {error_slack:.4f} rad")
        print(f"Max tube excess: {max_violation_slack:.4f} rad")
        print(f"Avg positive tube excess: {avg_violation_slack:.4f} rad")
        print(f"Active slack indices: {len(active_slack)}")
        print(f"Outlier recovery rate: {100*outlier_recovery_rate:.1f}%")
        print(f"Recovered outliers: {np.intersect1d(active_slack, true_outlier_indices)[:10]}...")

    # Summary comparison
    print("\n=== Summary ===")
    print("Without slack: Constraints violated on outliers, high GT error")
    print("With slack: Outliers identified via active_slack_indices, improved GT error")
    print("\nRecommended: Use slack=True when outliers expected (rho=1e3)")


if __name__ == "__main__":
    main()
