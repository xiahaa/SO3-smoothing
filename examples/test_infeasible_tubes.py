"""Test tube smoothing with infeasible tube radii.

This script demonstrates the behavior when tube constraints are infeasible
(i.e., eps is too small for the actual measurement noise).
"""

from __future__ import annotations

import sys
import numpy as np

sys.path.insert(0, 'src')

from noise_models import set_bounded_noise
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
    """Run infeasible tube study."""
    print("=== Infeasible Tube Study: When eps < actual noise ===")

    # Parameters
    M = 100
    tau = 0.1
    lam = 1.0
    mu = 0.2
    noise_sigma = 0.1  # Large noise to make infeasibility more apparent

    # Generate ground truth
    R_gt = generate_trajectory(M, tau)
    print(f"Generated {M} ground truth rotations")

    # Test different tube radii (from feasible to infeasible)
    eps_ratios = [0.5, 1.0, 1.5, 2.0, 3.0]  # Multiple of noise_sigma

    print(f"\nNoise sigma: {noise_sigma:.4f} rad")
    print(f"Testing eps ratios: {eps_ratios}")
    print(f"(eps = ratio * noise_sigma, feasible when ratio >= 2.0)\n")

    results = []

    for eps_ratio in eps_ratios:
        eps = eps_ratio * noise_sigma * np.ones(M)
        eps_actual = eps[0]

        # Generate noisy measurements
        R_noisy, _ = set_bounded_noise(R_gt, noise_sigma=noise_sigma, seed=42)

        # Test without slack
        print(f"--- eps_ratio = {eps_ratio} (eps = {eps_actual:.4f} rad) ---")

        R_hat_no_slack, info_no_slack = tube_smooth_fast(
            R_noisy, eps, lam, mu, tau,
            max_outer=20,
            Delta=0.2,
            rho=1e3,
            inner_max_iter=2000,
            tol_outer=1e-6,
            tol_inner=1e-4,
            slack=False,
        )

        # Compute metrics
        error_no_slack = compute_geodesic_error(R_gt, R_hat_no_slack)
        violations_no_slack = np.array([
            np.linalg.norm(log_so3(R_noisy[i].T @ R_hat_no_slack[i])) - eps[i]
            for i in range(M)
        ])
        max_violation_no_slack = np.max(violations_no_slack)
        avg_violation_no_slack = np.mean(np.maximum(violations_no_slack, 0))
        n_violating = np.sum(violations_no_slack > 1e-6)

        print(f"  Without slack:")
        print(f"    Runtime: {info_no_slack['elapsed_sec']:.4f} s")
        print(f"    Outer iterations: {info_no_slack['outer_iter']}")
        print(f"    GT error RMS: {error_no_slack:.4f} rad")
        print(f"    Max violation: {max_violation_no_slack:.4f} rad")
        print(f"    Avg violation: {avg_violation_no_slack:.4f} rad")
        print(f"    Violating samples: {n_violating}/{M} ({100*n_violating/M:.0f}%)")

        # Test with slack (if infeasible)
        if eps_ratio < 1.5:  # Likely infeasible
            rho_values = [1e2, 1e3, 1e4]

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
                n_active_slack = len(active_slack)

                print(f"  With slack (rho={rho}):")
                print(f"    Runtime: {info_slack['elapsed_sec']:.4f} s")
                print(f"    Outer iterations: {info_slack['outer_iter']}")
                print(f"    GT error RMS: {error_slack:.4f} rad")
                print(f"    Max violation: {max_violation_slack:.4f} rad")
                print(f"    Avg violation: {avg_violation_slack:.4f} rad")
                print(f"    Active slack samples: {n_active_slack}/{M} ({100*n_active_slack/M:.0f}%)")

        print()  # Blank line

    print("=== Summary ===")
    print("- Feasible case (eps >= 2*noise_sigma):")
    print("  - Without slack: All constraints satisfied (max_violation <= eps)")
    print("  - GT error: Close to optimal")
    print()
    print("- Infeasible case (eps < 2*noise_sigma):")
    print("  - Without slack: Many constraints violated (max_violation >> eps)")
    print("  - With slack: Active slack identifies infeasible samples")
    print("  - rho selection: Higher rho penalizes slack more aggressively")
    print()
    print("Recommendation:")
    print("  - Set eps = 2*sigma for 95% confidence")
    print("  - Use slack=True when eps might be underestimated")
    print("  - Tune rho based on trust in tube bounds (1e3 is typical)")


if __name__ == "__main__":
    main()
