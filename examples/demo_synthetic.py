"""Synthetic demo for SO(3) tube smoothing baseline."""

from __future__ import annotations

import argparse
from typing import Tuple

import numpy as np

from smoother_socp import one_outer_iteration_baseline, tube_smooth_socp
from so3 import exp_so3, log_so3


def make_ground_truth(N: int, tau: float) -> np.ndarray:
    """Generate a smooth synthetic SO(3) trajectory."""
    t = np.arange(N) * tau
    w = np.stack(
        [
            0.7 * np.sin(0.8 * t),
            0.5 * np.sin(0.5 * t + 0.3),
            0.6 * np.cos(0.6 * t - 0.2),
        ],
        axis=1,
    )
    phi = np.cumsum(w * tau, axis=0)
    return np.stack([exp_so3(phi_i) for phi_i in phi], axis=0)


def add_noisy_measurement(R_gt: np.ndarray, noise_sigma: float, seed: int = 0) -> np.ndarray:
    """Right-multiply isotropic Gaussian perturbation in tangent space."""
    rng = np.random.default_rng(seed)
    N = R_gt.shape[0]
    noise = rng.normal(0.0, noise_sigma, size=(N, 3))
    return np.stack([R_gt[i] @ exp_so3(noise[i]) for i in range(N)], axis=0)


def smoothness_metrics(phi_seq: np.ndarray, tau: float) -> Tuple[float, float]:
    """Return RMS angular velocity and acceleration proxies."""
    d1 = np.diff(phi_seq, axis=0) / tau
    d2 = np.diff(phi_seq, n=2, axis=0) / (tau**2)
    vel_rms = float(np.sqrt(np.mean(np.sum(d1**2, axis=1)))) if len(d1) else 0.0
    acc_rms = float(np.sqrt(np.mean(np.sum(d2**2, axis=1)))) if len(d2) else 0.0
    return vel_rms, acc_rms


def geodesic_errors(R_ref: np.ndarray, R_est: np.ndarray) -> np.ndarray:
    """Compute per-sample geodesic angle error."""
    return np.array([np.linalg.norm(log_so3(R_ref[i].T @ R_est[i])) for i in range(R_ref.shape[0])])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=40)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--mu", type=float, default=0.2)
    parser.add_argument("--eps", type=float, default=0.18)
    parser.add_argument("--noise", type=float, default=0.08)
    parser.add_argument("--solver", choices=["ECOS", "SCS"], default="ECOS")
    parser.add_argument("--slack", action="store_true")
    args = parser.parse_args()

    R_gt = make_ground_truth(args.N, args.tau)
    R_meas = add_noisy_measurement(R_gt, noise_sigma=args.noise)
    eps = args.eps * np.ones(args.N)

    R_one, info_one = one_outer_iteration_baseline(
        R_meas, eps, lam=args.lam, mu=args.mu, tau=args.tau, solver=args.solver
    )
    R_hat, info = tube_smooth_socp(
        R_meas,
        eps,
        lam=args.lam,
        mu=args.mu,
        tau=args.tau,
        max_outer=20,
        Delta=0.25,
        tol=1e-7,
        solver=args.solver,
        slack=args.slack,
    )

    err_meas = geodesic_errors(R_gt, R_meas)
    err_one = geodesic_errors(R_gt, R_one)
    err_hat = geodesic_errors(R_gt, R_hat)

    phi_meas = np.vstack([log_so3(R) for R in R_meas])
    phi_hat = np.vstack([log_so3(R) for R in R_hat])

    meas_vel, meas_acc = smoothness_metrics(phi_meas, args.tau)
    hat_vel, hat_acc = smoothness_metrics(phi_hat, args.tau)

    tube_res = np.array([np.linalg.norm(log_so3(R_meas[i].T @ R_hat[i])) - eps[i] for i in range(args.N)])

    print("=== Tube smoothing on SO(3): synthetic demo ===")
    print(f"N={args.N}, tau={args.tau}, solver={args.solver}, slack={args.slack}")
    print(f"Outer iterations: {info['outer_iter']}")
    print(f"Runtime: {info['elapsed_sec']:.4f} s")
    print(f"Max tube violation: {tube_res.max():.3e} rad")
    print(f"Avg positive violation: {np.maximum(tube_res, 0).mean():.3e} rad")
    print(f"GT error (meas / 1-outer / multi-outer) RMS: {np.sqrt(np.mean(err_meas**2)):.4f} / {np.sqrt(np.mean(err_one**2)):.4f} / {np.sqrt(np.mean(err_hat**2)):.4f} rad")
    print(f"Smoothness vel RMS (meas -> hat): {meas_vel:.4f} -> {hat_vel:.4f}")
    print(f"Smoothness acc RMS (meas -> hat): {meas_acc:.4f} -> {hat_acc:.4f}")

    if args.slack and "active_slack_indices" in info:
        print(f"Active slack indices: {info['active_slack_indices']}")


if __name__ == "__main__":
    main()
