"""Benchmark for high-performance SO(3) tube smoothing with ADMM inner solver."""

from __future__ import annotations

import argparse
import time
from typing import Dict, Tuple

import numpy as np

from smoother_fast import solve_inner_with_cvxpy_reference, tube_smooth_fast
from hessian import build_H
from so3 import exp_so3, log_so3


def make_data(M: int, tau: float, noise: float, eps_val: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = np.arange(M) * tau
    omega = np.stack(
        [0.6 * np.sin(0.13 * t), 0.4 * np.cos(0.09 * t + 0.2), 0.5 * np.sin(0.07 * t - 0.4)], axis=1
    )
    phi_gt = np.cumsum(omega * tau, axis=0)
    R_gt = np.stack([exp_so3(phi_gt[i]) for i in range(M)], axis=0)

    noise_vec = rng.normal(0.0, noise, size=(M, 3))
    R_meas = np.stack([R_gt[i] @ exp_so3(noise_vec[i]) for i in range(M)], axis=0)
    eps = np.full(M, eps_val, dtype=np.float64)
    return R_gt, R_meas, eps


def smoothness_metrics(phi_seq: np.ndarray, tau: float) -> Tuple[float, float]:
    d1 = np.diff(phi_seq, axis=0) / tau
    d2 = np.diff(phi_seq, n=2, axis=0) / (tau**2)
    vel = float(np.sqrt(np.mean(np.sum(d1**2, axis=1)))) if len(d1) else 0.0
    acc = float(np.sqrt(np.mean(np.sum(d2**2, axis=1)))) if len(d2) else 0.0
    return vel, acc


def run_case(M: int, args) -> Dict[str, float]:
    R_gt, R_meas, eps = make_data(M, args.tau, args.noise, args.eps)
    t0 = time.perf_counter()
    R_hat, info = tube_smooth_fast(
        R_meas,
        eps,
        lam=args.lam,
        mu=args.mu,
        tau=args.tau,
        max_outer=args.max_outer,
        Delta=args.delta,
        rho=args.rho,
        inner_max_iter=args.inner_max_iter,
        inner_tol=args.inner_tol,
        tol=args.outer_tol,
    )
    total = time.perf_counter() - t0

    max_violation = float(np.max([np.linalg.norm(log_so3(R_meas[i].T @ R_hat[i])) - eps[i] for i in range(M)]))

    phi_meas = np.vstack([log_so3(R_meas[i]) for i in range(M)])
    phi_hat = np.vstack([log_so3(R_hat[i]) for i in range(M)])
    vel_m, acc_m = smoothness_metrics(phi_meas, args.tau)
    vel_h, acc_h = smoothness_metrics(phi_hat, args.tau)

    inner_iters = [it["iter"] for it in info["inner_stats"]]

    print(f"M={M}")
    print(f"  outer_iter={info['outer_iter']}, total={total:.3f}s, max_violation={max_violation:.3e}")
    print(f"  outer time per iter (s): {[round(x, 4) for x in info['outer_elapsed']]}")
    print(f"  inner avg iter={np.mean(inner_iters):.1f}, inner iters={inner_iters}")
    print(f"  smooth vel RMS meas->hat: {vel_m:.4f}->{vel_h:.4f}")
    print(f"  smooth acc RMS meas->hat: {acc_m:.4f}->{acc_h:.4f}")

    return {
        "M": M,
        "outer_iter": info["outer_iter"],
        "time_sec": total,
        "max_violation": max_violation,
        "inner_avg_iter": float(np.mean(inner_iters)),
    }


def cvxpy_reference_check(args) -> None:
    M = min(args.cvxpy_M, 200)
    _, R_meas, eps = make_data(M, args.tau, args.noise, args.eps, seed=7)
    phi_k = np.vstack([log_so3(R) for R in R_meas])
    phi_meas = phi_k.copy()

    r_list = [log_so3(exp_so3(-phi_meas[j]) @ exp_so3(phi_k[j])) for j in range(M)]
    from so3 import right_jacobian, right_jacobian_inv

    J_list = [right_jacobian_inv(r_list[j]) @ right_jacobian(phi_k[j]) for j in range(M)]

    H = build_H(M, args.lam, args.mu, args.tau)
    g = H @ phi_k.reshape(-1)

    d_admm, st = __import__("admm_solver").solve_inner_admm(
        H, g, r_list, J_list, eps, args.delta, rho=args.rho, max_iter=args.inner_max_iter, tol=args.inner_tol
    )
    d_ref, ref = solve_inner_with_cvxpy_reference(H, g, r_list, J_list, eps, args.delta, solver="SCS")

    obj_admm = 0.5 * d_admm @ (H @ d_admm) + g @ d_admm
    obj_ref = 0.5 * d_ref @ (H @ d_ref) + g @ d_ref

    def max_viol(d):
        db = d.reshape(M, 3)
        v1 = [np.linalg.norm(r_list[j] + J_list[j] @ db[j]) - eps[j] for j in range(M)]
        v2 = [np.linalg.norm(db[j]) - args.delta for j in range(M)]
        return max(max(v1), max(v2))

    print("CVXPY reference check (single outer inner-subproblem):")
    print(f"  M={M}, ADMM iter={st['iter']}, CVXPY status={ref['status']}")
    print(f"  objective admm/ref: {obj_admm:.6e} / {obj_ref:.6e}")
    print(f"  relative objective gap: {abs(obj_admm - obj_ref) / max(1.0, abs(obj_ref)):.3e}")
    print(f"  max violation admm/ref: {max_viol(d_admm):.3e} / {max_viol(d_ref):.3e}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000, 50000])
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--mu", type=float, default=0.2)
    parser.add_argument("--eps", type=float, default=0.18)
    parser.add_argument("--noise", type=float, default=0.06)
    parser.add_argument("--delta", type=float, default=0.25)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--max_outer", type=int, default=5)
    parser.add_argument("--inner_max_iter", type=int, default=800)
    parser.add_argument("--inner_tol", type=float, default=1e-4)
    parser.add_argument("--outer_tol", type=float, default=1e-6)
    parser.add_argument("--run_cvxpy_ref", action="store_true")
    parser.add_argument("--cvxpy_M", type=int, default=200)
    args = parser.parse_args()

    print("=== Fast SO(3) tube smoothing benchmark (ADMM inner) ===")
    for M in args.sizes:
        run_case(M, args)

    if args.run_cvxpy_ref:
        cvxpy_reference_check(args)


if __name__ == "__main__":
    main()
