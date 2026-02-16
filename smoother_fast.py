"""High-performance sequential convexification on SO(3) with custom ADMM inner solver."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Literal, Tuple

import numpy as np

from admm_solver import solve_inner_admm
from hessian import build_H
from so3 import exp_so3, log_so3, right_jacobian, right_jacobian_inv

InnerName = Literal["admm"]


def _stack_phi(phi_seq: np.ndarray) -> np.ndarray:
    return phi_seq.reshape(-1)


def _unstack_phi(phi_vec: np.ndarray) -> np.ndarray:
    return phi_vec.reshape(-1, 3)


def solve_inner_with_cvxpy_reference(
    H,
    g: np.ndarray,
    r_list: List[np.ndarray],
    J_list: List[np.ndarray],
    eps: np.ndarray,
    Delta: float,
    solver: str = "SCS",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Solve one inner problem with CVXPY for small-scale reference validation."""
    import cvxpy as cp

    M = len(r_list)
    delta = cp.Variable(3 * M)
    constraints = []
    for j in range(M):
        dj = delta[3 * j : 3 * (j + 1)]
        constraints.append(cp.norm(r_list[j] + J_list[j] @ dj, 2) <= eps[j])
        constraints.append(cp.norm(dj, 2) <= Delta)

    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(delta, H) + g @ delta), constraints)
    if solver.upper() == "ECOS":
        prob.solve(solver=cp.ECOS, verbose=False)
    else:
        prob.solve(solver=cp.SCS, verbose=False, eps=1e-5, max_iters=20_000)

    if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"CVXPY reference failed: {prob.status}")

    return np.asarray(delta.value).reshape(-1), {"status": prob.status, "obj": float(prob.value)}


def tube_smooth_fast(
    R_meas: np.ndarray,
    eps: np.ndarray,
    lam: float,
    mu: float,
    tau: float,
    max_outer: int = 20,
    Delta: float = 0.2,
    inner: InnerName = "admm",
    rho: float = 1.0,
    inner_max_iter: int = 2000,
    inner_tol: float = 1e-4,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Tube smoothing with sequential convexification and custom ADMM inner iterations."""
    if inner != "admm":
        raise ValueError("Only inner='admm' is supported in smoother_fast")

    R_meas = np.asarray(R_meas, dtype=np.float64)
    eps = np.asarray(eps, dtype=np.float64).reshape(-1)
    M = R_meas.shape[0]
    if R_meas.shape != (M, 3, 3):
        raise ValueError("R_meas must have shape (M,3,3)")
    if eps.shape[0] != M:
        raise ValueError("eps length must match R_meas")

    H = build_H(M, lam, mu, tau)
    phi_meas = np.vstack([log_so3(R_meas[i]) for i in range(M)])
    phi_k = phi_meas.copy()

    objective_hist: List[float] = []
    max_violation_hist: List[float] = []
    avg_violation_hist: List[float] = []
    inner_stats_hist: List[Dict[str, Any]] = []
    outer_elapsed_hist: List[float] = []

    total_start = time.perf_counter()
    final_delta_inf = np.inf

    for _ in range(max_outer):
        t_outer = time.perf_counter()

        r_list = []
        J_list = []
        for j in range(M):
            r_j = log_so3(exp_so3(-phi_meas[j]) @ exp_so3(phi_k[j]))
            J_j = right_jacobian_inv(r_j) @ right_jacobian(phi_k[j])
            r_list.append(r_j)
            J_list.append(J_j)

        g = H @ _stack_phi(phi_k)

        delta, inner_stats = solve_inner_admm(
            H=H,
            g=g,
            r_list=r_list,
            J_list=J_list,
            eps=eps,
            Delta=Delta,
            rho=rho,
            max_iter=inner_max_iter,
            tol=inner_tol,
        )
        inner_stats_hist.append(inner_stats)

        phi_k = phi_k + _unstack_phi(delta)

        phi_vec = _stack_phi(phi_k)
        objective_hist.append(float(0.5 * phi_vec @ (H @ phi_vec)))

        violations = np.array(
            [np.linalg.norm(log_so3(R_meas[i].T @ exp_so3(phi_k[i]))) - eps[i] for i in range(M)]
        )
        max_violation_hist.append(float(np.max(violations)))
        avg_violation_hist.append(float(np.mean(np.maximum(violations, 0.0))))

        final_delta_inf = float(np.max(np.abs(delta)))
        outer_elapsed_hist.append(time.perf_counter() - t_outer)
        if final_delta_inf < tol:
            break

    elapsed = time.perf_counter() - total_start
    R_hat = np.stack([exp_so3(phi_k[i]) for i in range(M)], axis=0)

    info: Dict[str, Any] = {
        "outer_iter": len(objective_hist),
        "objective": objective_hist,
        "max_violation": max_violation_hist,
        "avg_violation": avg_violation_hist,
        "inner_stats": inner_stats_hist,
        "outer_elapsed": outer_elapsed_hist,
        "elapsed_sec": elapsed,
        "final_delta_inf": final_delta_inf,
    }
    return R_hat, info
