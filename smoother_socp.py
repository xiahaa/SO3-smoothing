"""Sequential convexification baseline for set-membership tube smoothing on SO(3)."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Literal, Tuple

import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from so3 import exp_so3, log_so3, right_jacobian, right_jacobian_inv

SolverName = Literal["ECOS", "SCS"]


def build_difference_matrices(M: int) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    """Build first/second-order finite difference matrices.

    D1 has shape (M-1, M), D2 has shape (M-2, M).
    """
    if M < 2:
        raise ValueError("M must be >= 2")

    D1 = sp.diags(
        diagonals=[-np.ones(M - 1), np.ones(M - 1)],
        offsets=[0, 1],
        shape=(M - 1, M),
        format="csr",
    )

    if M >= 3:
        D2 = sp.diags(
            diagonals=[np.ones(M - 2), -2.0 * np.ones(M - 2), np.ones(M - 2)],
            offsets=[0, 1, 2],
            shape=(M - 2, M),
            format="csr",
        )
    else:
        D2 = sp.csr_matrix((0, M))
    return D1, D2


def build_hessian(M: int, lam: float, mu: float, tau: float) -> sp.csc_matrix:
    """Build sparse Hessian H for stacked phi ∈ R^{3M}."""
    if tau <= 0:
        raise ValueError("tau must be positive")

    D1, D2 = build_difference_matrices(M)
    I3 = sp.eye(3, format="csc")
    H1 = (lam / tau) * sp.kron(D1.T @ D1, I3, format="csc")
    H2 = (mu / tau**3) * sp.kron(D2.T @ D2, I3, format="csc")
    H = (H1 + H2).tocsc()
    # Numerical regularization to avoid singularity from gauge freedom.
    H = H + 1e-9 * sp.eye(3 * M, format="csc")
    return H


def _stack_phi(phi_seq: np.ndarray) -> np.ndarray:
    return phi_seq.reshape(-1)


def _unstack_phi(phi_vec: np.ndarray) -> np.ndarray:
    return phi_vec.reshape(-1, 3)


def tube_smooth_socp(
    R_meas: np.ndarray,
    eps: np.ndarray,
    lam: float,
    mu: float,
    tau: float,
    max_outer: int = 20,
    Delta: float = 0.2,
    tol: float = 1e-6,
    solver: SolverName = "ECOS",
    slack: bool = False,
    rho: float = 1e3,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Smooth rotations with sequential convexification and SOCP subproblems.

    Args:
        R_meas: Measured rotations, shape (M,3,3).
        eps: Tube radius per sample in radians, shape (M,).
        lam: First-order smoothness weight.
        mu: Second-order smoothness weight.
        tau: Time step.
        max_outer: Max outer sequential-convexification iterations.
        Delta: Trust-region radius for each increment block.
        tol: Stop if ||delta||_inf < tol.
        solver: "ECOS" or "SCS".
        slack: Add nonnegative slack to tube constraints.
        rho: L1 penalty on slack.

    Returns:
        R_hat: Smoothed rotations (M,3,3).
        info: Diagnostic dictionary.
    """
    R_meas = np.asarray(R_meas, dtype=float)
    eps = np.asarray(eps, dtype=float).reshape(-1)
    M = R_meas.shape[0]
    if R_meas.shape != (M, 3, 3):
        raise ValueError("R_meas must have shape (M,3,3)")
    if eps.shape[0] != M:
        raise ValueError("eps length must match R_meas")

    phi_meas = np.vstack([log_so3(R_meas[i]) for i in range(M)])
    phi_k = phi_meas.copy()

    H = build_hessian(M, lam, mu, tau)

    objective_hist: List[float] = []
    max_violation_hist: List[float] = []
    avg_violation_hist: List[float] = []
    slack_hist: List[np.ndarray] = []

    start_t = time.perf_counter()
    last_delta_norm = np.inf

    for outer in range(max_outer):
        r_list: List[np.ndarray] = []
        J_list: List[np.ndarray] = []

        for i in range(M):
            r_i = log_so3(exp_so3(-phi_meas[i]) @ exp_so3(phi_k[i]))
            J_i = right_jacobian_inv(r_i) @ right_jacobian(phi_k[i])
            r_list.append(r_i)
            J_list.append(J_i)

        phi_k_vec = _stack_phi(phi_k)
        g = H @ phi_k_vec

        delta = cp.Variable(3 * M)
        constraints = []

        if slack:
            s = cp.Variable(M, nonneg=True)
        else:
            s = None

        for i in range(M):
            di = delta[3 * i : 3 * (i + 1)]
            affine = r_list[i] + J_list[i] @ di
            if slack:
                constraints.append(cp.norm(affine, 2) <= eps[i] + s[i])
            else:
                constraints.append(cp.norm(affine, 2) <= eps[i])
            constraints.append(cp.norm(di, 2) <= Delta)

        obj = 0.5 * cp.quad_form(delta, H) + g @ delta
        if slack:
            obj = obj + rho * cp.sum(s)
        prob = cp.Problem(cp.Minimize(obj), constraints)

        if solver == "SCS":
            prob.solve(solver=cp.SCS, verbose=False, eps=1e-5, max_iters=10_000)
        else:
            prob.solve(solver=cp.ECOS, verbose=False, abstol=1e-8, reltol=1e-8, feastol=1e-8)

        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            raise RuntimeError(f"SOCP subproblem failed at outer={outer}, status={prob.status}")

        delta_val = np.asarray(delta.value).reshape(-1)
        phi_k = phi_k + _unstack_phi(delta_val)

        if slack:
            slack_hist.append(np.asarray(s.value).reshape(-1))

        phi_vec = _stack_phi(phi_k)
        obj_val = float(0.5 * phi_vec @ (H @ phi_vec))
        objective_hist.append(obj_val)

        violations = []
        for i in range(M):
            v = np.linalg.norm(log_so3(R_meas[i].T @ exp_so3(phi_k[i]))) - eps[i]
            violations.append(v)
        violations = np.array(violations)
        max_violation_hist.append(float(np.max(violations)))
        avg_violation_hist.append(float(np.mean(np.maximum(violations, 0.0))))

        last_delta_norm = float(np.max(np.abs(delta_val)))
        if last_delta_norm < tol:
            break

    elapsed = time.perf_counter() - start_t

    R_hat = np.stack([exp_so3(phi_k[i]) for i in range(M)], axis=0)

    info: Dict[str, Any] = {
        "outer_iter": len(objective_hist),
        "objective": objective_hist,
        "max_violation": max_violation_hist,
        "avg_violation": avg_violation_hist,
        "elapsed_sec": elapsed,
        "final_delta_inf": last_delta_norm,
    }
    if slack and slack_hist:
        final_slack = slack_hist[-1]
        info["final_slack"] = final_slack
        info["active_slack_indices"] = np.where(final_slack > 1e-9)[0].tolist()

    return R_hat, info


def one_outer_iteration_baseline(
    R_meas: np.ndarray,
    eps: np.ndarray,
    lam: float,
    mu: float,
    tau: float,
    Delta: float = 0.2,
    solver: SolverName = "ECOS",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Convenience baseline equivalent to outer loop max_outer=1."""
    return tube_smooth_socp(
        R_meas=R_meas,
        eps=eps,
        lam=lam,
        mu=mu,
        tau=tau,
        max_outer=1,
        Delta=Delta,
        solver=solver,
    )
