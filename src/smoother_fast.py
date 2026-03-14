"""High-performance sequential convexification on SO(3) with custom ADMM inner solver."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Literal, Tuple

import numpy as np

from admm_solver import solve_inner_admm
from hessian import build_H
from so3 import exp_so3, log_so3, right_jacobian, right_jacobian_inv, retract

InnerName = Literal["admm"]


def _stack_phi(phi_seq: np.ndarray) -> np.ndarray:
    return phi_seq.reshape(-1)


def _unstack_phi(phi_vec: np.ndarray) -> np.ndarray:
    return phi_vec.reshape(-1, 3)


def _initialize_phi_warmstart(
    phi_meas: np.ndarray,
    prev_delta: Optional[np.ndarray] = None,
    warmstart_damping: float = 0.5,
) -> np.ndarray:
    """Initialize phi_k with optional warm-start from previous iteration.

    Args:
        phi_meas: Log of measured rotations, shape (M, 3)
        prev_delta: Previous outer iteration delta for warm-starting
        warmstart_damping: Damping factor for warm-start (0-1)

    Returns:
        phi_k: Initial tangent space parameters
    """
    if prev_delta is None:
        return phi_meas.copy()
    else:
        # Damped warm-start: phi_k = phi_meas + damping * prev_delta
        prev_delta_unstacked = _unstack_phi(prev_delta)
        return phi_meas + warmstart_damping * prev_delta_unstacked


def _compute_adaptive_eta(
    rho_history: List[float],
    eta_min: float = 0.1,
    eta_good: float = 0.75,
    eta_bad: float = 0.25,
    adaptive: bool = True,
) -> Tuple[float, float, float]:
    """Compute adaptive trust-region parameters based on acceptance history.

    Args:
        rho_history: List of recent rho_k values (acceptance ratios)
        eta_min: Minimum acceptance ratio (lower bound)
        eta_good: Good improvement threshold (increase trust region)
        eta_bad: Poor improvement threshold (decrease trust region)
        adaptive: Enable adaptive parameter adjustment

    Returns:
        (eta_min, eta_good, eta_bad) - Possibly adjusted thresholds
    """
    if not adaptive or len(rho_history) < 5:
        return eta_min, eta_good, eta_bad

    # Adjust thresholds based on recent performance
    recent_rho = rho_history[-min(10, len(rho_history)):]
    avg_rho = np.mean(recent_rho)
    std_rho = np.std(recent_rho)

    # If highly variable, be more conservative
    if std_rho > 0.3:
        eta_bad = max(eta_bad - 0.1, 0.1)
        eta_good = min(eta_good + 0.1, 0.9)

    # If consistently accepting, be more aggressive
    if avg_rho > 0.8:
        eta_min = max(eta_min + 0.05, 0.2)

    return eta_min, eta_good, eta_bad


def _compute_convergence_metrics(
    delta_history: List[np.ndarray],
    obj_history: List[float],
    max_outer: int,
) -> Dict[str, Any]:
    """Compute convergence metrics for theoretical analysis.

    Returns:
        Dict with:
        - 'convergence_rate': Estimated linear convergence rate
        - 'final_delta_norm': Norm of final update
        - 'iterations_used': Actual iterations run
        - 'obj_decrease_total': Total objective decrease
        - 'obj_decrease_last10': Objective decrease in last 10 iterations
    """
    if len(delta_history) < 2:
        return {
            'convergence_rate': None,
            'final_delta_norm': None,
            'iterations_used': len(delta_history),
            'obj_decrease_total': None,
            'obj_decrease_last10': None,
        }

    # Estimate convergence rate from last several updates
    recent_deltas = delta_history[-min(10, len(delta_history)):]
    norms = [np.linalg.norm(d) for d in recent_deltas]

    # Linear convergence rate estimate
    if len(norms) >= 3:
        # Fit exponential decay: ||delta||_k ≈ C * rate^k
        log_norms = np.log(np.asarray(norms) + 1e-12)
        k = np.arange(len(log_norms))
        A = np.vstack([k, np.ones_like(k)]).T
        try:
            coeffs = np.linalg.lstsq(A, log_norms, rcond=None)[0]
            rate = np.exp(coeffs[0])
            convergence_rate = min(max(rate, 0.0), 1.0)
        except (np.linalg.LinAlgError, ValueError):
            convergence_rate = None
    else:
        convergence_rate = None

    return {
        'convergence_rate': float(convergence_rate) if convergence_rate is not None else None,
        'final_delta_norm': float(norms[-1]),
        'iterations_used': len(delta_history),
        'obj_decrease_total': obj_history[0] - obj_history[-1] if len(obj_history) >= 2 else None,
        'obj_decrease_last10': obj_history[-10] - obj_history[-1] if len(obj_history) >= 10 else None,
    }


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

    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(delta, cp.psd_wrap(H)) + g @ delta), constraints)
    if solver.upper() == "ECOS":
        prob.solve(solver=cp.ECOS, verbose=False)
    else:
        prob.solve(solver=cp.SCS, verbose=False, eps=1e-5, max_iters=20_000)

    if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"CVXPY reference failed: {prob.status}")

    return np.asarray(delta.value).reshape(-1), {"status": prob.status, "obj": float(prob.value)}


def tube_smooth_fast(
    R_meas: List[np.ndarray],
    eps: np.ndarray,
    lam: float,
    mu: float,
    tau: float,
    max_outer: int = 20,
    Delta: float = 0.2,
    rho: float = 1.0,
    inner_max_iter: int = 2000,
    tol_outer: float = 1e-6,
    tol_inner: float = 1e-4,
    gauge_fix: bool = False,  # If True, anchor first rotation to zero (explicit gauge fixing).
    slack: bool = False,  # If True, add slack variables for infeasible constraints.
    eta_min: float = 0.1,  # Minimum accept ratio for trust region
    eta_good: float = 0.75,  # Threshold for increasing trust region
    eta_bad: float = 0.25,  # Threshold for decreasing trust region
    warmstart: bool = False,  # If True, enable warm-starting from previous iterations.
    warmstart_damping: float = 0.5,  # Damping factor for warm-starting (0-1).
    adaptive_eta: bool = False,  # If True, enable adaptive trust-region parameter adjustment.
    check_kkt: bool = False,  # If True, check KKT conditions for optimality verification.
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """Tube smoothing with sequential convexification and custom ADMM inner iterations.

    Args:
        R_meas: Measured rotations, shape (M,3,3).
        eps: Tube radius per sample in radians, shape (M,).
        lam: First-order smoothness weight.
        mu: Second-order smoothness weight.
        tau: Time step.
        max_outer: Max outer sequential-convexification iterations.
        Delta: Initial trust-region radius for each increment block.
        rho: float.
        inner_max_iter: Maximum ADMM iterations per outer iteration.
        tol_outer: Stop if ||delta||_inf < tol.
        tol_inner: ADMM convergence tolerance.
        gauge_fix: If True, anchor first rotation to zero (explicit gauge fixing).
        slack: If True, add slack variables for infeasible constraints.
        eta_min: Minimum accept ratio (default 0.1).
        eta_good: Threshold for increasing trust region (default 0.75).
        eta_bad: Threshold for decreasing trust region (default 0.25).
        warmstart: If True, enable warm-starting from previous iterations.
        warmstart_damping: Damping factor for warm-starting (0-1, default 0.5).
        adaptive_eta: If True, enable adaptive trust-region parameter adjustment.
        check_kkt: If True, check KKT conditions for optimality verification.

    Returns:
        R_hat: Smoothed rotations (M,3,3).
        info: Diagnostic dictionary, including 'acceptances' and 'final_Delta'.
    """

    R_meas = np.asarray(R_meas, dtype=np.float64)
    eps = np.asarray(eps, dtype=np.float64).reshape(-1)
    M = R_meas.shape[0]
    if R_meas.shape != (M, 3, 3):
        raise ValueError("R_meas must have shape (M,3,3)")
    if eps.shape[0] != M:
        raise ValueError("eps length must match R_meas")

    H = build_H(M, lam, mu, tau, damping=1e-9)
    phi_meas = np.vstack([log_so3(R_meas[i]) for i in range(M)])
    phi_k = _initialize_phi_warmstart(phi_meas, None, warmstart_damping)
    prev_delta = None  # Store previous delta for warm-starting
    prev_u = None  # Store previous dual variables for warm-starting
    prev_v = None  # Store previous dual variables for warm-starting

    # Principled gauge fixing: penalize first rotation away from zero
    # This removes gauge freedom by anchoring the reference frame
    if gauge_fix:
        # Large penalty on first rotation's first two components
        gauge_weight = 1e6  # Strong constraint to fix phi[0] ≈ 0
        # Add to Hessian: modify the (0,0) and (0,1) diagonal entries
        # Since phi is stacked as [phi_0[0], phi_0[1], phi_0[2], phi_1[0], ...],
        # indices 0 and 1 correspond to phi[0][0] and phi[0][1]
        H = H.tocsc()
        H[0, 0] += gauge_weight
        H[1, 1] += gauge_weight

    objective_hist: List[float] = []
    max_violation_hist: List[float] = []
    avg_violation_hist: List[float] = []
    inner_stats_hist: List[Dict[str, Any]] = []
    outer_elapsed_hist: List[float] = []
    rho_history: List[float] = []  # Track acceptance ratios for adaptive eta
    delta_history: List[np.ndarray] = []  # Track deltas for convergence analysis

    total_start = time.perf_counter()
    final_delta_inf = np.inf

    for _ in range(max_outer):
        t_outer = time.perf_counter()

        Delta_curr = Delta  # Current trust region radius

        # Track acceptance statistics
        acceptances: List[int] = []

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
            tol=tol_inner,
            slack=slack,  # Pass slack parameter to ADMM solver
            u_init=prev_u,  # Warm-start dual variables for tube constraints
            v_init=prev_v,  # Warm-start dual variables for trust region
            check_kkt=check_kkt,  # Pass KKT checking flag
        )
        inner_stats_hist.append(inner_stats)

        # Compute current objective
        phi_k_vec = _stack_phi(phi_k)
        obj_current = float(0.5 * phi_k_vec @ (H @ phi_k_vec))

        # Trust-region logic: evaluate actual vs predicted decrease
        # Compute candidate phi update
        phi_candidate = np.vstack([retract(phi_k[i], delta[3*i:3*(i+1)]) for i in range(M)])
        phi_candidate_vec = _stack_phi(phi_candidate)
        # Actual objective decrease
        obj_candidate = float(0.5 * phi_candidate_vec @ (H @ phi_candidate_vec))
        actual_decrease = obj_current - obj_candidate
        # Predicted decrease from linear model (gradient = H@phi_k + g)
        # Linear approximation: obj(phi_k + delta) ≈ obj_k + delta^T (H@phi_k + g)
        # Since g = H @ phi_k, predicted: -delta^T g
        pred_decrease = -delta @ g
        # Avoid division by zero
        if abs(pred_decrease) > 1e-12:
            rho_k = actual_decrease / pred_decrease
        else:
            rho_k = np.inf

        # Track rho_k for adaptive eta
        rho_history.append(rho_k)

        # Compute adaptive trust-region parameters
        eta_min_curr, eta_good_curr, eta_bad_curr = _compute_adaptive_eta(
            rho_history, eta_min, eta_good, eta_bad, adaptive_eta
        )

        # Acceptance rule based on rho_k
        if rho_k >= eta_min_curr:
            # Accept update
            phi_k = phi_candidate
            acceptances.append(1)
            # Store delta and dual variables for warm-starting next iteration
            if warmstart:
                prev_delta = delta.copy()
                # Extract dual variables from inner_stats if available
                if "u" in inner_stats:
                    prev_u = inner_stats["u"].copy()
                if "v" in inner_stats:
                    prev_v = inner_stats["v"].copy()
            # Adjust trust region
            if rho_k > eta_good_curr:
                Delta_curr = min(2.0 * Delta_curr, 1.0)  # Increase
            elif rho_k < eta_bad_curr:
                Delta_curr = max(0.5 * Delta_curr, 1e-4)  # Decrease
        else:
            # Reject update
            acceptances.append(0)
            # Decrease trust region more aggressively
            Delta_curr = max(0.5 * Delta_curr, 1e-4)

        # Use manifold retraction instead of additive update

        phi_vec = _stack_phi(phi_k)
        objective_hist.append(float(0.5 * phi_vec @ (H @ phi_vec)))

        violations = np.array(
            [np.linalg.norm(log_so3(R_meas[i].T @ exp_so3(phi_k[i]))) - eps[i] for i in range(M)]
        )
        max_violation_hist.append(float(np.max(violations)))
        avg_violation_hist.append(float(np.mean(np.maximum(violations, 0.0))))

        final_delta_inf = float(np.max(np.abs(delta)))
        delta_history.append(delta.copy())  # Track delta for convergence analysis
        outer_elapsed_hist.append(time.perf_counter() - t_outer)
        if final_delta_inf < tol_outer:
            break

    elapsed = time.perf_counter() - total_start
    R_hat = np.stack([exp_so3(phi_k[i]) for i in range(M)], axis=0)

    # Compute convergence metrics
    convergence_metrics = _compute_convergence_metrics(delta_history, objective_hist, max_outer)

    info: Dict[str, Any] = {
        "outer_iter": len(objective_hist),
        "objective": objective_hist,
        "max_violation": max_violation_hist,
        "avg_violation": avg_violation_hist,
        "inner_stats": inner_stats_hist,
        "outer_elapsed": outer_elapsed_hist,
        "elapsed_sec": elapsed,
        "final_delta_inf": final_delta_inf,
        "converged": final_delta_inf < tol_outer,
        "acceptances": acceptances,  # Track acceptance rate
        "final_Delta": Delta_curr,  # Final trust region radius
        "rho_history": rho_history,  # Acceptance ratio history
        **convergence_metrics,  # Convergence rate and other metrics
    }
    # Track active slack indices from ADMM solver if slack is enabled
    if slack and len(inner_stats_hist) > 0:
        stats = inner_stats_hist[-1]  # Last iteration's stats
        info["active_slack_indices"] = stats.get("active_slack_indices", [])
    else:
        info["active_slack_indices"] = []
    return R_hat, info
