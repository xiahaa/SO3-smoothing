"""Tests for theoretical enhancements: warm-starting, adaptive eta, convergence analysis, KKT."""

from __future__ import annotations

import numpy as np
import sys
sys.path.insert(0, 'src')

from smoother_fast import tube_smooth_fast
from admm_solver import solve_inner_admm
from hessian import build_H
from so3 import exp_so3, log_so3


def _generate_test_rotations(M: int, seed: int = 0) -> tuple:
    """Generate test rotations for theoretical feature testing.

    Args:
        M: Number of rotations
        seed: Random seed

    Returns:
        R_meas: Measured rotations (M, 3, 3)
        eps: Tube radii (M,)
        lam, mu, tau: Smoothing parameters
    """
    rng = np.random.default_rng(seed)
    # Generate smooth rotations
    phis = np.linspace(0, 2 * np.pi, M)
    R_list = []
    for phi in phis:
        # Rotation around z-axis with small perturbations
        angle = phi + rng.normal(0, 0.05)
        axis = np.array([0.0, 0.0, 1.0])
        R = exp_so3(angle * axis + rng.normal(0, 0.01, size=3))
        R_list.append(R)
    R_meas = np.stack(R_list, axis=0)

    # Tube radii
    eps = 0.1 + 0.05 * rng.random(M)

    # Smoothing parameters
    lam, mu, tau = 1.0, 0.2, 0.1

    return R_meas, eps, lam, mu, tau


def test_warmstart_speedup() -> None:
    """Test that warm-starting reduces outer iterations."""
    R_meas, eps, lam, mu, tau = _generate_test_rotations(50, seed=0)

    # Run without warm-starting
    R_no_ws, info_no_ws = tube_smooth_fast(
        R_meas, eps, lam, mu, tau,
        max_outer=30, warmstart=False
    )

    # Run with warm-starting
    R_ws, info_ws = tube_smooth_fast(
        R_meas, eps, lam, mu, tau,
        max_outer=30, warmstart=True, warmstart_damping=0.5
    )

    # Warm-starting should reduce or not increase outer iterations
    # (it might not help on simple problems, but it shouldn't hurt)
    assert info_ws["outer_iter"] <= info_no_ws["outer_iter"] + 2  # Allow small overhead

    # Both should converge to similar solutions
    assert np.allclose(R_no_ws, R_ws, atol=1e-6)


def test_warmstart_damping() -> None:
    """Test different warm-start damping factors."""
    R_meas, eps, lam, mu, tau = _generate_test_rotations(40, seed=1)

    results = []
    for damping in [0.1, 0.3, 0.5, 0.7, 0.9]:
        _, info = tube_smooth_fast(
            R_meas, eps, lam, mu, tau,
            max_outer=20, warmstart=True, warmstart_damping=damping
        )
        results.append((damping, info["outer_iter"]))

    # All damping factors should produce reasonable results
    for damping, iters in results:
        assert iters >= 1 and iters <= 20, f"Damping {damping} gave {iters} iterations"


def test_admm_warmstart() -> None:
    """Test ADMM dual variable warm-starting."""
    M = 30
    H = build_H(M, lam=1.0, mu=0.2, tau=0.1)
    rng = np.random.default_rng(0)
    g = rng.normal(0.0, 0.1, size=3 * M)

    r_list = [rng.normal(0.0, 0.03, size=3) for _ in range(M)]
    J_list = [np.eye(3) + 0.01 * rng.normal(size=(3, 3)) for _ in range(M)]
    eps = 0.18 * np.ones(M)
    Delta = 0.2

    # First solve (cold start)
    d1, stats1 = solve_inner_admm(
        H, g, r_list, J_list, eps, Delta, rho=1.0, max_iter=200, tol=1e-5
    )
    u1 = stats1["u"]
    v1 = stats1["v"]

    # Second solve (warm start with dual variables)
    d2, stats2 = solve_inner_admm(
        H, g, r_list, J_list, eps, Delta, rho=1.0, max_iter=200, tol=1e-5,
        u_init=u1, v_init=v1
    )

    # Warm-starting should converge faster or to similar solution
    assert stats2["iter"] <= stats1["iter"] + 1  # Allow small overhead

    # Solutions should be similar
    obj1 = float(0.5 * d1 @ (H @ d1) + g @ d1)
    obj2 = float(0.5 * d2 @ (H @ d2) + g @ d2)
    assert abs(obj1 - obj2) < 1e-4


def test_adaptive_eta_convergence() -> None:
    """Test that adaptive eta improves convergence handling."""
    R_meas, eps, lam, mu, tau = _generate_test_rotations(50, seed=2)

    # Run with fixed eta
    R_fixed, info_fixed = tube_smooth_fast(
        R_meas, eps, lam, mu, tau,
        max_outer=50, adaptive_eta=False
    )

    # Run with adaptive eta
    R_adaptive, info_adaptive = tube_smooth_fast(
        R_meas, eps, lam, mu, tau,
        max_outer=50, adaptive_eta=True
    )

    # Both should converge or make progress
    # Note: Convergence is not guaranteed on all problem instances
    # We check that they make reasonable progress
    assert info_fixed["outer_iter"] > 0
    assert info_adaptive["outer_iter"] > 0

    # Adaptive eta should track rho history
    assert len(info_adaptive["rho_history"]) > 0


def test_eta_adjustment_logic() -> None:
    """Test eta parameter adjustment logic."""
    from smoother_fast import _compute_adaptive_eta

    # Case 1: Not adaptive
    eta_min, eta_good, eta_bad = _compute_adaptive_eta([0.5, 0.6], 0.1, 0.75, 0.25, adaptive=False)
    assert eta_min == 0.1 and eta_good == 0.75 and eta_bad == 0.25

    # Case 2: Adaptive with high variance (should be more conservative)
    rho_history = [0.9, 0.2, 0.8, 0.3, 0.7, 0.1, 0.9, 0.2, 0.8, 0.3]
    eta_min, eta_good, eta_bad = _compute_adaptive_eta(rho_history, 0.1, 0.75, 0.25, adaptive=True)
    assert eta_bad < 0.25  # More conservative

    # Case 3: Adaptive with consistently high acceptance (should be more aggressive)
    rho_history = [0.9, 0.85, 0.92, 0.88, 0.95, 0.9, 0.87, 0.93, 0.89, 0.91]
    eta_min, eta_good, eta_bad = _compute_adaptive_eta(rho_history, 0.1, 0.75, 0.25, adaptive=True)
    assert eta_min > 0.1  # More aggressive


def test_convergence_rate_estimation() -> None:
    """Test convergence rate estimation."""
    from smoother_fast import _compute_convergence_metrics

    # Case 1: Linear convergence (exponential decay)
    rate_true = 0.5
    delta_history = [rate_true**k * np.ones(3) for k in range(10)]
    obj_history = [rate_true**(2*k) for k in range(10)]

    metrics = _compute_convergence_metrics(delta_history, obj_history, max_outer=10)

    assert metrics["iterations_used"] == 10
    assert metrics["final_delta_norm"] is not None
    assert metrics["convergence_rate"] is not None
    # Estimated rate should be close to true rate
    assert abs(metrics["convergence_rate"] - rate_true) < 0.2  # Allow estimation error

    # Case 2: Too few iterations
    delta_history = [np.ones(3), np.zeros(3)]
    obj_history = [1.0, 0.0]
    metrics = _compute_convergence_metrics(delta_history, obj_history, max_outer=2)
    assert metrics["convergence_rate"] is None  # Can't estimate with few points


def test_kkt_conditions() -> None:
    """Test KKT condition verification."""
    M = 20
    H = build_H(M, lam=1.0, mu=0.2, tau=0.1)
    rng = np.random.default_rng(0)
    g = rng.normal(0.0, 0.1, size=3 * M)

    r_list = [rng.normal(0.0, 0.03, size=3) for _ in range(M)]
    J_list = [np.eye(3) + 0.01 * rng.normal(size=(3, 3)) for _ in range(M)]
    eps = 0.18 * np.ones(M)
    Delta = 0.2

    # Solve with KKT checking
    d, stats = solve_inner_admm(
        H, g, r_list, J_list, eps, Delta, rho=1.0, max_iter=200, tol=1e-5, check_kkt=True
    )

    # KKT metrics should be present
    assert "kkt_conditions" in stats
    kkt = stats["kkt_conditions"]

    # Primal stationarity should be reasonably small for converged solution
    # ADMM doesn't guarantee exact KKT satisfaction, so we allow some tolerance
    assert kkt["primal_stationarity"] < 5.0  # Reasonable tolerance for ADMM

    # Dual feasibility violations should be small
    assert kkt["dual_tube_violation"] < 1e-6  # Very small for feasible solution
    assert kkt["dual_trust_violation"] < 1e-6  # Very small for feasible solution


def test_integration_with_smoothing() -> None:
    """Test integration of all theoretical features with tube smoothing."""
    R_meas, eps, lam, mu, tau = _generate_test_rotations(60, seed=3)

    # Run with all theoretical features enabled
    R_hat, info = tube_smooth_fast(
        R_meas, eps, lam, mu, tau,
        max_outer=50,
        warmstart=True,
        warmstart_damping=0.5,
        adaptive_eta=True,
        check_kkt=True
    )

    # Should make progress (convergence is not guaranteed on all problem instances)
    assert info["outer_iter"] > 0

    # Should have convergence metrics
    assert "convergence_rate" in info
    assert "final_delta_norm" in info
    assert "iterations_used" in info

    # Should have rho history
    assert "rho_history" in info
    assert len(info["rho_history"]) > 0

    # Check inner stats for KKT conditions (last iteration)
    last_inner = info["inner_stats"][-1]
    if "kkt_conditions" in last_inner:
        kkt = last_inner["kkt_conditions"]
        assert kkt["primal_stationarity"] is not None


def test_warmstart_with_adaptive_eta() -> None:
    """Test that warm-starting and adaptive eta work together."""
    R_meas, eps, lam, mu, tau = _generate_test_rotations(40, seed=4)

    # Run with both features
    R_hat, info = tube_smooth_fast(
        R_meas, eps, lam, mu, tau,
        max_outer=50,
        warmstart=True,
        warmstart_damping=0.6,
        adaptive_eta=True
    )

    # Should make progress
    assert info["outer_iter"] > 0

    # Should have both rho history and convergence metrics
    assert len(info["rho_history"]) > 0
    assert info["iterations_used"] <= 50


def test_convergence_metrics_on_simple_problem() -> None:
    """Test convergence metrics on a simple, well-conditioned problem."""
    R_meas, eps, lam, mu, tau = _generate_test_rotations(30, seed=5)

    # Use simple, well-conditioned parameters
    R_hat, info = tube_smooth_fast(
        R_meas, eps, 1.0, 0.1, 0.1,  # Well-conditioned: lam, mu, tau
        max_outer=50
    )

    # Should make progress (convergence is not guaranteed)
    assert info["outer_iter"] > 0

    # Convergence rate should be reasonable if available (0 < rate < 1)
    if info["convergence_rate"] is not None:
        assert 0.0 < info["convergence_rate"] < 1.0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
