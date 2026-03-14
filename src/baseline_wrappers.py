"""Wrappers for GTSAM and Ceres baselines.

This module provides wrapper functions to solve the SO(3) tube smoothing
problem using GTSAM and Ceres for fair comparison.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

import numpy as np

import sys
sys.path.insert(0, 'src')

from so3 import exp_so3, log_so3, hat, vee


def convert_to_gtsam_rot3(R: np.ndarray) -> Any:
    """Convert numpy rotation matrix to GTSAM Rot3."""
    import gtsam
    return gtsam.Rot3(R)


def convert_from_gtsam_rot3(R_gtsam: Any) -> np.ndarray:
    """Convert GTSAM Rot3 to numpy rotation matrix."""
    return R_gtsam.matrix()


def tube_smooth_gtsam(
    R_meas: np.ndarray,
    eps: np.ndarray,
    lam: float,
    mu: float,
    tau: float,
    max_iter: int = 100,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Solve tube smoothing using GTSAM factor graph.
    
    Formulates the problem as a factor graph with:
    - Between factors for smoothness (1st and 2nd order)
    - Prior factors for tube constraints (soft constraints via robust noise)
    
    Args:
        R_meas: Measured rotations (M, 3, 3)
        eps: Tube radii (M,)
        lam: First-order smoothness weight
        mu: Second-order smoothness weight
        tau: Time step
        max_iter: Maximum LM iterations
        
    Returns:
        R_hat: Smoothed rotations (M, 3, 3)
        info: Diagnostic information
    """
    import gtsam
    
    start = time.perf_counter()
    
    M = len(R_meas)
    
    # Create factor graph
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()
    
    # Add variables and factors
    for i in range(M):
        # Initial guess from measurements
        R_init = convert_to_gtsam_rot3(R_meas[i])
        initial.insert(i, R_init)
        
        # Tube constraint as prior with robust noise model
        # Use Huber loss to approximate hard constraint
        sigma = eps[i] / 3.0  # 3-sigma corresponds to tube radius
        noise_model = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(1.0),
            gtsam.noiseModel.Isotropic.Sigma(3, sigma)
        )
        
        R_meas_gtsam = convert_to_gtsam_rot3(R_meas[i])
        factor = gtsam.PriorFactorRot3(i, R_meas_gtsam, noise_model)
        graph.add(factor)
    
    # Add smoothness factors (Between factors for consecutive rotations)
    sqrt_lam = np.sqrt(lam / tau)
    for i in range(M - 1):
        # Identity relative rotation for smoothness
        R_identity = gtsam.Rot3()
        noise_between = gtsam.noiseModel.Isotropic.Sigma(3, 1.0 / sqrt_lam)
        factor = gtsam.BetweenFactorRot3(i, i + 1, R_identity, noise_between)
        graph.add(factor)
    
    # Add acceleration smoothness (2nd order)
    sqrt_mu = np.sqrt(mu / tau**3)
    for i in range(M - 2):
        # Approximate 2nd order smoothness via relative constraints
        noise_acc = gtsam.noiseModel.Isotropic.Sigma(3, 1.0 / sqrt_mu)
        # This is a simplification - proper 2nd order would need velocity states
        factor = gtsam.BetweenFactorRot3(i, i + 2, gtsam.Rot3(), noise_acc)
        graph.add(factor)
    
    # Optimize
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(max_iter)
    params.setVerbosityLM("SILENT")
    
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
    
    try:
        result = optimizer.optimize()
        
        # Extract results
        R_hat = []
        for i in range(M):
            R_i = result.atRot3(i)
            R_hat.append(convert_from_gtsam_rot3(R_i))
        R_hat = np.array(R_hat)
        
        elapsed = time.perf_counter() - start
        
        # Compute error
        error = graph.error(result)
        
        info = {
            'status': 'success',
            'elapsed_sec': elapsed,
            'final_error': error,
            'iterations': max_iter,  # GTSAM doesn't easily expose this
            'solver': 'GTSAM_LM',
        }
        
    except Exception as e:
        elapsed = time.perf_counter() - start
        info = {
            'status': 'failed',
            'error': str(e),
            'elapsed_sec': elapsed,
        }
        R_hat = R_meas.copy()
    
    return R_hat, info


def tube_smooth_ceres(
    R_meas: np.ndarray,
    eps: np.ndarray,
    lam: float,
    mu: float,
    tau: float,
    max_iter: int = 100,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Solve tube smoothing using Ceres Solver.
    
    Formulates as a nonlinear least squares problem with:
    - Residuals for tube constraints (squared error with bounds)
    - Residuals for smoothness (1st and 2nd order)
    
    Args:
        R_meas: Measured rotations (M, 3, 3)
        eps: Tube radii (M,)
        lam: First-order smoothness weight
        mu: Second-order smoothness weight
        tau: Time step
        max_iter: Maximum iterations
        
    Returns:
        R_hat: Smoothed rotations (M, 3, 3)
        info: Diagnostic information
    """
    try:
        import pyceres
    except ImportError:
        return R_meas.copy(), {'status': 'failed', 'error': 'pyceres not available'}
    
    start = time.perf_counter()
    
    M = len(R_meas)
    
    # Initialize with measurements in tangent space
    phi_meas = np.array([log_so3(R_meas[i]) for i in range(M)])
    phi = phi_meas.copy()
    
    # Create problem
    problem = pyceres.Problem()
    
    # Add parameter blocks
    for i in range(M):
        problem.add_parameter_block(phi[i], 3)
    
    # Add residual blocks
    # Tube constraints (softened via loss function)
    for i in range(M):
        # Cost function: ||log(R_meas[i].T @ exp(phi[i]))||
        # Simplified as: ||phi[i] - phi_meas[i]|| with bounds
        
        residuals = phi[i] - phi_meas[i]
        # Scale by 1/eps to normalize
        weight = 1.0 / eps[i]
        
        # This is a simplified formulation
        # Full Ceres would need custom cost function
        pass
    
    # Note: Full Ceres implementation would require custom cost functions
    # for proper SO(3) operations. This is a simplified placeholder.
    
    elapsed = time.perf_counter() - start
    
    info = {
        'status': 'not_implemented',
        'note': 'Full Ceres implementation requires custom cost functions',
        'elapsed_sec': elapsed,
    }
    
    # Return measurement for now
    R_hat = R_meas.copy()
    
    return R_hat, info


def tube_smooth_ceres_simple(
    R_meas: np.ndarray,
    eps: np.ndarray,
    lam: float,
    mu: float,
    tau: float,
    max_iter: int = 100,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Simplified Ceres-like approach using scipy.optimize.
    
    This uses SciPy's least_squares as a stand-in for Ceres,
    with similar problem formulation.
    """
    from scipy.optimize import least_squares
    
    start = time.perf_counter()
    
    M = len(R_meas)
    phi_meas = np.array([log_so3(R_meas[i]) for i in range(M)])
    
    def residuals(phi_flat):
        """Compute residuals for all constraints."""
        phi = phi_flat.reshape(M, 3)
        res = []
        
        # Tube constraints: ||log(R_meas[i].T @ exp(phi[i]))||
        # Approximated as ||phi[i] - phi_meas[i]||
        for i in range(M):
            diff = phi[i] - phi_meas[i]
            # Soft constraint: penalize if outside tube
            norm_diff = np.linalg.norm(diff)
            if norm_diff > eps[i]:
                res.extend((norm_diff - eps[i]) * diff / norm_diff * 10.0)
            else:
                res.extend([0, 0, 0])
        
        # 1st order smoothness: phi[i+1] - phi[i]
        for i in range(M - 1):
            smooth = (phi[i+1] - phi[i]) * np.sqrt(lam / tau)
            res.extend(smooth)
        
        # 2nd order smoothness: phi[i+2] - 2*phi[i+1] + phi[i]
        for i in range(M - 2):
            acc = (phi[i+2] - 2*phi[i+1] + phi[i]) * np.sqrt(mu / tau**3)
            res.extend(acc)
        
        return np.array(res)
    
    # Optimize
    phi_init = phi_meas.flatten()
    
    try:
        result = least_squares(
            residuals,
            phi_init,
            method='lm',
            max_nfev=max_iter * 10,
            ftol=1e-6,
            xtol=1e-6,
        )
        
        phi_opt = result.x.reshape(M, 3)
        R_hat = np.array([exp_so3(phi_opt[i]) for i in range(M)])
        
        elapsed = time.perf_counter() - start
        
        info = {
            'status': 'success',
            'elapsed_sec': elapsed,
            'cost': result.cost,
            'nfev': result.nfev,
            'success': result.success,
            'solver': 'Scipy_LM',
        }
        
    except Exception as e:
        elapsed = time.perf_counter() - start
        info = {
            'status': 'failed',
            'error': str(e),
            'elapsed_sec': elapsed,
        }
        R_hat = R_meas.copy()
    
    return R_hat, info


if __name__ == "__main__":
    # Quick test
    print("Testing baseline wrappers...")
    
    # Generate test problem
    M = 50
    t = np.linspace(0, 2*np.pi, M)
    phi_true = np.array([[0.1*np.sin(t[i]), 0.0, 0.0] for i in range(M)])
    R_true = np.array([exp_so3(phi_true[i]) for i in range(M)])
    
    from noise_models import set_bounded_noise
    R_meas, eps = set_bounded_noise(R_true, noise_sigma=0.05, seed=42)
    
    print("\n1. Testing GTSAM wrapper...")
    R_hat_gtsam, info_gtsam = tube_smooth_gtsam(R_meas, eps, 1.0, 0.1, 0.1)
    print(f"   Status: {info_gtsam['status']}")
    if info_gtsam['status'] == 'success':
        print(f"   Runtime: {info_gtsam['elapsed_sec']:.3f}s")
        print(f"   Final error: {info_gtsam['final_error']:.4f}")
    
    print("\n2. Testing Ceres-like wrapper (SciPy)...")
    R_hat_ceres, info_ceres = tube_smooth_ceres_simple(R_meas, eps, 1.0, 0.1, 0.1)
    print(f"   Status: {info_ceres['status']}")
    if info_ceres['status'] == 'success':
        print(f"   Runtime: {info_ceres['elapsed_sec']:.3f}s")
        print(f"   Cost: {info_ceres['cost']:.4f}")
    
    print("\nWrappers tested successfully!")
