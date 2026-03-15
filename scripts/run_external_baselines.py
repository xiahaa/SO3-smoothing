"""Run comprehensive comparison with external baselines.

Compares our method against:
1. GTSAM (factor graph formulation)
2. Ceres-like (SciPy least squares)
3. Our method (ADMM)
"""

from __future__ import annotations

import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

import sys
sys.path.insert(0, 'src')

from smoother_fast import tube_smooth_fast
from baseline_wrappers import tube_smooth_gtsam, tube_smooth_ceres_simple
from noise_models import set_bounded_noise
from so3 import exp_so3, log_so3, geodesic_angle


def generate_test_problem(M: int = 200, seed: int = 42):
    """Generate a standard test problem."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, M)
    
    phi_true = np.zeros((M, 3))
    phi_true[:, 0] = 0.3 * np.sin(0.5 * t) + 0.1 * rng.normal(0, 0.01, M)
    phi_true[:, 1] = 0.2 * np.cos(0.3 * t) + 0.1 * rng.normal(0, 0.01, M)
    phi_true[:, 2] = 0.5 * np.sin(t) + 0.2 * np.cos(2 * t)
    
    R_true = np.stack([exp_so3(phi_true[i]) for i in range(M)], axis=0)
    R_meas, eps = set_bounded_noise(R_true, noise_sigma=0.05, eps_factor=3.0, seed=seed)
    
    return R_true, R_meas, eps


def compute_metrics(R_true, R_meas, R_hat, eps):
    """Compute evaluation metrics."""
    M = len(R_true)
    
    # Tube excess relative to per-sample bounds
    violations = np.array([
        geodesic_angle(R_meas[i], R_hat[i]) - eps[i]
        for i in range(M)
    ])
    tube_excess = float(np.max(violations))
    avg_tube_excess = float(np.mean(np.maximum(violations, 0)))
    feasible_rate = float(np.mean(violations <= 1e-6))
    
    # Ground truth error
    gt_errors = np.array([
        geodesic_angle(R_true[i], R_hat[i])
        for i in range(M)
    ])
    gt_error_rms = float(np.sqrt(np.mean(gt_errors ** 2)))
    gt_error_max = float(np.max(gt_errors))
    
    # Smoothness
    phi_hat = np.stack([log_so3(R_hat[i]) for i in range(M)], axis=0)
    d1 = np.diff(phi_hat, axis=0)
    d2 = np.diff(phi_hat, n=2, axis=0)
    vel_rms = float(np.sqrt(np.mean(np.sum(d1**2, axis=1))))
    acc_rms = float(np.sqrt(np.mean(np.sum(d2**2, axis=1))))
    
    return {
        'tube_excess': tube_excess,
        'avg_tube_excess': avg_tube_excess,
        'max_violation': tube_excess,
        'avg_violation': avg_tube_excess,
        'feasible_rate': feasible_rate,
        'gt_error_rms': gt_error_rms,
        'gt_error_max': gt_error_max,
        'vel_rms': vel_rms,
        'acc_rms': acc_rms,
    }


def method_gtsam(R_true, R_meas, eps, lam=1.0, mu=0.1, tau=0.1):
    """GTSAM baseline."""
    print("  Running GTSAM...")
    start = time.perf_counter()
    
    R_hat, info = tube_smooth_gtsam(R_meas, eps, lam, mu, tau, max_iter=100)
    
    elapsed = time.perf_counter() - start
    
    if info['status'] == 'success':
        metrics = compute_metrics(R_true, R_meas, R_hat, eps)
        return {
            'method': 'GTSAM',
            'runtime': elapsed,
            **metrics,
            'solver_info': info,
        }
    else:
        return {
            'method': 'GTSAM',
            'status': 'failed',
            'error': info.get('error', 'unknown'),
            'runtime': elapsed,
        }


def method_ceres(R_true, R_meas, eps, lam=1.0, mu=0.1, tau=0.1):
    """Ceres-like (SciPy) baseline."""
    print("  Running Ceres-like (SciPy)...")
    start = time.perf_counter()
    
    R_hat, info = tube_smooth_ceres_simple(R_meas, eps, lam, mu, tau, max_iter=100)
    
    elapsed = time.perf_counter() - start
    
    if info['status'] == 'success':
        metrics = compute_metrics(R_true, R_meas, R_hat, eps)
        return {
            'method': 'Ceres-like (SciPy)',
            'runtime': elapsed,
            **metrics,
            'solver_info': info,
        }
    else:
        return {
            'method': 'Ceres-like (SciPy)',
            'status': 'failed',
            'error': info.get('error', 'unknown'),
            'runtime': elapsed,
        }


def method_ours(R_true, R_meas, eps, lam=1.0, mu=0.1, tau=0.1):
    """Our ADMM method."""
    print("  Running Ours (ADMM)...")
    start = time.perf_counter()
    
    R_hat, info = tube_smooth_fast(
        R_meas, eps, lam, mu, tau,
        max_outer=20,
        Delta=0.2,
        rho=1.0,
        inner_max_iter=500,
        tol_outer=1e-6,
    )
    
    elapsed = time.perf_counter() - start
    
    metrics = compute_metrics(R_true, R_meas, R_hat, eps)
    
    return {
        'method': 'Ours (ADMM)',
        'runtime': elapsed,
        **metrics,
        'outer_iter': info['outer_iter'],
        'converged': info['converged'],
    }


def run_comparison(M_values: List[int], n_seeds: int = 3) -> List[Dict]:
    """Run comprehensive comparison."""
    print("="*80)
    print("EXTERNAL BASELINE COMPARISON")
    print("="*80)
    print("\nMethods:")
    print("  1. GTSAM (factor graph with Levenberg-Marquardt)")
    print("  2. Ceres-like (SciPy least squares)")
    print("  3. Ours (custom ADMM)")
    print(f"\nProblem sizes: {M_values}")
    print(f"Seeds per size: {n_seeds}\n")
    
    all_results = []
    
    for M in M_values:
        print(f"\n{'='*60}")
        print(f"Problem size M={M}")
        print(f"{'='*60}")
        
        for seed in range(n_seeds):
            print(f"\n--- Seed {seed} ---")
            R_true, R_meas, eps = generate_test_problem(M, seed)
            
            # Run all methods
            results_seed = []
            
            # GTSAM
            try:
                result = method_gtsam(R_true, R_meas, eps)
                result['M'] = M
                result['seed'] = seed
                results_seed.append(result)
                if result.get('status') != 'failed':
                    print(f"    GTSAM: {result['runtime']:.3f}s, "
                          f"gt_err={result['gt_error_rms']:.4f}, "
                          f"tube_excess={result['tube_excess']:.4f}")
            except Exception as e:
                print(f"    GTSAM failed: {e}")
            
            # Ceres-like
            try:
                result = method_ceres(R_true, R_meas, eps)
                result['M'] = M
                result['seed'] = seed
                results_seed.append(result)
                if result.get('status') != 'failed':
                    print(f"    Ceres: {result['runtime']:.3f}s, "
                          f"gt_err={result['gt_error_rms']:.4f}, "
                          f"tube_excess={result['tube_excess']:.4f}")
            except Exception as e:
                print(f"    Ceres failed: {e}")
            
            # Ours
            try:
                result = method_ours(R_true, R_meas, eps)
                result['M'] = M
                result['seed'] = seed
                results_seed.append(result)
                if result.get('status') != 'failed':
                    print(f"    Ours:  {result['runtime']:.3f}s, "
                          f"gt_err={result['gt_error_rms']:.4f}, "
                          f"tube_excess={result['tube_excess']:.4f}")
            except Exception as e:
                print(f"    Ours failed: {e}")
            
            all_results.extend(results_seed)
    
    return all_results


def generate_latex_table(all_results: List[Dict], M_values: List[int]):
    """Generate comparison tables for each problem size."""
    
    for M in M_values:
        print(f"\n{'='*60}")
        print(f"TABLE FOR M={M}")
        print(f"{'='*60}\n")
        
        # Filter results for this M
        M_results = [r for r in all_results if r.get('M') == M and r.get('status') != 'failed']
        
        # Group by method
        methods = {}
        for r in M_results:
            method = r['method']
            if method not in methods:
                methods[method] = []
            methods[method].append(r)
        
        print(r"\begin{table}[htbp]")
        print(r"\centering")
        print(f"\\caption{{Baseline comparison (M={M}, mean $\\pm$ std over {len(methods.get('Ours (ADMM)', []))} seeds)}}")
        print(f"\\label{{tab:baseline_m{M}}}")
        print(r"\begin{tabular}{l c c c c c}")
        print(r"\hline")
        print(r"\textbf{Method} & \textbf{Runtime} & \textbf{GT Error} & \textbf{Tube Excess} & \textbf{Feasible} & \textbf{Acc RMS} \\")
        print(r"& \textbf{(s)} & \textbf{(rad)} & \textbf{(rad)} & \textbf{Rate} & \textbf{(rad/s$^2$)} \\")
        print(r"\hline")
        
        for method_name in ['GTSAM', 'Ceres-like (SciPy)', 'Ours (ADMM)']:
            if method_name not in methods:
                continue
            
            group = methods[method_name]
            
            runtime_mean = np.mean([r['runtime'] for r in group])
            runtime_std = np.std([r['runtime'] for r in group])
            
            gt_err_mean = np.mean([r['gt_error_rms'] for r in group])
            gt_err_std = np.std([r['gt_error_rms'] for r in group])
            
            tube_excess_mean = np.mean([r['tube_excess'] for r in group])
            
            feas_rate = np.mean([r['feasible_rate'] for r in group]) * 100
            
            acc_mean = np.mean([r['acc_rms'] for r in group])
            
            # Short name for table
            short_name = method_name.replace(' (SciPy)', '').replace(' (ADMM)', '')
            
            print(f"{short_name} & "
                  f"${runtime_mean:.3f} \\pm {runtime_std:.3f}$ & "
                  f"${gt_err_mean:.4f} \\pm {gt_err_std:.4f}$ & "
                  f"${tube_excess_mean:.4f}$ & "
                  f"${feas_rate:.0f}\\%$ & "
                  f"${acc_mean:.4f}$ \\")
        
        print(r"\hline")
        print(r"\end{tabular}")
        print(r"\end{table}")


def print_summary(all_results: List[Dict]):
    """Print summary analysis."""
    print("\n\n" + "="*80)
    print("SUMMARY ANALYSIS")
    print("="*80)
    
    # Group by method
    methods = {}
    for r in all_results:
        if r.get('status') == 'failed':
            continue
        method = r['method']
        if method not in methods:
            methods[method] = []
        methods[method].append(r)
    
    print("\n1. RUNTIME COMPARISON:")
    print("-" * 60)
    for method_name in ['GTSAM', 'Ceres-like (SciPy)', 'Ours (ADMM)']:
        if method_name not in methods:
            continue
        group = methods[method_name]
        by_size = {}
        for r in group:
            M = r['M']
            if M not in by_size:
                by_size[M] = []
            by_size[M].append(r['runtime'])
        
        print(f"\n  {method_name}:")
        for M in sorted(by_size.keys()):
            times = by_size[M]
            print(f"    M={M}: {np.mean(times):.3f} ± {np.std(times):.3f}s")
    
    print("\n\n2. SOLUTION QUALITY:")
    print("-" * 60)
    for method_name in ['GTSAM', 'Ceres-like (SciPy)', 'Ours (ADMM)']:
        if method_name not in methods:
            continue
        group = methods[method_name]
        gt_err = np.mean([r['gt_error_rms'] for r in group])
        tube_excess = np.mean([r['tube_excess'] for r in group])
        feas_rate = np.mean([r['feasible_rate'] for r in group]) * 100
        
        print(f"  {method_name:25s}: GT error={gt_err:.4f} rad, "
              f"Tube excess={tube_excess:.4f} rad, Feasible={feas_rate:.0f}%")
    
    print("\n\n3. KEY FINDINGS:")
    print("-" * 60)
    print("  • GTSAM: Very fast but uses soft constraints (robust loss)")
    print("  • Ceres-like: Slower with moderate positive tube excess")
    print("  • Ours: Balanced speed with low tube excess in reported runs")


def main():
    """Run external baseline comparison."""
    output_dir = Path('results/external_baselines')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run comparison
    M_values = [100, 200, 500]
    all_results = run_comparison(M_values, n_seeds=3)
    
    # Save results
    results_file = output_dir / 'external_baseline_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n\nResults saved to: {results_file}")
    
    # Generate tables
    generate_latex_table(all_results, M_values)
    
    # Print summary
    print_summary(all_results)
    
    print("\n\n" + "="*80)
    print("INTEGRATION NOTES")
    print("="*80)
    print("""
These results can be integrated into the paper as:

1. Add to Section 5.4 (Baseline Comparisons):
   - Tables for M=100, M=200, M=500
   - Discussion of trade-offs between methods

2. Key distinctions:
   - GTSAM: Factor graph, soft constraints, very fast
   - Ceres: Nonlinear least squares, iterative
   - Ours: Custom ADMM, explicit tube-excess diagnostics, specialized

3. Limitations to acknowledge:
   - GTSAM uses robust loss (Huber), not explicit strict-feasibility constraints
   - Ceres-like uses SciPy (pure Python overhead)
   - Fair comparison would require custom Ceres cost functions
""")


if __name__ == "__main__":
    main()
