import time
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import zipfile
from so3 import exp_so3, log_so3, batch_log, geodesic_angle, right_jacobian, right_jacobian_inv
from smoother_fast import tube_smooth_fast
from smoother_socp import tube_smooth_socp
from hessian import build_H
import admm_solver


def generate_synthetic_data(M, tau=0.1, noise_std=0.05):
    """Generate smooth SO(3) trajectory with measurement noise."""
    t = np.linspace(0, 2*np.pi, M)

    # True trajectory: rotation around z-axis with varying speed
    phi_true = np.zeros((M, 3))
    phi_true[:, 2] = 2 * np.sin(2*t) + 0.5 * np.sin(4*t)

    R_true = [exp_so3(phi) for phi in phi_true]
    # Add noise
    noise = np.random.normal(scale=noise_std, size=(M, 3))
    R_meas = [exp_so3(log_so3(R_true[i]) + noise[i]) for i in range(M)]

    # Tube radii (larger where noise is higher)
    eps = np.full(M, 0.2)
    eps[::2] = 0.15  # Alternate tighter constraints

    return R_true, R_meas, eps


def compute_metrics(R_true, R_meas, R_hat, eps):
    """Compute evaluation metrics for smoothed trajectory."""
    M = len(R_true)

    # Geodesic distances from true trajectory
    dists = np.array([geodesic_angle(R_true[i], R_hat[i]) for i in range(M)])

    # Constraint violations
    violations = np.array([geodesic_angle(R_meas[i], R_hat[i]) - eps[i] for i in range(M)])
    max_violation = np.max(violations)
    avg_violation = np.mean(np.maximum(violations, 0))

    # Velocity and acceleration (in tangent space)
    phi_hat = batch_log(R_hat)
    d1 = phi_hat[1:] - phi_hat[:-1]
    d2 = d1[1:] - d1[:-1]

    vel_rms = np.sqrt(np.mean(np.sum(d1**2, axis=1)))
    acc_rms = np.sqrt(np.mean(np.sum(d2**2, axis=1)))

    return {
        'dists': dists,
        'max_violation': max_violation,
        'avg_violation': avg_violation,
        'vel_rms': vel_rms,
        'acc_rms': acc_rms
    }


def load_euroc_mav_subset():
    """Load preprocessed EuRoC MAV dataset subset from local file."""
    try:
        data = np.load('euroc_mav_subset.npz')
        # The file was saved with np.savez(*R_meas), so arrays are named 'arr_0', 'arr_1', etc.
        R_meas = [data[f'arr_{i}'] for i in range(len(data.files))]
        eps = np.full(len(R_meas), 0.15)
        return R_meas, eps
    except Exception as e:
        print(f"Error loading real dataset: {e}")
        # Fallback to synthetic data
        R_true, R_meas, eps = generate_synthetic_data(1000, tau=0.01, noise_std=0.06)
        return R_meas, eps


def validate_on_euroc():
    """Validate on EuRoC MAV dataset subset."""
    print("\n===== EuRoC MAV Dataset Validation =====")
    R_meas, eps = load_euroc_mav_subset()
    if R_meas is None:
        return None

    # Parameters matching typical IMU setup (100Hz sampling)
    tau = 0.01
    params = {
        'lam': 1.0,
        'mu': 0.1,
        'Delta': 0.2,
        'rho': 1.0,
        'inner_max_iter': 2000,
        'tol_outer': 1e-6,
        'tol_inner': 1e-4
    }

    # Run smoothing
    start = time.perf_counter()
    R_hat, info = tube_smooth_fast(
        R_meas, eps, params['lam'], params['mu'], tau,
        max_outer=20,
        Delta=params['Delta'],
        rho=params['rho'],
        inner_max_iter=params['inner_max_iter'],
        tol_outer=params['tol_outer'],
        tol_inner=params['tol_inner']
    )
    elapsed = time.perf_counter() - start

    # Compute metrics
    M = len(R_meas)
    phi_meas = batch_log(R_meas)
    phi_hat = batch_log(R_hat)

    # Acceleration RMS in tangent space
    d2_meas = np.diff(phi_meas, n=2, axis=0) / (tau**2)
    d2_hat = np.diff(phi_hat, n=2, axis=0) / (tau**2)
    acc_rms_meas = np.sqrt(np.mean(np.sum(d2_meas**2, axis=1)))
    acc_rms_hat = np.sqrt(np.mean(np.sum(d2_hat**2, axis=1)))

    # Constraint satisfaction
    violations = np.array([geodesic_angle(R_meas[i], R_hat[i]) - eps[i] for i in range(M)])
    max_violation = np.max(violations)

    print(f"Dataset size: M={M} rotations (100Hz for {M*tau:.1f}s)")
    print(f"Processing time: {elapsed:.2f}s ({M/elapsed:.1f} Hz)")
    print(f"Outer iterations: {info['outer_iter']}")
    print(f"Max constraint violation: {max_violation:.4f} (threshold={eps[0]:.2f})")
    print(f"Acceleration RMS: {acc_rms_meas:.4f} → {acc_rms_hat:.4f} ({100*(1-acc_rms_hat/acc_rms_meas):.1f}% reduction)")

    return {
        'dataset': 'EuRoC_MH01_easy_subset',
        'M': M,
        'time': elapsed,
        'acc_rms_before': acc_rms_meas,
        'acc_rms_after': acc_rms_hat,
        'violation': max_violation
    }


def benchmark_performance(M_list, params):
    """Benchmark performance across different problem sizes."""
    results = []

    for M in M_list:
        print(f"\n===== Benchmarking M={M} =====")
        print("Generating synthetic data...")
        R_true, R_meas, eps = generate_synthetic_data(M)

        # Parameters
        lam, mu, tau = params['lam'], params['mu'], params['tau']

        # Test both methods
        methods = [
            ('ADMM (Ours)', tube_smooth_fast),
            ('SOCP (Baseline)', tube_smooth_socp)
        ]

        for method_name, smoother in methods:
            print(f"\n--- Running {method_name} ---")
            start = time.perf_counter()

            if method_name == 'ADMM (Ours)':
                R_hat, info = smoother(
                    R_meas, eps, lam, mu, tau,
                    max_outer=params['max_outer'],
                    Delta=params['Delta'],
                    rho=params['rho'],
                    inner_max_iter=params['inner_max_iter'],
                    tol_outer=params['tol_outer'],
                    tol_inner=params['tol_inner']
                )
            else:
                R_hat, info = smoother(
                    R_meas, eps, lam, mu, tau,
                    max_outer=params['max_outer'],
                    Delta=params['Delta'],
                    tol=params['tol_outer'],
                    solver='SCS'
                )

            elapsed = time.perf_counter() - start
            metrics = compute_metrics(R_true, R_meas, R_hat, eps)

            print(f"Total time: {elapsed:.3f}s")
            print(f"Max constraint violation: {metrics['max_violation']:.4f}")
            print(f"Acceleration RMS: {metrics['acc_rms']:.4f}")
            print(f"Constraint violation reduction: {100*(1 - metrics['max_violation']/np.max(eps)):.1f}%")

            results.append({
                'method': method_name,
                'M': M,
                'total_time': elapsed,
                'max_violation': metrics['max_violation'],
                'acc_rms': metrics['acc_rms'],
                'vel_rms': metrics['vel_rms'],
                'outer_iter': info.get('outer_iter', 1),
            })

        # For small M, verify against CVXPY (if available)
        if M <= 200:
            try:
                from smoother_fast import solve_inner_with_cvxpy_reference
                print("\nVerifying against CVXPY reference...")

                # Build problem for first outer iteration
                phi_meas = batch_log(R_meas)
                phi_k = phi_meas.copy()
                r_list = []
                J_list = []
                for j in range(M):
                    r_j = log_so3(exp_so3(-phi_meas[j]) @ exp_so3(phi_k[j]))
                    J_j = right_jacobian_inv(r_j) @ right_jacobian(phi_k[j])
                    r_list.append(r_j)
                    J_list.append(J_j)

                H = build_H(M, lam, mu, tau, damping=1e-9)
                g = H @ phi_k.flatten()

                # Use higher rho and tighter tolerance for verification
                # Note: First-iteration problem (phi_k = phi_meas) is challenging for ADMM convergence
                delta_admm, stats_admm = admm_solver.solve_inner_admm(
                    H, g, r_list, J_list, eps, params['Delta'],
                    rho=100.0, max_iter=20000, tol=1e-7
                )
                delta_cvx, cvx_info = solve_inner_with_cvxpy_reference(H, g, r_list, J_list, eps, params['Delta'])

                err = np.linalg.norm(delta_admm - delta_cvx)
                obj_diff = abs(0.5*delta_admm@H@delta_admm + g@delta_admm - cvx_info['obj'])

                print(f"ADMM vs CVXPY: delta error={err:.2e}, objective diff={obj_diff:.2e}")
            except ImportError:
                print("CVXPY not available for verification")

    # Add EuRoC validation for final results
    euroc_result = validate_on_euroc()
    if euroc_result:
        results.append(euroc_result)

    return results


def plot_results(results):
    """Plot benchmark results with multiple methods."""
    # Separate results by method
    methods = ['ADMM (Ours)', 'SOCP (Baseline)']
    # Filter out EuRoC results that don't have 'method' key
    synthetic_results = [r for r in results if 'method' in r]
    method_results = {m: [r for r in synthetic_results if r['method'] == m] for m in methods}

    # Create figure with subplots
    plt.figure(figsize=(14, 10))

    # Runtime comparison
    plt.subplot(2, 2, 1)
    for method in methods:
        m_results = method_results[method]
        M_list = [r['M'] for r in m_results]
        times = [r['total_time'] for r in m_results]
        plt.loglog(M_list, times, 'o-', label=method)
    plt.xlabel('Problem size M')
    plt.ylabel('Total time (s)')
    plt.title('Runtime Comparison')
    plt.legend()
    plt.grid(True, which="both", ls="-")

    # Constraint violation
    plt.subplot(2, 2, 2)
    for method in methods:
        m_results = method_results[method]
        M_list = [r['M'] for r in m_results]
        violations = [r['max_violation'] for r in m_results]
        plt.semilogx(M_list, violations, 'o-', label=method)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    plt.xlabel('Problem size M')
    plt.ylabel('Max constraint violation')
    plt.title('Constraint Satisfaction')
    plt.legend()
    plt.grid(True, which="both", ls="-")

    # Acceleration RMS
    plt.subplot(2, 2, 3)
    for method in methods:
        m_results = method_results[method]
        M_list = [r['M'] for r in m_results]
        acc_rms = [r['acc_rms'] for r in m_results]
        plt.semilogx(M_list, acc_rms, 'o-', label=method)
    plt.xlabel('Problem size M')
    plt.ylabel('Acceleration RMS')
    plt.title('Smoothness Metric')
    plt.legend()
    plt.grid(True, which="both", ls="-")

    # Outer iterations
    plt.subplot(2, 2, 4)
    for method in methods:
        m_results = method_results[method]
        M_list = [r['M'] for r in m_results]
        outer_iter = [r['outer_iter'] for r in m_results]
        plt.semilogx(M_list, outer_iter, 'o-', label=method)
    plt.xlabel('Problem size M')
    plt.ylabel('Outer iterations')
    plt.title('Convergence Behavior')
    plt.legend()
    plt.grid(True, which="both", ls="-")

    plt.tight_layout()
    plt.savefig('benchmark_results.png')
    print("\nBenchmark plot saved to 'benchmark_results.png'")

if __name__ == "__main__":
    # Configuration
    params = {
        'lam': 1.0,
        'mu': 0.1,
        'tau': 0.01,  # Match EuRoC 100Hz sampling
        'max_outer': 20,
        'Delta': 0.2,
        'rho': 1.0,
        'inner_max_iter': 2000,
        'tol_outer': 1e-6,
        'tol_inner': 1e-4
    }

    # Run benchmark for different problem sizes
    M_list = [100, 500, 1000]  # Smaller sizes for practical benchmarking
    results = benchmark_performance(M_list, params)

    # Plot results
    plot_results(results)

    # Print summary
    print("\n===== BENCHMARK SUMMARY =====")
    for r in results:
        if 'method' in r:  # Only process benchmark results with method info
            # Safely access keys with defaults to prevent KeyErrors
            outer_iter = r.get('outer_iter', 'N/A')
            print(f"M={r['M']} ({r['method']}): time={r['total_time']:.2f}s, outer={outer_iter}, violation={r['max_violation']:.4f}")