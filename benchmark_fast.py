import time
import numpy as np
import matplotlib.pyplot as plt
from so3 import exp_so3, log_so3, batch_log, geodesic_angle
from smoother_fast import tube_smooth_fast
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


def benchmark_performance(M_list, params):
    """Benchmark performance across different problem sizes."""
    results = []

    for M in M_list:
        print(f"\n===== Benchmarking M={M} =====")

        # Generate data
        R_true, R_meas, eps = generate_synthetic_data(M)

        # Parameters
        lam, mu, tau = params['lam'], params['mu'], params['tau']

        # Warm-up run
        _, _ = tube_smooth_fast(R_meas, eps, lam, mu, tau, max_outer=1)

        # Timing run
        start = time.perf_counter()
        R_hat, info = tube_smooth_fast(
            R_meas, eps, lam, mu, tau,
            max_outer=params['max_outer'],
            Delta=params['Delta'],
            rho=params['rho'],
            inner_max_iter=params['inner_max_iter'],
            tol_outer=params['tol_outer'],
            tol_inner=params['tol_inner']
        )
        elapsed = time.perf_counter() - start

        # Compute metrics
        metrics = compute_metrics(R_true, R_meas, R_hat, eps)

        # Print results
        print(f"Total time: {elapsed:.3f}s")
        print(f"Outer iterations: {info['outer_iter']}")
        print(f"Avg inner iterations: {np.mean([s['iter'] for s in info['inner_stats']]):.1f}")
        print(f"Max constraint violation: {metrics['max_violation']:.4f}")
        print(f"Velocity RMS: {metrics['vel_rms']:.4f}, Acceleration RMS: {metrics['acc_rms']:.4f}")

        results.append({
            'M': M,
            'total_time': elapsed,
            'outer_iter': info['outer_iter'],
            'avg_inner_iter': np.mean([s['iter'] for s in info['inner_stats']]),
            'max_violation': metrics['max_violation'],
            'vel_rms': metrics['vel_rms'],
            'acc_rms': metrics['acc_rms'],
            'converged': info['converged']
        })

        # For small M, verify against CVXPY (if available)
        if M <= 200:
            try:
                from smoother_fast import solve_inner_with_cvxpy_reference
                print("\nVerifying against CVXPY reference...")

                # Build problem for first outer iteration
                phi_k = batch_log(R_meas)
                r_list = []
                J_list = []
                for j in range(M):
                    r_j = log_so3(exp_so3(-batch_log([R_meas[j]])[0]) @ exp_so3(phi_k[j]))
                    J_j = np.eye(3)  # Simplified for reference
                    r_list.append(r_j)
                    J_list.append(J_j)

                H = build_H(M, lam, mu, tau)
                g = H @ batch_log(R_meas).flatten()

                delta_admm, _ = admm_solver.solve_inner_admm(H, g, r_list, J_list, eps, params['Delta'])
                delta_cvx, cvx_info = solve_inner_with_cvxpy_reference(H, g, r_list, J_list, eps, params['Delta'])

                err = np.linalg.norm(delta_admm - delta_cvx)
                obj_diff = abs(0.5*delta_admm@H@delta_admm + g@delta_admm - cvx_info['obj'])


                print(f"ADMM vs CVXPY: delta error={err:.2e}, objective diff={obj_diff:.2e}")
            except ImportError:
                print("CVXPY not available for verification")

    return results


def plot_results(results):
    """Plot benchmark results."""
    M_list = [r['M'] for r in results]

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.loglog(M_list, [r['total_time'] for r in results], 'o-')
    plt.xlabel('Problem size M')
    plt.ylabel('Total time (s)')
    plt.title('Runtime vs Problem Size')
    plt.grid(True, which="both", ls="-")

    plt.subplot(2, 2, 2)
    plt.plot(M_list, [r['avg_inner_iter'] for r in results], 'o-')
    plt.xlabel('Problem size M')
    plt.ylabel('Avg inner iterations')
    plt.title('ADMM Iterations')
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.semilogx(M_list, [r['max_violation'] for r in results], 'o-')
    plt.xlabel('Problem size M')
    plt.ylabel('Max constraint violation')
    plt.title('Constraint Satisfaction')
    plt.grid(True, which="both", ls="-")

    plt.subplot(2, 2, 4)
    plt.semilogx(M_list, [r['acc_rms'] for r in results], 'o-')
    plt.xlabel('Problem size M')
    plt.ylabel('Acceleration RMS')
    plt.title('Smoothness Metric')
    plt.grid(True, which="both", ls="-")

    plt.tight_layout()
    plt.savefig('benchmark_results.png')
    print("\nBenchmark plot saved to 'benchmark_results.png'")

if __name__ == "__main__":
    # Configuration
    params = {
        'lam': 1.0,
        'mu': 0.1,
        'tau': 0.1,
        'max_outer': 20,
        'Delta': 0.2,
        'rho': 1.0,
        'inner_max_iter': 2000,
        'tol_outer': 1e-6,
        'tol_inner': 1e-4
    }

    # Run benchmark for different problem sizes
    M_list = [100, 1000, 5000, 10000]  # Extend to 50000 if needed
    results = benchmark_performance(M_list, params)

    # Plot results
    plot_results(results)

    # Print summary
    print("\n===== BENCHMARK SUMMARY =====")
    for r in results:
        print(f"M={r['M']}: time={r['total_time']:.2f}s, outer={r['outer_iter']}, inner={r['avg_inner_iter']:.1f}, violation={r['max_violation']:.4f}")