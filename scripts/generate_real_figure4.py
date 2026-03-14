"""Generate Figure 4 with REAL ADMM traces (not mock data).

This replaces the synthetic exponential decay curves in the original
figure with actual ADMM solver residuals from a real run.
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Set publication-quality styling
rcParams['font.size'] = 15
rcParams['font.family'] = 'Times New Roman'
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['axes.linewidth'] = 1.5
rcParams['lines.linewidth'] = 2
rcParams['grid.linewidth'] = 0.5

# Color palette
COLORS = {
    'blue': '#2E86C1',
    'red': '#FF6B6B',
    'green': '#00C853',
    'orange': '#F39C12',
    'purple': '#8E44AD',
    'gray': '#7F8C8D',
}

from smoother_fast import tube_smooth_fast
from noise_models import set_bounded_noise
from so3 import exp_so3, log_so3


def generate_real_admm_data():
    """Run solver and capture real ADMM traces."""
    print("Running solver to capture real ADMM traces...")
    
    # Generate test problem
    np.random.seed(42)
    M = 50
    t = np.arange(M) * 0.1
    phi = np.array([[0.1 * np.sin(0.8 * t[i]), 0.0, 0.0] for i in range(M)])
    R_gt = np.stack([exp_so3(phi[i]) for i in range(M)], axis=0)
    R_meas, eps = set_bounded_noise(R_gt, noise_sigma=0.05, eps_factor=3.0, seed=42)
    
    # Run solver
    R_hat, info = tube_smooth_fast(
        R_meas, eps, 1.0, 0.1, 0.1,
        max_outer=3,  # Just need first few outer iterations
        Delta=0.2,
        rho=1.0,
        inner_max_iter=1000,
        tol_outer=1e-6,
        tol_inner=1e-4,
    )
    
    print(f"Completed {info['outer_iter']} outer iterations")
    
    # Extract inner ADMM stats from first outer iteration
    inner_stats = info['inner_stats'][0]
    
    real_data = {
        'primal_residual': inner_stats['primal_residual'],
        'dual_residual': inner_stats['dual_residual'],
        'iterations': len(inner_stats['primal_residual']),
        'solver_type': inner_stats.get('solver', 'unknown'),
        'total_time': info['elapsed_sec'],
    }
    
    return real_data


def plot_real_admm_updates(real_data, output_path='docs/paper/figures/fig4_admm_updates_real.png'):
    """Generate Figure 4 with real data."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    iterations = np.arange(1, real_data['iterations'] + 1)
    primal_res = np.array(real_data['primal_residual'])
    dual_res = np.array(real_data['dual_residual'])
    
    # Subplot 1: Primal residual (δ-update convergence)
    ax = axes[0]
    ax.semilogy(iterations, primal_res, 'o-', color=COLORS['blue'],
                markersize=8, linewidth=2.5, label='Primal residual')
    ax.set_xlabel('ADMM Iteration', fontsize=15)
    ax.set_ylabel('Residual Norm (log scale)', fontsize=15)
    ax.set_title('(a) Primal Residual Convergence', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=12)
    
    # Subplot 2: Dual residual
    ax = axes[1]
    ax.semilogy(iterations, dual_res, 's-', color=COLORS['red'],
                markersize=8, linewidth=2.5, label='Dual residual')
    ax.set_xlabel('ADMM Iteration', fontsize=15)
    ax.set_ylabel('Residual Norm (log scale)', fontsize=15)
    ax.set_title('(b) Dual Residual Convergence', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=12)
    
    # Subplot 3: Both residuals together
    ax = axes[2]
    ax.semilogy(iterations, primal_res, 'o-', color=COLORS['blue'],
                markersize=6, linewidth=2, label='Primal')
    ax.semilogy(iterations, dual_res, 's-', color=COLORS['red'],
                markersize=6, linewidth=2, label='Dual')
    ax.axhline(y=1e-4, color=COLORS['gray'], linestyle='--', 
               linewidth=1.5, alpha=0.7, label='Tolerance')
    ax.set_xlabel('ADMM Iteration', fontsize=15)
    ax.set_ylabel('Residual Norm (log scale)', fontsize=15)
    ax.set_title('(c) Convergence to Tolerance', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"\nFigure saved to: {output_path}")
    print(f"  - Solver: {real_data['solver_type']}")
    print(f"  - Total ADMM iterations: {real_data['iterations']}")
    print(f"  - Final primal residual: {primal_res[-1]:.2e}")
    print(f"  - Final dual residual: {dual_res[-1]:.2e}")
    print(f"  - Total solver time: {real_data['total_time']:.3f}s")


def main():
    print("=" * 80)
    print("GENERATING REAL FIGURE 4")
    print("=" * 80)
    print("\nThis replaces the mock data in the original figure with actual")
    print("ADMM solver traces from a real smoothing problem.\n")
    
    # Generate real data
    real_data = generate_real_admm_data()
    
    # Create figure
    plot_real_admm_updates(real_data)
    
    # Also save the data for reproducibility
    import json
    from pathlib import Path
    
    output_dir = Path('results/real_figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data_file = output_dir / 'fig4_real_data.json'
    with open(data_file, 'w') as f:
        json.dump(real_data, f, indent=2)
    print(f"\nData saved to: {data_file}")
    
    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("1. Update the paper to use fig4_admm_updates_real.png")
    print("2. Add caption noting 'Real ADMM convergence traces from M=50 problem'")
    print("3. Archive the old mock figure")


if __name__ == "__main__":
    main()
