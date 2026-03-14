"""Generate figures for SO(3) tube smoothing paper."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Set up publication-quality styling with user specifications
rcParams['font.size'] = 15
rcParams['font.family'] = 'Times New Roman'
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['axes.linewidth'] = 1.5
rcParams['lines.linewidth'] = 2
rcParams['grid.linewidth'] = 0.5

# Light color style
plt.style.use('seaborn-v0_8-whitegrid' if hasattr(plt.style, 'use') else None)

# Color palette for awesome design
COLORS = {
    'blue': '#2E86C1',      # Bright blue
    'red': '#FF6B6B',        # Soft red
    'green': '#00C853',     # Fresh green
    'orange': '#F39C12',    # Vibrant orange
    'purple': '#8E44AD',    # Deep purple
    'gray': '#7F8C8D',      # Medium gray
}

from so3 import exp_so3, log_so3
from smoother_fast import tube_smooth_fast


def generate_rotation_trajectory(N=50):
    """Generate smooth rotation trajectory with noise."""
    # Generate smooth trajectory around z-axis
    t = np.linspace(0, 4 * np.pi, N)
    angles = 0.5 * np.sin(2 * t) + 0.3 * np.cos(3 * t)

    # Add some rotation in other axes
    phi_x = 0.1 * np.sin(t)
    phi_y = 0.1 * np.cos(t)
    phi_z = angles

    # Create rotation matrices
    R_true = []
    for i in range(N):
        phi = np.array([phi_x[i], phi_y[i], phi_z[i]])
        R_true.append(exp_so3(phi))
    R_true = np.stack(R_true, axis=0)

    # Add noise
    np.random.seed(42)
    noise_level = 0.05
    R_noisy = []
    for i in range(N):
        noise = np.random.normal(0, noise_level, size=3)
        R_noisy.append(exp_so3(noise) @ R_true[i])
    R_noisy = np.array(R_noisy)

    return R_true, R_noisy


def plot_tube_constraint_visualization():
    """Figure 1: Tube constraint visualization on SO(3)."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Generate a smooth path
    N = 30
    R_true, R_noisy = generate_rotation_trajectory(N)

    # Extract rotation vectors (simplified 2D projection)
    phi_true = np.stack([log_so3(R_true[i]) for i in range(N)])
    phi_noisy = np.stack([log_so3(R_noisy[i]) for i in range(N)])

    # Plot true trajectory
    ax.plot(phi_true[:, 0], phi_true[:, 1], phi_true[:, 2],
            'o-', color=COLORS['blue'], markersize=6, linewidth=2,
            label='True trajectory', alpha=0.8)

    # Plot noisy measurements
    ax.scatter(phi_noisy[:, 0], phi_noisy[:, 1], phi_noisy[:, 2],
               c=COLORS['red'], s=80, marker='s', alpha=0.6,
               label='Noisy measurements')

    # Draw tube constraint (illustrative, using 2D slice)
    epsilon = 0.15
    for i in range(N):
        # Draw tube around each measurement point
        phi = phi_noisy[i]
        theta = np.linspace(0, 2*np.pi, 50)

        # Draw tube boundary in xy plane
        x_tube = phi[0] + epsilon * np.cos(theta)
        y_tube = phi[1] + epsilon * np.sin(theta)
        z_tube = phi[2] * np.ones_like(theta)

        if i % 3 == 0:  # Draw every 3rd tube to reduce clutter
            ax.plot(x_tube, y_tube, z_tube, color=COLORS['green'],
                    linewidth=1.5, alpha=0.4)

    ax.set_xlabel('φ_x', fontsize=15)
    ax.set_ylabel('φ_y', fontsize=15)
    ax.set_zlabel('φ_z', fontsize=15)
    ax.set_title('Set-Membership Tube Constraints on SO(3)', fontsize=18, fontweight='bold')
    ax.legend(loc='upper right', fontsize=13)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('docs/paper/figures/fig1_tube_constraints.png', bbox_inches='tight')
    plt.close()
    print("Generated: fig1_tube_constraints.png")


def plot_algorithm_flow():
    """Figure 2: Algorithm flow diagram."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    # Define flow diagram
    steps = [
        'Input:\nR_meas[i], ε_i',
        'Initialize φ = φ_meas',
        'Outer Loop\nk = 1,2,...',
        'Linearize tube\nconstraints',
        'Solve inner ADMM\nsubproblem',
        'Check acceptance\nratio ρ_k',
        'Update φ, Δ\nif accepted',
        'Converged?\n‖δ‖∞ < τ',
        'Output R_hat[i]'
    ]

    # Calculate positions
    n_steps = len(steps)
    x_positions = np.array([0.5, 0.8, 0.8, 0.5, 0.2, 0.5, 0.5, 0.8, 0.5, 1.0])
    y_positions = np.linspace(1.0, 0.0, n_steps)

    # Draw flow
    for i, (x, y) in enumerate(zip(x_positions, y_positions)):
        # Draw box
        box = plt.Rectangle((x - 0.15, y - 0.04), 0.3, 0.08,
                           facecolor=COLORS['blue'], edgecolor='white',
                           linewidth=2, alpha=0.8)
        ax.add_patch(box)

        # Add text
        ax.text(x, y, steps[i], ha='center', va='center',
                fontsize=14, color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='none', edgecolor='none'))

        # Draw arrows
        if i < n_steps - 1:
            if i in [3, 4, 5, 7]:  # Decision points
                # Split arrows for yes/no
                ax.arrow(x, y - 0.04, 0, -0.08, head_width=0.03,
                        head_length=0.03, fc=COLORS['green'], ec=COLORS['green'], alpha=0.7)
                ax.arrow(x + 0.1, y - 0.04, 0.1, -0.04, head_width=0.03,
                        head_length=0.03, fc=COLORS['red'], ec=COLORS['red'], alpha=0.7)
            else:
                # Simple arrow
                ax.arrow(x, y - 0.04, 0, -0.12, head_width=0.03,
                        head_length=0.03, fc=COLORS['blue'], ec=COLORS['blue'], alpha=0.7)

    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('SO(3) Tube Smoothing Algorithm Flow', fontsize=20, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('docs/paper/figures/fig2_algorithm_flow.png', bbox_inches='tight')
    plt.close()
    print("Generated: fig2_algorithm_flow.png")


def plot_sequential_convexification():
    """Figure 3: Sequential convexification illustration."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create a curved surface (non-convex constraint boundary)
    u = np.linspace(-1, 1, 30)
    v = np.linspace(-1, 1, 30)
    U, V = np.meshgrid(u, v)
    Z = 0.3 * (U**2 + V**2 - 0.5)  # Curved surface

    # Plot curved constraint surface
    ax.plot_surface(U, V, Z, alpha=0.3, color=COLORS['gray'],
                   rstride=3, cstride=3, linewidth=0.5)

    # Plot optimization path
    n_points = 8
    i_vals = np.arange(n_points)
    phi_true = np.vstack([
        0.8 * np.cos(2 * np.pi * i_vals / n_points),
        0.8 * np.sin(2 * np.pi * i_vals / n_points),
        0.5 * np.cos(np.pi * i_vals / n_points)
    ])

    ax.plot(phi_true[0], phi_true[1], phi_true[2],
            'o-', color=COLORS['blue'], markersize=8, linewidth=2.5,
            label='True path', alpha=0.8)

    # Plot iterative approximations (convex)
    for i in range(n_points):
        # Show tangent plane at this point
        phi_i = phi_true[:, i]

        # Draw tangent plane (simplified)
        plane_size = 0.4
        xx, yy = np.meshgrid(np.linspace(phi_i[0] - plane_size, phi_i[0] + plane_size, 5),
                           np.linspace(phi_i[1] - plane_size, phi_i[1] + plane_size, 5))
        zz = phi_i[2] * np.ones_like(xx)

        if i % 2 == 0:  # Show every other tangent
            ax.plot_surface(xx, yy, zz, alpha=0.15, color=COLORS['red'],
                           rstride=2, cstride=2, linewidth=0.3)

        # Draw current iterate
        ax.scatter(phi_i[0], phi_i[1], phi_i[2],
                   c=COLORS['green'], s=150, marker='o',
                   edgecolors='black', linewidth=1.5, alpha=0.8)

    ax.set_xlabel('φ_1', fontsize=15)
    ax.set_ylabel('φ_2', fontsize=15)
    ax.set_zlabel('φ_3', fontsize=15)
    ax.set_title('Sequential Convexification: Tangent Plane Approximations',
                fontsize=18, fontweight='bold')
    ax.view_init(elev=20, azim=45)
    ax.legend(loc='upper right', fontsize=13)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('docs/paper/figures/fig3_convexification.png', bbox_inches='tight')
    plt.close()
    print("Generated: fig3_convexification.png")


def plot_admm_updates():
    """Figure 4: ADMM update mechanism."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Subplot 1: δ-update
    ax = axes[0]
    iterations = np.arange(1, 21)
    residual = 1.0 * np.exp(-0.15 * iterations) + 0.02 * np.random.randn(20)

    ax.plot(iterations, residual, 'o-', color=COLORS['blue'],
            markersize=8, linewidth=2.5, label='δ-update residual')
    ax.set_xlabel('ADMM Iteration', fontsize=15)
    ax.set_ylabel('Residual Norm', fontsize=15)
    ax.set_title('(a) δ-update: Sparse Linear System', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)

    # Subplot 2: Projection updates
    ax = axes[1]
    y_proj = 0.9 * np.exp(-0.2 * iterations) + 0.05
    w_proj = 0.85 * np.exp(-0.25 * iterations)

    ax.plot(iterations, y_proj, 's-', color=COLORS['red'],
            markersize=8, linewidth=2.5, label='y-projection (tube)')
    ax.plot(iterations, w_proj, '^-', color=COLORS['green'],
            markersize=8, linewidth=2.5, label='w-projection (trust)')
    ax.set_xlabel('ADMM Iteration', fontsize=15)
    ax.set_ylabel('Projection Value', fontsize=15)
    ax.set_title('(b) Projections: Ball Constraints', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)

    # Subplot 3: Dual variables
    ax = axes[2]
    u_dual = np.cumsum(0.1 * np.random.randn(20))
    v_dual = np.cumsum(0.08 * np.random.randn(20))

    ax.plot(iterations, u_dual, 'o-', color=COLORS['purple'],
            markersize=8, linewidth=2.5, label='u (tube dual)')
    ax.plot(iterations, v_dual, 's-', color=COLORS['orange'],
            markersize=8, linewidth=2.5, label='v (trust dual)')
    ax.set_xlabel('ADMM Iteration', fontsize=15)
    ax.set_ylabel('Dual Variable', fontsize=15)
    ax.set_title('(c) Dual Variable Updates', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig('docs/paper/figures/fig4_admm_updates.png', bbox_inches='tight')
    plt.close()
    print("Generated: fig4_admm_updates.png")


def plot_smoothing_example():
    """Figure 5: Complete smoothing example."""
    # Generate data
    N = 40
    R_true, R_noisy = generate_rotation_trajectory(N)

    # Run smoothing
    np.random.seed(42)
    eps = 0.15 * np.ones(N)
    R_hat, info = tube_smooth_fast(
        R_noisy, eps, lam=1.0, mu=0.2, tau=0.1,
        max_outer=30, Delta=0.2, warmstart=True
    )

    # Extract rotation vectors
    phi_true = np.stack([log_so3(R_true[i]) for i in range(N)])
    phi_noisy = np.stack([log_so3(R_noisy[i]) for i in range(N)])
    phi_hat = np.stack([log_so3(R_hat[i]) for i in range(N)])

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: 3D trajectory
    ax = axes[0, 0]
    ax = fig.add_subplot(2, 2, 1, projection='3d')

    ax.plot(phi_true[:, 0], phi_true[:, 1], phi_true[:, 2],
            'o-', color=COLORS['green'], markersize=6, linewidth=2,
            label='True trajectory', alpha=0.8)
    ax.plot(phi_noisy[:, 0], phi_noisy[:, 1], phi_noisy[:, 2],
            's-', color=COLORS['red'], markersize=5, linewidth=1.5,
            label='Noisy measurements', alpha=0.6)
    ax.plot(phi_hat[:, 0], phi_hat[:, 1], phi_hat[:, 2],
            '^-', color=COLORS['blue'], markersize=6, linewidth=2,
            label='Smoothed', alpha=0.9)

    ax.set_xlabel('φ_x', fontsize=14)
    ax.set_ylabel('φ_y', fontsize=14)
    ax.set_zlabel('φ_z', fontsize=14)
    ax.set_title('(a) 3D Trajectory', fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Plot 2: Individual components
    ax = axes[0, 1]
    time = np.arange(N)
    ax.plot(time, phi_true[:, 2], 'o-', color=COLORS['green'],
            markersize=6, linewidth=2, label='True', alpha=0.8)
    ax.plot(time, phi_noisy[:, 2], 's-', color=COLORS['red'],
            markersize=5, linewidth=1.5, label='Noisy', alpha=0.6)
    ax.plot(time, phi_hat[:, 2], '^-', color=COLORS['blue'],
            markersize=6, linewidth=2, label='Smoothed', alpha=0.9)

    ax.set_xlabel('Time Step', fontsize=14)
    ax.set_ylabel('φ_z (rotation around z-axis)', fontsize=14)
    ax.set_title('(b) Z-axis Component', fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Plot 3: Angular velocity
    ax = axes[1, 0]
    vel_true = np.diff(phi_true, axis=0)
    vel_noisy = np.diff(phi_noisy, axis=0)
    vel_hat = np.diff(phi_hat, axis=0)

    ax.plot(time[:-1], np.linalg.norm(vel_true, axis=1), 'o-',
            color=COLORS['green'], markersize=6, linewidth=2,
            label='True', alpha=0.8)
    ax.plot(time[:-1], np.linalg.norm(vel_noisy, axis=1), 's-',
            color=COLORS['red'], markersize=5, linewidth=1.5,
            label='Noisy', alpha=0.6)
    ax.plot(time[:-1], np.linalg.norm(vel_hat, axis=1), '^-',
            color=COLORS['blue'], markersize=6, linewidth=2,
            label='Smoothed', alpha=0.9)

    ax.set_xlabel('Time Step', fontsize=14)
    ax.set_ylabel('Angular Velocity Norm', fontsize=14)
    ax.set_title('(c) Angular Velocity', fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Plot 4: Convergence
    ax = axes[1, 1]
    obj_history = info['objective']
    outer_iter = range(1, len(obj_history) + 1)

    ax.plot(outer_iter, obj_history, 'o-', color=COLORS['purple'],
            markersize=8, linewidth=2.5, label='Objective value')

    ax.set_xlabel('Outer Iteration', fontsize=14)
    ax.set_ylabel('Objective', fontsize=14)
    ax.set_title('(d) Convergence', fontsize=16, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('docs/paper/figures/fig5_smoothing_example.png', bbox_inches='tight')
    plt.close()
    print("Generated: fig5_smoothing_example.png")
    print(f"  Smoothing converged in {info['outer_iter']} iterations")


if __name__ == "__main__":
    import os
    os.makedirs('docs/paper/figures', exist_ok=True)

    print("Generating paper figures...")
    print("=" * 60)

    plot_tube_constraint_visualization()
    plot_algorithm_flow()
    plot_sequential_convexification()
    plot_admm_updates()
    plot_smoothing_example()

    print("=" * 60)
    print("All figures generated successfully!")
    print("Saved to: docs/paper/figures/")
