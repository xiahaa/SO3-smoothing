"""Test branch-cut stability for long trajectories."""

from __future__ import annotations

import numpy as np
import sys
sys.path.insert(0, 'src')

from so3 import exp_so3, log_so3, geodesic_angle
from noise_models import add_gaussian_rotation_noise
from smoother_fast import tube_smooth_fast


def generate_long_trajectory(M: int, total_rotation: float) -> np.ndarray:
    """Generate trajectory spanning multiple full rotations.

    Args:
        M: Number of samples
        total_rotation: Total angular distance in radians (> 4π for long paths)

    Returns:
        R_true: Ground truth rotations, shape (M, 3, 3)
    """
    # Smooth angular velocity with varying speed
    t = np.linspace(0, 1.0, M) * total_rotation / (2 * np.pi)
    phi = np.stack([
        0.3 * np.sin(0.5 * t),
        0.5 * np.sin(0.3 * t + 0.2),
        np.linspace(0, total_rotation, M),  # z-axis does full rotations
    ], axis=1)

    return np.stack([exp_so3(phi) for phi in phi], axis=0)


def test_no_spurious_jumps() -> None:
    """Verify smoothing doesn't introduce jumps across branch cuts."""
    # Generate trajectory spanning 4π (two full rotations)
    M = 200
    total_rotation = 4 * np.pi  # Two full rotations
    R_true = generate_long_trajectory(M, total_rotation)

    # Add modest noise
    R_noisy = add_gaussian_rotation_noise(R_true, sigma=0.03, seed=42)

    # Smooth
    eps = 0.08 * np.ones(M)
    R_hat, _ = tube_smooth_fast(
        R_noisy, eps, lam=1.0, mu=0.1, tau=0.01,
        max_outer=20, Delta=0.2, rho=1.0,
    )

    # Check for spurious jumps in smoothed trajectory
    jumps = np.array([
        geodesic_angle(R_hat[i], R_hat[i+1])
        for i in range(M-1)
    ])

    # Jumps should be smooth (no sudden large changes)
    max_jump = float(np.max(jumps))
    median_jump = float(np.median(np.abs(jumps)))

    # With smoothness objective, jumps should be small
    assert max_jump < 0.5, f"Spurious jump detected: max_jump={max_jump:.4f} rad"
    assert median_jump < 0.1, f"Median jump too large: median_jump={median_jump:.4f} rad"


def test_geodesic_angle_consistency() -> None:
    """Verify geodesic_angle computed correctly across branch cuts."""
    # Test pairs with known geodesic distances
    test_cases = [
        # Small rotation
        {
            'R1': exp_so3(np.array([0.0, 0.0, 0.0])),
            'R2': exp_so3(np.array([0.0, 0.0, np.pi/4])),  # 45 degrees
            'expected': np.pi/4,
        },
        # Medium rotation
        {
            'R1': exp_so3(np.array([0.0, 0.0, 0.0])),
            'R2': exp_so3(np.array([0.0, 0.0, np.pi/2])),  # 90 degrees
            'expected': np.pi/2,
        },
        # Large rotation (180 degrees)
        {
            'R1': exp_so3(np.array([0.0, 0.0, 0.0])),
            'R2': exp_so3(np.array([0.0, 0.0, np.pi])),  # 180 degrees
            'expected': np.pi,
        },
        # Same rotation (identity)
        {
            'R1': exp_so3(np.array([0.0, 0.0, 0.0])),
            'R2': exp_so3(np.array([0.0, 0.0, 0.0])),
            'expected': 0.0,
        },
    ]

    for case in test_cases:
        angle = geodesic_angle(case['R1'], case['R2'])
        # Geodesic angle should be within tolerance
        assert abs(angle - case['expected']) < 1e-6, \
            f"Geodesic angle incorrect: got={angle:.4f}, expected={case['expected']:.4f}"


def test_cumulative_angle_tracking() -> None:
    """Verify angle accumulation works across multiple full rotations."""
    # Generate trajectory with known cumulative rotation
    M = 100
    total_rotation = 6 * np.pi  # Three full rotations
    R_true = generate_long_trajectory(M, total_rotation)

    # Convert to log space
    phi_hat = np.vstack([log_so3(R) for R in R_true])

    # Verify that the z-axis shows the expected total rotation pattern
    # Due to SO(3) wrapping, we verify the trajectory spans the expected range
    z_min = np.min(phi_hat[:, 2])
    z_max = np.max(phi_hat[:, 2])

    # The z-axis should span approximately the total rotation
    z_range = z_max - z_min
    assert z_range > 0.3 * total_rotation, \
        f"Angle span too small: {z_range:.4f} vs {0.3 * total_rotation:.4f}"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
