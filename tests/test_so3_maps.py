"""SO(3) exponential/logarithmic map tests and branch-cut stability."""

from __future__ import annotations

import numpy as np
import sys
sys.path.insert(0, 'src')

from so3 import exp_so3, log_so3, right_jacobian


def _rand_phi(n: int, scale: float = 0.7, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, scale, size=(n, 3))


def test_exp_log_roundtrip_rotation() -> None:
    phis = _rand_phi(50, scale=1.0, seed=1)
    for phi in phis:
        R = exp_so3(phi)
        R2 = exp_so3(log_so3(R))
        assert np.allclose(R, R2, atol=1e-7)


def test_log_exp_roundtrip_vector() -> None:
    phis = _rand_phi(50, scale=0.6, seed=2)
    for phi in phis:
        phi2 = log_so3(exp_so3(phi))
        assert np.allclose(phi, phi2, atol=1e-7)


def test_right_jacobian_matches_finite_difference() -> None:
    """Test that right_jacobian matches numerical finite difference."""
    rng = np.random.default_rng(3)
    phi = rng.normal(0.0, 0.4, size=3)
    Jr = right_jacobian(phi)

    eps = 1e-7
    J_num = np.zeros((3, 3))
    for k in range(3):
        e = np.zeros(3)
        e[k] = 1.0
        R_phi = exp_so3(phi)
        R_perturbed = exp_so3(phi + eps * e)
        # Use right-invariant relative rotation for finite difference
        phi_diff = log_so3(R_phi.T @ R_perturbed)
        J_num[:, k] = phi_diff / eps

    assert np.allclose(Jr, J_num, atol=5e-5)


def test_near_pi_roundtrip() -> None:
    """Test exp/log roundtrip near π for branch-cut stability."""
    # Test angles around π with various offsets
    angles = np.array([np.pi - 0.1, np.pi, np.pi + 0.1, np.pi - 0.05, np.pi + 0.05])

    for angle in angles:
        # Create rotation around z-axis
        phi = np.array([0.0, 0.0, angle])
        R = exp_so3(phi)

        # Log and re-exponentiate
        phi2 = log_so3(R)
        R2 = exp_so3(phi2)

        # Check roundtrip
        assert np.allclose(R, R2, atol=1e-7)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
