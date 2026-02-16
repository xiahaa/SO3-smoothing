import numpy as np

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
    rng = np.random.default_rng(3)
    phi = rng.normal(0.0, 0.4, size=3)
    Jr = right_jacobian(phi)

    eps = 1e-7
    J_num = np.zeros((3, 3))
    for k in range(3):
        e = np.zeros(3)
        e[k] = 1.0
        # Exp(phi + d) ≈ Exp(phi) Exp(Jr(phi) d)
        R_rel = exp_so3(phi).T @ exp_so3(phi + eps * e)
        J_num[:, k] = log_so3(R_rel) / eps

    assert np.allclose(Jr, J_num, atol=5e-5)
