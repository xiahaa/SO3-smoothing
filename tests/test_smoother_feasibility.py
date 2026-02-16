import numpy as np

from smoother_socp import tube_smooth_socp
from so3 import exp_so3, log_so3


def _make_data(N: int = 20, tau: float = 0.1):
    t = np.arange(N) * tau
    phi_gt = np.stack([0.4 * np.sin(t), 0.2 * np.cos(0.7 * t), 0.3 * np.sin(1.2 * t)], axis=1)
    R_gt = np.stack([exp_so3(phi_gt[i]) for i in range(N)], axis=0)

    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, 0.03, size=(N, 3))
    R_meas = np.stack([R_gt[i] @ exp_so3(noise[i]) for i in range(N)], axis=0)
    eps = 0.12 * np.ones(N)
    return R_meas, eps


def test_tube_constraint_feasibility() -> None:
    R_meas, eps = _make_data()
    R_hat, info = tube_smooth_socp(
        R_meas,
        eps,
        lam=1.0,
        mu=0.2,
        tau=0.1,
        max_outer=10,
        Delta=0.25,
        tol=1e-7,
        solver="ECOS",
    )

    violations = np.array([np.linalg.norm(log_so3(R_meas[i].T @ R_hat[i])) - eps[i] for i in range(len(eps))])
    assert float(np.max(violations)) <= 1e-6
    assert info["outer_iter"] >= 1
