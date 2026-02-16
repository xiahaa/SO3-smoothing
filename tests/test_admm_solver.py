import numpy as np

from admm_solver import proj_ball, solve_inner_admm
from hessian import build_H


def test_proj_ball_basic() -> None:
    x = np.array([3.0, 4.0, 0.0])
    y = proj_ball(x, 2.0)
    assert np.isclose(np.linalg.norm(y), 2.0)

    z = np.array([0.1, 0.2, 0.1])
    z2 = proj_ball(z, 1.0)
    assert np.allclose(z, z2)


def test_admm_shapes_and_residual_drop() -> None:
    M = 30
    H = build_H(M, lam=1.0, mu=0.2, tau=0.1)
    g = np.zeros(3 * M)

    rng = np.random.default_rng(0)
    r_list = [rng.normal(0.0, 0.05, size=3) for _ in range(M)]
    J_list = [np.eye(3) + 0.02 * rng.normal(size=(3, 3)) for _ in range(M)]
    eps = 0.2 * np.ones(M)
    Delta = 0.25

    d, stats = solve_inner_admm(H, g, r_list, J_list, eps, Delta, rho=1.0, max_iter=200, tol=1e-5)
    assert d.shape == (3 * M,)
    assert stats["iter"] >= 1
    assert stats["primal_residual"][-1] <= stats["primal_residual"][0] + 1e-12


def test_admm_close_to_cvxpy_small() -> None:
    cp = __import__("cvxpy")

    M = 20
    H = build_H(M, lam=1.0, mu=0.2, tau=0.1)
    rng = np.random.default_rng(1)
    g = rng.normal(0.0, 0.1, size=3 * M)

    r_list = [rng.normal(0.0, 0.03, size=3) for _ in range(M)]
    J_list = [np.eye(3) + 0.01 * rng.normal(size=(3, 3)) for _ in range(M)]
    eps = 0.18 * np.ones(M)
    Delta = 0.2

    d_admm, _ = solve_inner_admm(H, g, r_list, J_list, eps, Delta, rho=1.0, max_iter=1200, tol=5e-5)

    d = cp.Variable(3 * M)
    cons = []
    for j in range(M):
        dj = d[3 * j : 3 * (j + 1)]
        cons.append(cp.norm(r_list[j] + J_list[j] @ dj, 2) <= eps[j])
        cons.append(cp.norm(dj, 2) <= Delta)
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(d, H) + g @ d), cons)
    prob.solve(solver=cp.SCS, verbose=False, eps=1e-5, max_iters=20000)

    assert prob.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
    d_ref = np.asarray(d.value).reshape(-1)

    obj_admm = float(0.5 * d_admm @ (H @ d_admm) + g @ d_admm)
    obj_ref = float(0.5 * d_ref @ (H @ d_ref) + g @ d_ref)

    rel_gap = abs(obj_admm - obj_ref) / max(1.0, abs(obj_ref))
    assert rel_gap < 5e-3
