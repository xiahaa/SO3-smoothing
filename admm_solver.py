"""Custom ADMM solver for SO(3) tube-smoothing inner subproblem."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from hessian import extract_3x3_diag_blocks


Array = np.ndarray


def proj_ball(vec: Array, radius: float) -> Array:
    """Project a 3D vector to an L2 ball with given radius."""
    nrm = float(np.linalg.norm(vec))
    if nrm <= radius or nrm <= 1e-15:
        return vec
    return (radius / nrm) * vec


def _proj_ball_rows(X: Array, radii: Array) -> Array:
    """Row-wise projection of shape (M,3) array to per-row balls."""
    norms = np.linalg.norm(X, axis=1)
    scale = np.ones_like(norms)
    mask = norms > radii
    scale[mask] = radii[mask] / np.maximum(norms[mask], 1e-15)
    return X * scale[:, None]


def _build_blockdiag_from_blocks(blocks: Array) -> sp.csc_matrix:
    """Build sparse block-diagonal matrix from (M,3,3) blocks without Python append loops."""
    M = blocks.shape[0]
    base = 3 * np.arange(M)
    rr = (base[:, None, None] + np.arange(3)[None, :, None]).repeat(3, axis=2)
    cc = (base[:, None, None] + np.arange(3)[None, None, :]).repeat(3, axis=1)
    return sp.coo_matrix(
        (blocks.reshape(-1), (rr.reshape(-1), cc.reshape(-1))),
        shape=(3 * M, 3 * M),
        dtype=np.float64,
    ).tocsc()


def _make_linear_solver(
    A: sp.csc_matrix,
    prefer: str = "splu",
    cg_tol: float = 1e-8,
    cg_max_iter: int = 300,
) -> Tuple[str, Any]:
    """Create cached linear solver for Ax=b."""
    if prefer == "splu":
        try:
            lu = spla.splu(A)
            return "splu", lu
        except Exception:
            pass

    M = A.shape[0] // 3
    diag_blocks = extract_3x3_diag_blocks(A, M)
    inv_blocks = np.linalg.inv(diag_blocks)

    def _precond(x: Array) -> Array:
        xb = x.reshape(M, 3)
        yb = np.einsum("mij,mj->mi", inv_blocks, xb)
        return yb.reshape(-1)

    M_op = spla.LinearOperator(A.shape, matvec=_precond, dtype=np.float64)

    return "pcg", {"M": M_op, "tol": cg_tol, "maxiter": cg_max_iter}


def solve_inner_admm(
    H: sp.csc_matrix,
    g: Array,
    r_list: List[Array],
    J_list: List[Array],
    eps: Array,
    Delta: float,
    rho: float = 1.0,
    max_iter: int = 2000,
    tol: float = 1e-4,
    linear_solver: str = "splu",
) -> Tuple[Array, Dict[str, Any]]:
    """Solve the inner convex QP+SOC constraints by ADMM.

    Problem:
      min 0.5 δ^T H δ + g^T δ
      s.t. y_j = r_j + J_j δ_j, ||y_j||<=eps_j
           w_j = δ_j,           ||w_j||<=Delta
    """
    t0 = time.perf_counter()

    eps = np.asarray(eps, dtype=np.float64).reshape(-1)
    M = eps.shape[0]
    if len(r_list) != M or len(J_list) != M:
        raise ValueError("r_list/J_list length must match eps")

    r = np.vstack([np.asarray(x, dtype=np.float64).reshape(3) for x in r_list])
    J = np.stack([np.asarray(x, dtype=np.float64).reshape(3, 3) for x in J_list], axis=0)

    JT = np.transpose(J, (0, 2, 1))
    JTJ = np.einsum("mij,mjk->mik", JT, J)
    B = JTJ + np.eye(3, dtype=np.float64)[None, :, :]

    A = (H + rho * _build_blockdiag_from_blocks(B)).tocsc()
    solver_kind, solver_obj = _make_linear_solver(A, prefer=linear_solver)

    delta = np.zeros((M, 3), dtype=np.float64)
    y = _proj_ball_rows(r.copy(), eps)
    w = np.zeros((M, 3), dtype=np.float64)
    u = np.zeros((M, 3), dtype=np.float64)
    v = np.zeros((M, 3), dtype=np.float64)

    pri_hist: List[float] = []
    dual_hist: List[float] = []

    for k in range(max_iter):
        y_prev = y.copy()
        w_prev = w.copy()

        rhs_blocks = np.einsum("mij,mj->mi", JT, y - r - u) + (w - v)
        rhs = -g + rho * rhs_blocks.reshape(-1)

        if solver_kind == "splu":
            delta_vec = solver_obj.solve(rhs)
        else:
            delta_vec, info = spla.cg(
                A,
                rhs,
                M=solver_obj["M"],
                rtol=solver_obj["tol"],
                atol=0.0,
                maxiter=solver_obj["maxiter"],
            )
            if info != 0:
                raise RuntimeError(f"PCG failed in ADMM delta-update, info={info}")

        delta = delta_vec.reshape(M, 3)

        t = r + np.einsum("mij,mj->mi", J, delta) + u
        y = _proj_ball_rows(t, eps)

        s = delta + v
        y_tr = np.full(M, Delta, dtype=np.float64)
        w = _proj_ball_rows(s, y_tr)

        res1 = r + np.einsum("mij,mj->mi", J, delta) - y
        res2 = delta - w
        u = u + res1
        v = v + res2

        pri = float(np.sqrt(np.sum(res1**2) + np.sum(res2**2)))
        dual = float(rho * np.sqrt(np.sum((y - y_prev) ** 2) + np.sum((w - w_prev) ** 2)))
        pri_hist.append(pri)
        dual_hist.append(dual)

        if pri < tol and dual < tol:
            break

    elapsed = time.perf_counter() - t0
    stats = {
        "iter": len(pri_hist),
        "primal_residual": pri_hist,
        "dual_residual": dual_hist,
        "elapsed_sec": elapsed,
        "solver": solver_kind,
        "A_nnz": int(A.nnz),
    }

    return delta.reshape(-1), stats


__all__ = ["proj_ball", "solve_inner_admm"]
