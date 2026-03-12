"""Sparse Hessian assembly for SO(3) tube smoothing."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import scipy.sparse as sp


def build_D1_D2(M: int) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    """Build first/second-order finite-difference matrices.

    Args:
        M: Number of trajectory knots.

    Returns:
        D1: shape ((M-1), M)
        D2: shape ((M-2), M), empty when M < 3
    """
    if M < 2:
        raise ValueError("M must be >= 2")

    D1 = sp.diags(
        diagonals=[-np.ones(M - 1), np.ones(M - 1)],
        offsets=[0, 1],
        shape=(M - 1, M),
        format="csr",
        dtype=np.float64,
    )

    if M >= 3:
        D2 = sp.diags(
            diagonals=[np.ones(M - 2), -2.0 * np.ones(M - 2), np.ones(M - 2)],
            offsets=[0, 1, 2],
            shape=(M - 2, M),
            format="csr",
            dtype=np.float64,
        )
    else:
        D2 = sp.csr_matrix((0, M), dtype=np.float64)

    return D1, D2


def build_H(M: int, lam: float, mu: float, tau: float, damping: float = 1e-9) -> sp.csc_matrix:
    """Build Hessian H = H1 + H2 as sparse CSC matrix.

    H1 = (lam / tau)   * (D1^T D1) ⊗ I3
    H2 = (mu / tau^3)  * (D2^T D2) ⊗ I3
    """
    if tau <= 0:
        raise ValueError("tau must be positive")

    D1, D2 = build_D1_D2(M)
    I3 = sp.eye(3, format="csc", dtype=np.float64)

    H1 = (lam / tau) * sp.kron(D1.T @ D1, I3, format="csc")
    H2 = (mu / tau**3) * sp.kron(D2.T @ D2, I3, format="csc")
    H = (H1 + H2).tocsc()

    if damping > 0:
        H = H + damping * sp.eye(3 * M, format="csc", dtype=np.float64)

    return H


def extract_3x3_diag_blocks(A: sp.spmatrix, M: int) -> np.ndarray:
    """Extract dense 3x3 diagonal blocks from sparse matrix A (shape 3M x 3M)."""
    A = A.tocsr()
    out = np.zeros((M, 3, 3), dtype=np.float64)
    for j in range(M):
        sl = slice(3 * j, 3 * (j + 1))
        out[j] = A[sl, sl].toarray()
    return out
