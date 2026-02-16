"""SO(3) utilities with numerically stable exponential/logarithm maps and Jacobians."""

from __future__ import annotations

from typing import Iterable

import numpy as np

_EPS = 1e-12
_SMALL_ANGLE = 1e-8


def hat(phi: np.ndarray) -> np.ndarray:
    """Return skew-symmetric matrix (hat operator) of a 3-vector.

    Args:
        phi: Array-like shape (3,).

    Returns:
        3x3 skew-symmetric matrix.
    """
    phi = np.asarray(phi, dtype=float).reshape(3)
    return np.array(
        [
            [0.0, -phi[2], phi[1]],
            [phi[2], 0.0, -phi[0]],
            [-phi[1], phi[0], 0.0],
        ]
    )


def vee(Phi: np.ndarray) -> np.ndarray:
    """Return vector (vee operator) of a skew-symmetric matrix.

    Args:
        Phi: 3x3 skew-symmetric matrix.

    Returns:
        Vector shape (3,).
    """
    Phi = np.asarray(Phi, dtype=float).reshape(3, 3)
    return np.array([Phi[2, 1], Phi[0, 2], Phi[1, 0]], dtype=float)


def exp_so3(phi: np.ndarray) -> np.ndarray:
    """SO(3) exponential map via Rodrigues formula with small-angle stabilization."""
    phi = np.asarray(phi, dtype=float).reshape(3)
    theta = float(np.linalg.norm(phi))
    Phi = hat(phi)
    Phi2 = Phi @ Phi

    if theta < _SMALL_ANGLE:
        a = 1.0 - theta**2 / 6.0 + theta**4 / 120.0
        b = 0.5 - theta**2 / 24.0 + theta**4 / 720.0
    else:
        a = np.sin(theta) / theta
        b = (1.0 - np.cos(theta)) / (theta**2)

    return np.eye(3) + a * Phi + b * Phi2


def _log_near_pi(R: np.ndarray, theta: float) -> np.ndarray:
    """Robust axis extraction for log map near pi."""
    A = 0.5 * (R + np.eye(3))
    axis = np.empty(3)
    idx = int(np.argmax(np.diag(A)))
    axis[idx] = np.sqrt(max(A[idx, idx], 0.0))
    denom = max(axis[idx], _EPS)
    j = (idx + 1) % 3
    k = (idx + 2) % 3
    axis[j] = A[idx, j] / denom
    axis[k] = A[idx, k] / denom

    nrm = np.linalg.norm(axis)
    if nrm < _EPS:
        axis = np.array([1.0, 0.0, 0.0])
    else:
        axis /= nrm
    return theta * axis


def log_so3(R: np.ndarray) -> np.ndarray:
    """SO(3) logarithm map.

    Args:
        R: Rotation matrix shape (3, 3).

    Returns:
        Axis-angle vector phi in R^3.
    """
    R = np.asarray(R, dtype=float).reshape(3, 3)
    # Re-orthonormalize lightly for robustness.
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        R = U @ Vt

    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))

    if theta < _SMALL_ANGLE:
        return vee(0.5 * (R - R.T))

    if np.pi - theta < 1e-5:
        return _log_near_pi(R, theta)

    return vee((theta / (2.0 * np.sin(theta))) * (R - R.T))


def right_jacobian(phi: np.ndarray) -> np.ndarray:
    """Right Jacobian of SO(3), J_r(phi)."""
    phi = np.asarray(phi, dtype=float).reshape(3)
    theta = float(np.linalg.norm(phi))
    Phi = hat(phi)
    Phi2 = Phi @ Phi

    if theta < _SMALL_ANGLE:
        return np.eye(3) - 0.5 * Phi + (1.0 / 6.0) * Phi2

    a = (1.0 - np.cos(theta)) / (theta**2)
    b = (theta - np.sin(theta)) / (theta**3)
    return np.eye(3) - a * Phi + b * Phi2


def right_jacobian_inv(phi: np.ndarray) -> np.ndarray:
    """Inverse of right Jacobian of SO(3), J_r(phi)^{-1}."""
    phi = np.asarray(phi, dtype=float).reshape(3)
    theta = float(np.linalg.norm(phi))
    Phi = hat(phi)
    Phi2 = Phi @ Phi

    if theta < _SMALL_ANGLE:
        return np.eye(3) + 0.5 * Phi + (1.0 / 12.0) * Phi2

    half_theta = 0.5 * theta
    cot_half = 1.0 / np.tan(half_theta)
    c = (1.0 / theta**2) * (1.0 - half_theta * cot_half)
    return np.eye(3) + 0.5 * Phi + c * Phi2


def geodesic_angle(R1: np.ndarray, R2: np.ndarray) -> float:
    """Return geodesic angle d(R1, R2) on SO(3)."""
    return float(np.linalg.norm(log_so3(R1.T @ R2)))


def batch_log(R_seq: Iterable[np.ndarray]) -> np.ndarray:
    """Apply log_so3 on sequence of rotations and stack as (N, 3)."""
    return np.vstack([log_so3(R) for R in R_seq])
