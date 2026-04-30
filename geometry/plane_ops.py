"""Shared geometry helpers for plane operations and vertex ordering."""

from __future__ import annotations

import numpy as np


def orthonormal_basis_from_normal(
    normal: np.ndarray, eps: float = 1e-15
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate an orthonormal basis (u, v) for the plane defined by a normal vector.

    Args:
        normal: (3,) unit normal vector.
        eps: Tolerance for degeneracy checks.

    Returns:
        tuple of (u, v) basis vectors.
    """
    trial = np.array([1.0, 0.0, 0.0], dtype=float)
    # If normal is too close to trial, use Y axis.
    if abs(float(np.dot(trial, normal))) > 0.9:
        trial = np.array([0.0, 1.0, 0.0], dtype=float)

    u = trial - float(np.dot(trial, normal)) * normal
    nrm_u = float(np.linalg.norm(u))
    if nrm_u < eps:
        # Fallback for extreme cases (should not happen if normal is unit).
        u = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        u /= nrm_u

    v = np.cross(normal, u)
    nrm_v = float(np.linalg.norm(v))
    if nrm_v < eps:
        v = np.array([0.0, 1.0, 0.0], dtype=float)
    else:
        v /= nrm_v

    return u, v


def fit_plane_normal(points: np.ndarray, eps: float = 1e-15) -> np.ndarray | None:
    """
    Fit a plane to a set of points and return its unit normal (via PCA/SVD).

    Args:
        points: (N, 3) array of points.
        eps: Minimum norm for the resulting normal.

    Returns:
        (3,) unit normal or None if fitting fails or points < 3.
    """
    if points.shape[0] < 3:
        return None

    centroid = np.mean(points, axis=0)
    X = points - centroid
    try:
        # The smallest singular vector of the centered data is the normal.
        _, _, vh = np.linalg.svd(X, full_matrices=False)
    except np.linalg.LinAlgError:
        return None

    normal = vh[-1, :]
    nrm = float(np.linalg.norm(normal))
    if nrm < eps:
        return None
    return normal / nrm


def order_by_angle(
    positions: np.ndarray,
    *,
    center: np.ndarray,
    normal: np.ndarray,
    eps: float = 1e-15,
) -> np.ndarray:
    """
    Return indices that sort 3D positions by polar angle in a plane.

    Args:
        positions: (N, 3) coordinates.
        center: (3,) center of rotation.
        normal: (3,) plane normal defining the orientation.
        eps: Tolerance for basis generation.

    Returns:
        (N,) integer indices of sorted positions.
    """
    u, v = orthonormal_basis_from_normal(normal, eps=eps)
    rel = positions - center[None, :]

    # Project relative positions onto basis.
    # Note: rel_plane projection is implicitly handled by dot product with u, v
    # but we can also explicitly subtract the normal component for clarity if needed.
    # rel_plane = rel - np.einsum("ij,j->i", rel, normal)[:, None] * normal[None, :]
    x = rel @ u
    y = rel @ v

    angles = np.arctan2(y, x)
    return np.argsort(angles)
