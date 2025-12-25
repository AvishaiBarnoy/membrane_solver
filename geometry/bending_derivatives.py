from __future__ import annotations

import numpy as np

from geometry.entities import _fast_cross


def grad_dot(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Gradient of dot(u, v) w.r.t u and v.
    d(u.v) = du.v + u.dv
    Returns (v, u).
    """
    return v, u


def grad_norm_sq(u: np.ndarray) -> np.ndarray:
    """
    Gradient of dot(u, u) w.r.t u.
    Returns 2*u.
    """
    return 2.0 * u


def grad_norm(u: np.ndarray) -> np.ndarray:
    """Gradient of ``||u||`` w.r.t ``u`` for a single vector."""
    n = float(np.linalg.norm(u))
    if n < 1e-15:
        return np.zeros_like(u)
    return u / n


def grad_cross(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Gradient of cross(u, v) is a tensor.
    However, we usually use this composed with a dot product (scalar triple product).

    This function is not returning the Jacobian (tensor) but is a placeholder.
    In practice, for scalar functions f = a . (u x v),
    grad_u f = v x a
    grad_v f = a x u

    We will handle cross product derivatives in context.
    """
    raise NotImplementedError("Use scalar triple product derivative logic directly.")


def grad_cotan(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Gradient of cot(theta) where theta is angle between vectors u and v.
    cot = (u.v) / |u x v|

    Returns (grad_u, grad_v) for batched inputs (N, 3).
    """
    # Vector calculus:
    # cot = (u·v)/||u×v|| = C/S.
    # ∇_u cot = v/S - (C/S^3) * (v × (u×v))
    # ∇_v cot = u/S - (C/S^3) * ((u×v) × u)
    dot = np.einsum("ij,ij->i", u, v)
    w = _fast_cross(u, v)
    S = np.linalg.norm(w, axis=1)
    mask = S > 1e-15

    grad_u = np.zeros_like(u)
    grad_v = np.zeros_like(v)
    if not np.any(mask):
        return grad_u, grad_v

    C = dot[mask]
    Sm = S[mask]
    invS = 1.0 / Sm
    invS3 = 1.0 / (Sm * Sm * Sm)

    v_cross_w = _fast_cross(v[mask], w[mask])
    w_cross_u = _fast_cross(w[mask], u[mask])

    grad_u[mask] = v[mask] * invS[:, None] - (C * invS3)[:, None] * v_cross_w
    grad_v[mask] = u[mask] * invS[:, None] - (C * invS3)[:, None] * w_cross_u
    return grad_u, grad_v


def grad_triangle_area(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Gradient of triangle area T = 0.5 * |u x v| w.r.t u and v.
    Returns (grad_u, grad_v).
    """
    # Area T = 0.5 * ||u×v|| = 0.5 * S
    # ∇_u T = 0.5 * (v × (u×v)) / S
    # ∇_v T = 0.5 * ((u×v) × u) / S
    w = _fast_cross(u, v)
    S = np.linalg.norm(w, axis=1)
    mask = S > 1e-15

    grad_u = np.zeros_like(u)
    grad_v = np.zeros_like(v)
    if not np.any(mask):
        return grad_u, grad_v

    invS = 1.0 / S[mask]
    grad_u[mask] = 0.5 * _fast_cross(v[mask], w[mask]) * invS[:, None]
    grad_v[mask] = 0.5 * _fast_cross(w[mask], u[mask]) * invS[:, None]
    return grad_u, grad_v
