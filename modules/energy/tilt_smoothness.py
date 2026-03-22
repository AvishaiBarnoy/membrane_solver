"""Tilt smoothness (Dirichlet) energy module.

This module provides a simple diffusion/smoothness penalty for the per-vertex
3D tilt field ``t``:

    E = 1/2 * k_s * ∫ |∇t|^2 dA

Discretization
--------------
We apply the standard cotangent-formula Dirichlet energy component-wise to the
ambient 3D vector field:

    E = (k_s / 4) * Σ_tri [ c0 ||t1 - t2||^2 + c1 ||t2 - t0||^2 + c2 ||t0 - t1||^2 ]

where ``c0,c1,c2`` are the triangle cotangents opposite vertices 0,1,2
(Meyer et al. 2003).

Notes
-----
- This is a *robust proxy* for covariant smoothing of a tangent vector field;
  it does not apply parallel transport between vertex tangent planes. The
  minimizer keeps tilts tangent by projection after explicit updates.
- The current implementation provides an exact gradient w.r.t. the tilt field
  (for fixed geometry) and does not include a shape gradient contribution.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from geometry.curvature import compute_curvature_data
from geometry.entities import Mesh
from geometry.tangent_transport import minimal_rotation_transport, transport_vectors

USES_TILT = True


def _resolve_transport_model(global_params) -> str:
    """Return tilt transport model for smoothness evaluation."""
    if global_params is None:
        return "ambient_v1"
    mode = str(global_params.get("tilt_transport_model", "ambient_v1") or "ambient_v1")
    mode = mode.strip().lower()
    if mode not in {"ambient_v1", "connection_v1"}:
        raise ValueError(
            "tilt_transport_model must be 'ambient_v1' or 'connection_v1'."
        )
    return mode


def _cache_key(mesh: Mesh) -> tuple[int, int, int]:
    vertex_ids = getattr(mesh, "vertex_ids", None)
    n_vertices = int(len(vertex_ids)) if vertex_ids is not None else 0
    return (
        int(getattr(mesh, "_version", 0)),
        int(getattr(mesh, "_facet_loops_version", 0)),
        n_vertices,
    )


def _get_weights_and_tris(
    mesh: Mesh, *, positions: np.ndarray, index_map: Dict[int, int]
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    cache = getattr(mesh, "_tilt_smoothness_cache", None)
    key = _cache_key(mesh)
    if cache is not None and cache.get("key") == key:
        return cache["weights"], cache["tri_rows"]

    _k_vecs, _areas, weights, tri_rows = compute_curvature_data(
        mesh, positions, index_map
    )
    if tri_rows is None or len(tri_rows) == 0:
        return None, None

    mesh._tilt_smoothness_cache = {"key": key, "weights": weights, "tri_rows": tri_rows}
    return weights, tri_rows


def _compute_smoothness_energy_and_gradient(
    mesh: Mesh,
    *,
    k_smooth: float,
    positions: np.ndarray,
    tri_rows: np.ndarray,
    weights: np.ndarray,
    tilts: np.ndarray,
    tilt_grad_arr: np.ndarray | None,
    transport_model: str,
    ctx=None,
    scratch_tag: str = "tilt_smoothness",
) -> float:
    """Assemble smoothness energy and tilt gradient for a chosen transport model."""
    if k_smooth == 0.0 or tri_rows.size == 0:
        return 0.0

    c0 = weights[:, 0]
    c1 = weights[:, 1]
    c2 = weights[:, 2]

    t0_raw = tilts[tri_rows[:, 0]]
    t1_raw = tilts[tri_rows[:, 1]]
    t2_raw = tilts[tri_rows[:, 2]]

    d12 = None
    d20 = None
    d01 = None
    if transport_model == "ambient_v1":
        t0 = t0_raw
        t1 = t1_raw
        t2 = t2_raw
        r0_t = r1_t = r2_t = None
        if ctx is not None:
            d12 = ctx.scratch_array(
                f"{scratch_tag}_d12", shape=t0.shape, dtype=t0.dtype
            )
            d20 = ctx.scratch_array(
                f"{scratch_tag}_d20", shape=t0.shape, dtype=t0.dtype
            )
            d01 = ctx.scratch_array(
                f"{scratch_tag}_d01", shape=t0.shape, dtype=t0.dtype
            )
            np.subtract(t1, t2, out=d12)
            np.subtract(t2, t0, out=d20)
            np.subtract(t0, t1, out=d01)
        else:
            d12 = t1 - t2
            d20 = t2 - t0
            d01 = t0 - t1
    else:
        normals = np.asarray(mesh.vertex_normals(positions), dtype=float)
        v0 = positions[tri_rows[:, 0]]
        v1 = positions[tri_rows[:, 1]]
        v2 = positions[tri_rows[:, 2]]
        tri_normals = np.cross(v1 - v0, v2 - v0)
        tri_norms = np.linalg.norm(tri_normals, axis=1)
        if np.any(tri_norms <= 1.0e-15):
            raise ValueError(
                "Degenerate triangle encountered in connection_v1 smoothness."
            )
        tri_normals = tri_normals / tri_norms[:, None]
        r0 = minimal_rotation_transport(normals[tri_rows[:, 0]], tri_normals)
        r1 = minimal_rotation_transport(normals[tri_rows[:, 1]], tri_normals)
        r2 = minimal_rotation_transport(normals[tri_rows[:, 2]], tri_normals)
        t0 = transport_vectors(t0_raw, r0)
        t1 = transport_vectors(t1_raw, r1)
        t2 = transport_vectors(t2_raw, r2)
        r0_t = np.swapaxes(r0, 1, 2)
        r1_t = np.swapaxes(r1, 1, 2)
        r2_t = np.swapaxes(r2, 1, 2)
        d12 = t1 - t2
        d20 = t2 - t0
        d01 = t0 - t1

    if ctx is not None:
        n12 = ctx.scratch_array(f"{scratch_tag}_n12", shape=(d12.shape[0],))
        n20 = ctx.scratch_array(f"{scratch_tag}_n20", shape=(d20.shape[0],))
        n01 = ctx.scratch_array(f"{scratch_tag}_n01", shape=(d01.shape[0],))
        np.einsum("ij,ij->i", d12, d12, out=n12)
        np.einsum("ij,ij->i", d20, d20, out=n20)
        np.einsum("ij,ij->i", d01, d01, out=n01)
    else:
        n12 = np.einsum("ij,ij->i", d12, d12)
        n20 = np.einsum("ij,ij->i", d20, d20)
        n01 = np.einsum("ij,ij->i", d01, d01)

    energy = float(0.25 * k_smooth * np.sum(c0 * n12 + c1 * n20 + c2 * n01))

    if tilt_grad_arr is not None:
        factor = 0.5 * k_smooth
        g0_local = factor * (c1[:, None] * (t0 - t2) + c2[:, None] * (t0 - t1))
        g1_local = factor * (c2[:, None] * (t1 - t0) + c0[:, None] * (t1 - t2))
        g2_local = factor * (c0[:, None] * (t2 - t1) + c1[:, None] * (t2 - t0))
        if transport_model == "connection_v1":
            g0 = np.einsum("nij,nj->ni", r0_t, g0_local)
            g1 = np.einsum("nij,nj->ni", r1_t, g1_local)
            g2 = np.einsum("nij,nj->ni", r2_t, g2_local)
        else:
            g0 = g0_local
            g1 = g1_local
            g2 = g2_local
        np.add.at(tilt_grad_arr, tri_rows[:, 0], g0)
        np.add.at(tilt_grad_arr, tri_rows[:, 1], g1)
        np.add.at(tilt_grad_arr, tri_rows[:, 2], g2)

    return energy


def compute_energy_and_gradient(
    mesh: Mesh, global_params, param_resolver, *, compute_gradient: bool = True
) -> (
    Tuple[float, Dict[int, np.ndarray]]
    | Tuple[float, Dict[int, np.ndarray], Dict[int, np.ndarray]]
):
    """Dict-based wrapper returning (E, shape_grad[, tilt_grad])."""
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row
    grad_arr = np.zeros_like(positions)
    tilt_grad_arr = np.zeros_like(positions) if compute_gradient else None
    E = compute_energy_and_gradient_array(
        mesh,
        global_params,
        param_resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=grad_arr,
        tilts=None,
        tilt_grad_arr=tilt_grad_arr,
    )
    if not compute_gradient:
        return float(E), {}

    shape_grad = {
        int(vid): grad_arr[row].copy()
        for row, vid in enumerate(mesh.vertex_ids)
        if np.any(grad_arr[row])
    }
    tilt_grad = {
        int(vid): tilt_grad_arr[row].copy()
        for row, vid in enumerate(mesh.vertex_ids)
        if tilt_grad_arr is not None and np.any(tilt_grad_arr[row])
    }
    return float(E), shape_grad, tilt_grad


def compute_energy_and_gradient_array(
    mesh: Mesh,
    global_params,
    param_resolver,
    *,
    positions: np.ndarray,
    index_map: Dict[int, int],
    grad_arr: np.ndarray,
    tilts: np.ndarray | None = None,
    tilt_grad_arr: np.ndarray | None = None,
    ctx=None,
) -> float:
    """Dense-array smoothness energy accumulation (tilt gradient; no shape gradient)."""
    k_smooth = float(param_resolver.get(None, "tilt_smoothness_rigidity") or 0.0)
    if k_smooth == 0.0:
        return 0.0

    weights, tri_rows = _get_weights_and_tris(
        mesh, positions=positions, index_map=index_map
    )
    if tri_rows is None:
        return 0.0

    if tilts is None:
        tilts = mesh.tilts_view()
    else:
        tilts = np.asarray(tilts, dtype=float)
        if tilts.shape != (len(mesh.vertex_ids), 3):
            raise ValueError("tilts must have shape (N_vertices, 3)")

    if tilt_grad_arr is not None:
        tilt_grad_arr = np.asarray(tilt_grad_arr, dtype=float)
        if tilt_grad_arr.shape != (len(mesh.vertex_ids), 3):
            raise ValueError("tilt_grad_arr must have shape (N_vertices, 3)")
    return _compute_smoothness_energy_and_gradient(
        mesh,
        k_smooth=k_smooth,
        positions=positions,
        tri_rows=tri_rows,
        weights=weights,
        tilts=tilts,
        tilt_grad_arr=tilt_grad_arr,
        transport_model=_resolve_transport_model(global_params),
        ctx=ctx,
        scratch_tag="tilt_smoothness",
    )


def compute_energy_array(
    mesh: Mesh,
    global_params,
    param_resolver,
    *,
    positions: np.ndarray,
    index_map: Dict[int, int],
    tilts: np.ndarray | None = None,
    ctx=None,
) -> float:
    """Dense-array smoothness energy (energy only)."""
    _ = global_params
    k_smooth = float(param_resolver.get(None, "tilt_smoothness_rigidity") or 0.0)
    if k_smooth == 0.0:
        return 0.0

    weights, tri_rows = _get_weights_and_tris(
        mesh, positions=positions, index_map=index_map
    )
    if tri_rows is None:
        return 0.0

    if tilts is None:
        tilts = mesh.tilts_view()
    else:
        tilts = np.asarray(tilts, dtype=float)
        if tilts.shape != (len(mesh.vertex_ids), 3):
            raise ValueError("tilts must have shape (N_vertices, 3)")
    return _compute_smoothness_energy_and_gradient(
        mesh,
        k_smooth=k_smooth,
        positions=positions,
        tri_rows=tri_rows,
        weights=weights,
        tilts=tilts,
        tilt_grad_arr=None,
        transport_model=_resolve_transport_model(global_params),
        ctx=ctx,
        scratch_tag="tilt_smoothness",
    )


__all__ = [
    "compute_energy_and_gradient",
    "compute_energy_and_gradient_array",
    "compute_energy_array",
]
