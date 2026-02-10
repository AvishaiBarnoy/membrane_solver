"""Rigid disk constraint enforcing a rigid-body transform on a disk patch.

The disk is selected by an explicit group name via `rigid_disk_group` or
falls back to `preset: disk` if the group is not set.

The constraint projects disk vertices onto the closest rigid-body transform of
an initial reference configuration, and enforces a rim radius in the disk's
own plane using a second rigid fit.
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np

logger = logging.getLogger("membrane_solver")


def _resolve_group(global_params) -> str | None:
    if global_params is None:
        return None
    raw = global_params.get("rigid_disk_group")
    if raw is None:
        return None
    group = str(raw).strip()
    return group if group else None


def _resolve_rim_group(global_params) -> str:
    if global_params is None:
        return "rim"
    raw = global_params.get("rigid_disk_rim_group")
    if raw is None:
        return "rim"
    group = str(raw).strip()
    return group if group else "rim"


def _resolve_target_radius(mesh, global_params) -> float | None:
    if global_params is not None:
        raw = global_params.get("rigid_disk_radius")
        if raw is not None:
            try:
                return float(raw)
            except (TypeError, ValueError):
                pass
    # Fallback: use disk definition pin_to_circle_radius if present.
    defs = getattr(mesh, "definitions", {}) or {}
    disk_def = defs.get("disk") if isinstance(defs, dict) else None
    if isinstance(disk_def, dict):
        raw = disk_def.get("pin_to_circle_radius")
        if raw is not None:
            try:
                return float(raw)
            except (TypeError, ValueError):
                pass
    return None


def _collect_disk_vertices(mesh, *, group: str | None) -> list[int]:
    vids: list[int] = []
    for vid, vertex in mesh.vertices.items():
        opts = getattr(vertex, "options", None) or {}
        if group is None:
            if opts.get("preset") == "disk":
                vids.append(int(vid))
            continue
        if opts.get("rigid_disk_group") == group:
            vids.append(int(vid))
    return vids


def _collect_rim_indices(mesh, vids: list[int], *, rim_group: str) -> list[int]:
    rim: list[int] = []
    for idx, vid in enumerate(vids):
        opts = getattr(mesh.vertices[int(vid)], "options", None) or {}
        if opts.get("rim_slope_match_group") == rim_group:
            rim.append(idx)
    return rim


def _kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return rotation R and translation t mapping P -> Q (least squares)."""
    if P.shape != Q.shape:
        raise ValueError("Rigid disk: reference/current shapes mismatch.")
    Pc = P.mean(axis=0)
    Qc = Q.mean(axis=0)
    P0 = P - Pc
    Q0 = Q - Qc
    H = P0.T @ Q0
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0.0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    t = Qc - R @ Pc
    return R, t


def _fit_plane_normal(points: np.ndarray) -> np.ndarray:
    centroid = np.mean(points, axis=0)
    X = points - centroid
    _, _, vh = np.linalg.svd(X, full_matrices=False)
    normal = vh[-1, :]
    nrm = float(np.linalg.norm(normal))
    if nrm < 1e-15:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return normal / nrm


def _get_ref_cache(mesh) -> dict:
    cache = getattr(mesh, "_rigid_disk_ref", None)
    if cache is None:
        cache = {}
        setattr(mesh, "_rigid_disk_ref", cache)
    return cache


def _ref_key(group: str | None) -> str:
    return "__preset_disk__" if group is None else f"group:{group}"


def _flatten_reference(
    positions: np.ndarray, *, rim_indices: list[int], target_radius: float | None
) -> np.ndarray:
    center = positions.mean(axis=0)
    normal = _fit_plane_normal(positions)
    rel = positions - center[None, :]
    rel_plane = rel - (rel @ normal)[:, None] * normal[None, :]
    flattened = center[None, :] + rel_plane

    if rim_indices and target_radius is not None:
        for idx in rim_indices:
            v = flattened[idx] - center
            v_plane = v - np.dot(v, normal) * normal
            nrm = float(np.linalg.norm(v_plane))
            if nrm < 1e-12:
                continue
            flattened[idx] = center + (target_radius * v_plane / nrm)
    return flattened


def _build_reference(
    mesh, vids: Iterable[int], *, rim_indices: list[int], target_radius: float | None
) -> np.ndarray:
    positions = np.array([mesh.vertices[int(v)].position for v in vids], dtype=float)
    return _flatten_reference(
        positions, rim_indices=rim_indices, target_radius=target_radius
    )


def enforce_constraint(mesh, global_params=None, **_kwargs) -> None:
    """Project disk vertices onto a rigid-body transform of their reference."""
    group = _resolve_group(global_params)
    rim_group = _resolve_rim_group(global_params)
    vids = _collect_disk_vertices(mesh, group=group)
    if not vids:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("rigid_disk: no vertices matched; skipping")
        return
    if len(vids) < 3:
        raise ValueError("rigid_disk: need at least 3 vertices")

    rim_indices = _collect_rim_indices(mesh, vids, rim_group=rim_group)
    target_radius = _resolve_target_radius(mesh, global_params)

    cache = _get_ref_cache(mesh)
    key = _ref_key(group)
    ref = cache.get(key)
    if ref is None or ref.shape[0] != len(vids):
        ref = _build_reference(
            mesh, vids, rim_indices=rim_indices, target_radius=target_radius
        )
        cache[key] = ref

    current = np.array([mesh.vertices[int(v)].position for v in vids], dtype=float)
    R, t = _kabsch(ref, current)
    corrected = (ref @ R.T) + t

    if rim_indices and target_radius is not None:
        ref_center = ref.mean(axis=0)
        ref_normal = _fit_plane_normal(ref)
        center = R @ ref_center + t
        normal = R @ ref_normal
        normal = normal / max(float(np.linalg.norm(normal)), 1e-12)

        for idx in rim_indices:
            p = corrected[idx]
            v = p - center
            v_plane = v - np.dot(v, normal) * normal
            nrm = float(np.linalg.norm(v_plane))
            if nrm < 1e-12:
                continue
            corrected[idx] = center + (target_radius * v_plane / nrm)

        R, t = _kabsch(ref, corrected)
        corrected = (ref @ R.T) + t

    for idx, vid in enumerate(vids):
        mesh.vertices[int(vid)].position[:] = corrected[idx]


__all__ = ["enforce_constraint"]
