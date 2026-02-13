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


def _get_reference(mesh, vids: list[int], *, group: str | None, global_params):
    rim_group = _resolve_rim_group(global_params)
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
    return ref


def _choose_anchor_triplet(ref: np.ndarray) -> tuple[int, int, int | None]:
    n = ref.shape[0]
    if n < 2:
        return 0, 0, None
    anchor_a = 0
    d2 = np.sum((ref - ref[anchor_a][None, :]) ** 2, axis=1)
    anchor_b = int(np.argmax(d2))
    if n < 3:
        return anchor_a, anchor_b, None
    ab = ref[anchor_b] - ref[anchor_a]
    area_scores = np.linalg.norm(np.cross(ref - ref[anchor_a][None, :], ab), axis=1)
    area_scores[anchor_a] = -1.0
    area_scores[anchor_b] = -1.0
    anchor_c = int(np.argmax(area_scores))
    if area_scores[anchor_c] <= 1e-12:
        return anchor_a, anchor_b, None
    return anchor_a, anchor_b, anchor_c


def _independent_distance_pairs(ref: np.ndarray) -> list[tuple[int, int]]:
    n = ref.shape[0]
    if n < 2:
        return []
    a, b, c = _choose_anchor_triplet(ref)
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    def add_pair(i: int, j: int) -> None:
        if i == j:
            return
        key = (i, j) if i < j else (j, i)
        if key in seen:
            return
        seen.add(key)
        pairs.append(key)

    for i in range(n):
        if i != a:
            add_pair(a, i)
    for i in range(n):
        if i not in {a, b}:
            add_pair(b, i)
    if c is not None:
        for i in range(n):
            if i not in {a, b, c}:
                add_pair(c, i)
    return pairs


def constraint_gradients(mesh, global_params) -> list[dict[int, np.ndarray]] | None:
    """Return rigid-disk shape-constraint gradients for KKT projection.

    The constraints are independent pairwise distance invariants built from the
    cached rigid-disk reference configuration. This preserves rigid-body
    motions (translations/rotations) while suppressing non-rigid deformation.
    """
    group = _resolve_group(global_params)
    vids = _collect_disk_vertices(mesh, group=group)
    if len(vids) < 2:
        return None

    ref = _get_reference(mesh, vids, group=group, global_params=global_params)
    pairs = _independent_distance_pairs(ref)
    if not pairs:
        return None

    gradients: list[dict[int, np.ndarray]] = []
    for i, j in pairs:
        vid_i = int(vids[i])
        vid_j = int(vids[j])
        vi = mesh.vertices.get(vid_i)
        vj = mesh.vertices.get(vid_j)
        if vi is None or vj is None:
            continue
        if getattr(vi, "fixed", False) and getattr(vj, "fixed", False):
            continue
        diff = np.asarray(vi.position - vj.position, dtype=float)
        gC: dict[int, np.ndarray] = {}
        if not getattr(vi, "fixed", False):
            gC[vid_i] = diff.copy()
        if not getattr(vj, "fixed", False):
            gC[vid_j] = (-diff).copy()
        if gC:
            gradients.append(gC)

    return gradients or None


def constraint_gradients_array(
    mesh,
    global_params,
    *,
    positions: np.ndarray,
    index_map: dict[int, int],
) -> list[np.ndarray] | None:
    _ = global_params
    group = _resolve_group(global_params)
    vids = _collect_disk_vertices(mesh, group=group)
    if len(vids) < 2:
        return None

    ref = _get_reference(mesh, vids, group=group, global_params=global_params)
    pairs = _independent_distance_pairs(ref)
    if not pairs:
        return None

    gradients: list[np.ndarray] = []
    for i, j in pairs:
        vid_i = int(vids[i])
        vid_j = int(vids[j])
        row_i = index_map.get(vid_i)
        row_j = index_map.get(vid_j)
        if row_i is None or row_j is None:
            continue
        vi = mesh.vertices.get(vid_i)
        vj = mesh.vertices.get(vid_j)
        if vi is None or vj is None:
            continue
        if getattr(vi, "fixed", False) and getattr(vj, "fixed", False):
            continue
        diff = np.asarray(positions[row_i] - positions[row_j], dtype=float)
        g_arr = np.zeros_like(positions)
        if not getattr(vi, "fixed", False):
            g_arr[row_i] += diff
        if not getattr(vj, "fixed", False):
            g_arr[row_j] -= diff
        if np.any(g_arr):
            gradients.append(g_arr)

    return gradients or None


def constraint_gradients_rows_array(
    mesh,
    global_params,
    *,
    positions: np.ndarray,
    index_map: dict[int, int],
) -> list[tuple[np.ndarray, np.ndarray]] | None:
    """Sparse row variant of rigid-disk shape gradients."""
    group = _resolve_group(global_params)
    vids = _collect_disk_vertices(mesh, group=group)
    if len(vids) < 2:
        return None

    ref = _get_reference(mesh, vids, group=group, global_params=global_params)
    pairs = _independent_distance_pairs(ref)
    if not pairs:
        return None

    gradients: list[tuple[np.ndarray, np.ndarray]] = []
    for i, j in pairs:
        vid_i = int(vids[i])
        vid_j = int(vids[j])
        row_i = index_map.get(vid_i)
        row_j = index_map.get(vid_j)
        if row_i is None or row_j is None:
            continue
        vi = mesh.vertices.get(vid_i)
        vj = mesh.vertices.get(vid_j)
        if vi is None or vj is None:
            continue
        fix_i = getattr(vi, "fixed", False)
        fix_j = getattr(vj, "fixed", False)
        if fix_i and fix_j:
            continue
        diff = np.asarray(positions[row_i] - positions[row_j], dtype=float)
        if fix_i:
            gradients.append(
                (np.asarray([int(row_j)], dtype=int), (-diff).reshape(1, 3))
            )
            continue
        if fix_j:
            gradients.append((np.asarray([int(row_i)], dtype=int), diff.reshape(1, 3)))
            continue
        gradients.append(
            (
                np.asarray([int(row_i), int(row_j)], dtype=int),
                np.vstack([diff, -diff]),
            )
        )

    return gradients or None


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
    ref = _get_reference(mesh, vids, group=group, global_params=global_params)

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


__all__ = [
    "enforce_constraint",
    "constraint_gradients",
    "constraint_gradients_array",
    "constraint_gradients_rows_array",
]
