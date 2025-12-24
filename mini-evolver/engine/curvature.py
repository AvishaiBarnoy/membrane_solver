"""Discrete mean curvature calculations."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, List, Tuple

from .geometry import Vector, v_dot, v_norm, v_sub

if TYPE_CHECKING:
    from .mesh import Mesh


def _cotangent(u: Vector, v: Vector) -> float:
    cross = (
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    )
    denom = v_norm(cross)
    if denom < 1e-12:
        return 0.0
    return v_dot(u, v) / denom


def _mean_curvature_data(
    mesh: "Mesh", positions: Dict[int, Vector], triangles: List[Tuple[int, int, int]]
) -> Tuple[Dict[int, Vector], Dict[int, float]]:
    cot_weights: Dict[Tuple[int, int], float] = {}
    vertex_area: Dict[int, float] = {vid: 0.0 for vid in mesh.vertices}
    for a, b, c in triangles:
        pa = positions[a]
        pb = positions[b]
        pc = positions[c]
        ab = v_sub(pb, pa)
        ac = v_sub(pc, pa)
        area = 0.5 * v_norm(
            (
                ab[1] * ac[2] - ab[2] * ac[1],
                ab[2] * ac[0] - ab[0] * ac[2],
                ab[0] * ac[1] - ab[1] * ac[0],
            )
        )
        vertex_area[a] += area / 3.0
        vertex_area[b] += area / 3.0
        vertex_area[c] += area / 3.0
        cot_a = _cotangent(v_sub(pb, pa), v_sub(pc, pa))
        cot_b = _cotangent(v_sub(pa, pb), v_sub(pc, pb))
        cot_c = _cotangent(v_sub(pa, pc), v_sub(pb, pc))
        key = (b, c) if b < c else (c, b)
        cot_weights[key] = cot_weights.get(key, 0.0) + cot_a
        key = (a, c) if a < c else (c, a)
        cot_weights[key] = cot_weights.get(key, 0.0) + cot_b
        key = (a, b) if a < b else (b, a)
        cot_weights[key] = cot_weights.get(key, 0.0) + cot_c
    mean_vec: Dict[int, Vector] = {vid: (0.0, 0.0, 0.0) for vid in mesh.vertices}
    for (i, j), w in cot_weights.items():
        pi = positions[i]
        pj = positions[j]
        diff = v_sub(pi, pj)
        mean_vec[i] = (
            mean_vec[i][0] + w * diff[0],
            mean_vec[i][1] + w * diff[1],
            mean_vec[i][2] + w * diff[2],
        )
        mean_vec[j] = (
            mean_vec[j][0] - w * diff[0],
            mean_vec[j][1] - w * diff[1],
            mean_vec[j][2] - w * diff[2],
        )
    for vid, area in vertex_area.items():
        if area > 1e-12:
            mean_vec[vid] = (
                0.5 * mean_vec[vid][0] / area,
                0.5 * mean_vec[vid][1] / area,
                0.5 * mean_vec[vid][2] / area,
            )
    return mean_vec, vertex_area


def curvature_energy(
    mesh: "Mesh",
    positions: Dict[int, Vector],
    triangles: List[Tuple[int, int, int]],
    h0: float = 0.0,
) -> float:
    mean_vec, vertex_area = _mean_curvature_data(mesh, positions, triangles)
    energy = 0.0
    for vid, area in vertex_area.items():
        if area <= 0.0:
            continue
        hx, hy, hz = mean_vec[vid]
        h = math.sqrt(hx * hx + hy * hy + hz * hz)
        diff = h - h0
        energy += area * diff * diff
    return energy


def _collect_triangles(
    mesh: "Mesh", face_ids: List[int] | None
) -> List[Tuple[int, int, int]]:
    if not face_ids:
        return list(mesh.triangles)
    triangles: List[Tuple[int, int, int]] = []
    from .mesh import triangulate_face

    for fid in face_ids:
        face = mesh.faces.get(abs(fid))
        if not face:
            continue
        triangles.extend(triangulate_face(face.vertex_loop))
    return triangles


def curvature_energy_and_grads(
    mesh: "Mesh",
    positions: Dict[int, Vector],
    face_ids: List[int] | None = None,
    h0: float = 0.0,
    eps: float = 1e-6,
) -> Tuple[float, Dict[int, Vector]]:
    triangles = _collect_triangles(mesh, face_ids)
    base_energy = curvature_energy(mesh, positions, triangles, h0=h0)
    grads: Dict[int, Vector] = {}
    for vid in positions:
        if vid in mesh.fixed_ids:
            grads[vid] = (0.0, 0.0, 0.0)
            continue
        base = positions[vid]
        deriv = [0.0, 0.0, 0.0]
        for dim in range(3):
            if vid in mesh.constraint_vertices:
                axis = mesh.constraint_axes.get(mesh.vertices[vid].constraint, 2)
                if dim == axis:
                    continue
            plus = list(base)
            minus = list(base)
            plus[dim] += eps
            minus[dim] -= eps
            pos_plus = dict(positions)
            pos_minus = dict(positions)
            pos_plus[vid] = (plus[0], plus[1], plus[2])
            pos_minus[vid] = (minus[0], minus[1], minus[2])
            e_plus = curvature_energy(mesh, pos_plus, triangles, h0=h0)
            e_minus = curvature_energy(mesh, pos_minus, triangles, h0=h0)
            deriv[dim] = (e_plus - e_minus) / (2.0 * eps)
        grads[vid] = (deriv[0], deriv[1], deriv[2])
    return base_energy, grads
