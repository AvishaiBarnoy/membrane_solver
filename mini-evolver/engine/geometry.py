"""Vector utilities and planar polygon helpers."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

Vector = Tuple[float, float, float]


def v_add(a: Vector, b: Vector) -> Vector:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def v_sub(a: Vector, b: Vector) -> Vector:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def v_scale(a: Vector, s: float) -> Vector:
    return (a[0] * s, a[1] * s, a[2] * s)


def v_dot(a: Vector, b: Vector) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def v_cross(a: Vector, b: Vector) -> Vector:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def v_norm(a: Vector) -> float:
    return math.sqrt(v_dot(a, a))


def accumulate(grads: Dict[int, Vector], vid: int, delta: Vector) -> None:
    grads[vid] = v_add(grads.get(vid, (0.0, 0.0, 0.0)), delta)


def build_edge_loops(edge_ids: List[int], edges: Dict[int, "Edge"]) -> List[List[int]]:
    adjacency: Dict[int, List[int]] = {}
    for eid in edge_ids:
        edge = edges[eid]
        adjacency.setdefault(edge.tail, []).append(edge.head)
        adjacency.setdefault(edge.head, []).append(edge.tail)
    loops: List[List[int]] = []
    used: set[Tuple[int, int]] = set()
    for eid in edge_ids:
        edge = edges[eid]
        start = edge.tail
        next_v = edge.head
        if (start, next_v) in used or (next_v, start) in used:
            continue
        loop = [start, next_v]
        used.add((start, next_v))
        prev = start
        curr = next_v
        while True:
            neighbors = adjacency.get(curr, [])
            if len(neighbors) < 2:
                break
            nxt = neighbors[0] if neighbors[0] != prev else neighbors[1]
            if (curr, nxt) in used or (nxt, curr) in used:
                if nxt == start:
                    break
                break
            loop.append(nxt)
            used.add((curr, nxt))
            prev, curr = curr, nxt
            if curr == start:
                break
        if len(loop) > 2 and loop[-1] == start:
            loop.pop()
        if len(loop) > 2:
            loops.append(loop)
    return loops


def polygon_area_and_grads(
    loop: List[int], positions: Dict[int, Vector], axis: int
) -> Tuple[float, Dict[int, Vector]]:
    coords = []
    for vid in loop:
        x, y, z = positions[vid]
        if axis == 0:
            coords.append((y, z))
        elif axis == 1:
            coords.append((x, z))
        else:
            coords.append((x, y))
    n = len(coords)
    area = 0.0
    grads: Dict[int, Vector] = {}
    for i in range(n):
        u0, v0 = coords[i]
        u1, v1 = coords[(i + 1) % n]
        area += u0 * v1 - u1 * v0
    area *= 0.5
    for i, vid in enumerate(loop):
        u_prev, v_prev = coords[i - 1]
        u_next, v_next = coords[(i + 1) % n]
        du = 0.5 * (v_next - v_prev)
        dv = 0.5 * (u_prev - u_next)
        if axis == 0:
            g = (0.0, du, dv)
        elif axis == 1:
            g = (du, 0.0, dv)
        else:
            g = (du, dv, 0.0)
        grads[vid] = g
    return area, grads


# Forward reference for type checking only.
class Edge:  # pragma: no cover
    tail: int
    head: int
