from __future__ import annotations

import numpy as np

from geometry.geom_io import parse_geometry
from modules.constraints import rim_slope_match_out


def _build_mesh() -> dict:
    """Build a tiny 3-ring mesh for rim-slope thetaB-scalar constraint tests.

    This avoids loading from `meshes/` so the test remains stable when users
    edit repository YAML inputs.
    """

    def ring_vertices(r: float, *, n: int) -> list[list[float]]:
        out: list[list[float]] = []
        for k in range(n):
            ang = 2.0 * np.pi * k / n
            out.append(
                [float(r) * float(np.cos(ang)), float(r) * float(np.sin(ang)), 0.0]
            )
        return out

    n = 6
    vertices: list[list] = [[0.0, 0.0, 0.0]]
    disk_ring = list(range(1, 1 + n))
    rim_ring = list(range(1 + n, 1 + 2 * n))
    outer_ring = list(range(1 + 2 * n, 1 + 3 * n))

    for x, y, z in ring_vertices(0.5, n=n):
        vertices.append([x, y, z, {"rim_slope_match_group": "disk"}])
    for x, y, z in ring_vertices(1.0, n=n):
        vertices.append([x, y, z, {"rim_slope_match_group": "rim"}])
    for x, y, z in ring_vertices(1.5, n=n):
        vertices.append([x, y, z, {"rim_slope_match_group": "outer"}])

    triangles: list[tuple[int, int, int]] = []
    for k in range(n):
        triangles.append((0, disk_ring[k], disk_ring[(k + 1) % n]))
    for A, B in ((disk_ring, rim_ring), (rim_ring, outer_ring)):
        for k in range(n):
            a0 = A[k]
            a1 = A[(k + 1) % n]
            b0 = B[k]
            b1 = B[(k + 1) % n]
            triangles.append((a0, a1, b0))
            triangles.append((b0, a1, b1))

    edges: list[list[int]] = []
    edge_map: dict[tuple[int, int], int] = {}

    def get_edge(u: int, v: int) -> tuple[int, bool]:
        a, b = (u, v) if u < v else (v, u)
        idx = edge_map.get((a, b))
        if idx is None:
            idx = len(edges)
            edges.append([a, b])
            edge_map[(a, b)] = idx
        tail, head = edges[idx]
        return idx, (tail == u and head == v)

    def face_edges(v0: int, v1: int, v2: int) -> list:
        out: list = []
        for u, v in ((v0, v1), (v1, v2), (v2, v0)):
            ei, ok = get_edge(u, v)
            out.append(ei if ok else f"r{ei}")
        return out

    faces = [face_edges(*tri) for tri in triangles]

    return {
        "global_parameters": {
            "rim_slope_match_group": "rim",
            "rim_slope_match_outer_group": "outer",
            "rim_slope_match_disk_group": "disk",
            "rim_slope_match_center": [0.0, 0.0, 0.0],
            "rim_slope_match_normal": [0.0, 0.0, 1.0],
        },
        "energy_modules": [],
        "constraint_modules": [],
        "vertices": vertices,
        "edges": edges,
        "faces": faces,
    }


def test_rim_slope_match_theta_scalar_builds_inner_constraints():
    mesh = parse_geometry(_build_mesh())

    positions = mesh.positions_view()
    global_params = mesh.global_parameters
    global_params.set("rim_slope_match_thetaB_param", "tilt_thetaB_value")
    global_params.set("tilt_thetaB_value", 0.1)

    g_list = rim_slope_match_out.constraint_gradients_tilt_array(
        mesh,
        global_params,
        positions=positions,
        index_map=mesh.vertex_index_to_row,
        tilts_in=mesh.tilts_in_view(),
        tilts_out=mesh.tilts_out_view(),
    )

    assert g_list, "Expected rim slope constraints with theta_scalar enabled"

    # Ensure that at least one constraint includes a non-null inner-tilt gradient.
    has_inner = any(g_in is not None and np.any(g_in) for g_in, _ in g_list)
    assert has_inner, "Expected inner tilt constraints when theta_scalar is set"
