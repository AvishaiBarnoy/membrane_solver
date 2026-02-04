import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.curvature import compute_curvature_data  # noqa: E402
from geometry.geom_io import parse_geometry  # noqa: E402
from runtime.constraint_manager import ConstraintModuleManager  # noqa: E402
from runtime.energy_manager import EnergyModuleManager  # noqa: E402
from runtime.minimizer import Minimizer  # noqa: E402
from runtime.steppers.gradient_descent import GradientDescent  # noqa: E402


def _ring_vertices(r: float, *, n: int) -> list[list[float]]:
    out: list[list[float]] = []
    for k in range(n):
        ang = 2.0 * math.pi * k / n
        out.append([float(r) * math.cos(ang), float(r) * math.sin(ang), 0.0])
    return out


def _build_kozlov_free_disk_mesh_dict() -> dict:
    """Return a minimal free-disk ring mesh dict, independent of `meshes/`.

    This matches the repaired triangulation topology: concentric rings with
    consistent edge indexing (no planar edge crossings).
    """
    n = 12
    r_disk_inner = 4.0 / 15.0
    r_disk_edge = 7.0 / 15.0
    r_rim = 1.0
    r_mid = [
        2.833333333333334,
        4.666666666666668,
        6.5,
        8.333333333333336,
        10.166666666666668,
    ]
    r_outer = 12.0

    vertices: list[list] = []
    vertices.append([0.0, 0.0, 0.0, {"preset": "disk"}])

    # Ring ordering is not physically important for this diagnostic, but keep it
    # consistent with our repaired mesh generator: disk edge, disk inner, rim,
    # intermediate rings, outer rim.
    rings: list[tuple[float, dict | None]] = [
        (r_disk_edge, {"preset": "disk", "rim_slope_match_group": "disk"}),
        (r_disk_inner, {"preset": "disk"}),
        (r_rim, {"preset": "rim"}),
    ]
    rings.extend((r, None) for r in r_mid)
    rings.append((r_outer, {"preset": "outer_rim"}))

    ring_vids: list[list[int]] = []
    vid = 1
    for r, opts in rings:
        vids: list[int] = []
        for x, y, z in _ring_vertices(r, n=n):
            if opts is None:
                vertices.append([x, y, z])
            else:
                vertices.append([x, y, z, dict(opts)])
            vids.append(vid)
            vid += 1
        ring_vids.append(vids)

    edges: list[list[int]] = []
    edge_map: dict[tuple[int, int], int] = {}

    def get_edge(u: int, v: int):
        a, b = (u, v) if u < v else (v, u)
        key = (a, b)
        idx = edge_map.get(key)
        if idx is None:
            idx = len(edges)
            edges.append([a, b])
            edge_map[key] = idx
        stored_tail, stored_head = edges[idx]
        return idx, (stored_tail == u and stored_head == v)

    def face_edges(v0: int, v1: int, v2: int) -> list:
        out: list = []
        for u, v in ((v0, v1), (v1, v2), (v2, v0)):
            ei, ok = get_edge(u, v)
            out.append(ei if ok else f"r{ei}")
        return out

    faces: list[list] = []

    def add_tri(v0: int, v1: int, v2: int) -> None:
        faces.append(face_edges(v0, v1, v2))

    # Center fan to disk inner ring.
    disk_inner = ring_vids[1]
    for k in range(n):
        add_tri(0, disk_inner[k], disk_inner[(k + 1) % n])

    # Annuli between successive rings (disk inner -> disk edge -> rim -> ... -> outer).
    ordered_rings = [ring_vids[1], ring_vids[0], ring_vids[2]] + ring_vids[3:]
    for A, B in zip(ordered_rings, ordered_rings[1:]):
        for k in range(n):
            a0 = A[k]
            a1 = A[(k + 1) % n]
            b0 = B[k]
            b1 = B[(k + 1) % n]
            add_tri(a0, a1, b0)
            add_tri(b0, a1, b1)

    return {
        "global_parameters": {
            "bending_energy_model": "helfrich",
            "bending_modulus_in": 2.0,
            "bending_modulus_out": 2.0,
            "tilt_modulus_in": 450.0,
            "tilt_modulus_out": 450.0,
            "spontaneous_curvature": 0.0,
        },
        "definitions": {
            # Minimal preset stubs so `parse_geometry` can resolve `preset:` tags.
            "disk": {"constraints": []},
            "rim": {"constraints": []},
            "outer_rim": {"constraints": []},
        },
        "energy_modules": [
            "bending_tilt_in",
            "bending_tilt_out",
            "tilt_in",
            "tilt_out",
        ],
        "constraint_modules": [],
        "vertices": vertices,
        "edges": edges,
        "faces": faces,
    }


@pytest.mark.e2e
def test_kozlov_free_disk_flat_state_has_large_boundary_curvature_baseline() -> None:
    """Diagnostic regression for the flat-reference state vs docs/tex/1_disk_3d.tex.

    In the continuum theory, a perfectly flat membrane patch with zero tilt has
    zero elastic energy. On the current discrete free-disk mesh, the curvature
    operator assigns a nonzero mean-curvature proxy on the *open boundary* even
    when all vertices are coplanar.

    This test pins down the diagnostic facts (boundary dominance) so the follow-up
    physics work can stay honest about what the discrete operators do in a flat
    state.
    """
    mesh = parse_geometry(_build_kozlov_free_disk_mesh_dict())
    mesh.build_connectivity_maps()
    mesh.build_facet_vertex_loops()
    mesh.build_position_cache()

    # Force a flat, zero-tilt reference state.
    for vertex in mesh.vertices.values():
        vertex.position[2] = 0.0
    mesh.increment_version()

    n = len(mesh.vertex_ids)
    mesh.set_tilts_in_from_array(np.zeros((n, 3), dtype=float))
    mesh.set_tilts_out_from_array(np.zeros((n, 3), dtype=float))
    mesh.global_parameters.set("tilt_thetaB_value", 0.0)

    minim = Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
    )
    breakdown = minim.compute_energy_breakdown()

    # In the continuum theory, flat + zero tilt must have ~0 elastic energy.
    # If this ever becomes nontrivial again, it is a regression in the
    # flat-reference behavior we aim to match.
    assert float(breakdown.get("bending_tilt_in") or 0.0) < 1.0e-8
    assert float(breakdown.get("bending_tilt_out") or 0.0) < 1.0e-8

    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row
    k_vecs, _areas, _weights, _tri_rows = compute_curvature_data(
        mesh, positions, index_map
    )
    k_mag = np.linalg.norm(k_vecs, axis=1)

    boundary_vids = sorted(mesh.boundary_vertex_ids)
    assert boundary_vids, "expected an open boundary loop on this mesh"
    boundary_rows = np.asarray(
        [index_map[int(vid)] for vid in boundary_vids], dtype=int
    )

    total = float(np.sum(k_mag))
    boundary = float(np.sum(k_mag[boundary_rows]))
    share = boundary / total if total > 0.0 else 0.0

    # Diagnostic fact: the boundary dominates the curvature baseline.
    assert share > 0.85

    # Interior curvature should be comparatively small for the coplanar surface.
    interior_mask = np.ones_like(k_mag, dtype=bool)
    interior_mask[boundary_rows] = False
    assert float(np.max(k_mag[interior_mask])) < 1.0
