import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import parse_geometry
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.gradient_descent import GradientDescent

pytestmark = pytest.mark.e2e


def _build_mesh() -> dict:
    """Build a minimal bilayer-profile input without loading from `meshes/`."""

    def ring_vertices(r: float, *, n: int, z: float = 0.0) -> list[list[float]]:
        out: list[list[float]] = []
        for k in range(n):
            ang = 2.0 * np.pi * k / n
            out.append(
                [float(r) * float(np.cos(ang)), float(r) * float(np.sin(ang)), float(z)]
            )
        return out

    n = 12
    radii: list[tuple[float, dict | None, float]] = [
        (1.0 / 3.0, {"preset": "disk"}, 0.0),
        (2.0 / 3.0, {"preset": "disk", "rim_slope_match_group": "disk"}, 0.0),
        (1.0, {"preset": "rim"}, 0.0),
        (11.0 / 6.0, {"rim_slope_match_group": "outer"}, 0.001),
        (3.0, None, 0.0),
        (4.5, None, 0.0),
        (6.0, {"preset": "outer_rim"}, 0.0),
    ]

    vertices: list[list] = [
        [
            0.0,
            0.0,
            0.0,
            {
                "preset": "disk",
                "tilt_fixed_in": True,
                "tilt_fixed_out": True,
                "tilt_in": [0.0, 0.0, 0.0],
                "tilt_out": [0.0, 0.0, 0.0],
                "fixed": True,
            },
        ]
    ]
    ring_vids: list[list[int]] = []
    vid = 1
    for r, opts, z in radii:
        vids: list[int] = []
        for x, y, zc in ring_vertices(r, n=n, z=z):
            if opts is None:
                vertices.append([x, y, zc])
            else:
                vertices.append([x, y, zc, dict(opts)])
            vids.append(vid)
            vid += 1
        ring_vids.append(vids)

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

    faces: list[list] = []

    def add_tri(v0: int, v1: int, v2: int) -> None:
        faces.append(face_edges(v0, v1, v2))

    disk_inner = ring_vids[0]
    for k in range(n):
        add_tri(0, disk_inner[k], disk_inner[(k + 1) % n])
    for A, B in zip(ring_vids, ring_vids[1:]):
        for k in range(n):
            a0 = A[k]
            a1 = A[(k + 1) % n]
            b0 = B[k]
            b1 = B[(k + 1) % n]
            add_tri(a0, a1, b0)
            add_tri(b0, a1, b1)

    return {
        "global_parameters": {
            "surface_tension": 0.0,
            "bending_energy_model": "helfrich",
            "spontaneous_curvature": 0.0,
            "bending_modulus_in": 0.1,
            "bending_modulus_out": 0.1,
            "tilt_modulus_in": 1.0,
            "tilt_modulus_out": 1.0,
            "tilt_disk_target_group_in": "disk",
            "tilt_disk_target_strength_in": 200.0,
            "tilt_disk_target_group_out": "disk",
            "tilt_disk_target_strength_out": 200.0,
            "tilt_disk_target_theta_B": 1.0,
            "tilt_disk_target_lambda": 1.0,
            "tilt_disk_target_center": [0.0, 0.0, 0.0],
            "tilt_disk_target_normal": [0.0, 0.0, 1.0],
            "rim_slope_match_group": "rim",
            "rim_slope_match_outer_group": "outer",
            "rim_slope_match_disk_group": "disk",
            "rim_slope_match_strength": 0.0,
            "rim_slope_match_center": [0.0, 0.0, 0.0],
            "rim_slope_match_normal": [0.0, 0.0, 1.0],
            "tilt_solve_mode": "coupled",
            "tilt_step_size": 0.15,
            "tilt_inner_steps": 40,
            "tilt_tol": 1.0e-10,
            "step_size": 0.01,
            "step_size_mode": "fixed",
            "pin_to_plane_normal": [0.0, 0.0, 1.0],
            "pin_to_plane_point": [0.0, 0.0, 0.0],
        },
        "constraint_modules": ["pin_to_plane", "pin_to_circle", "rim_slope_match_out"],
        "definitions": {
            "disk": {
                "constraints": ["pin_to_plane"],
                "tilt_disk_target_group_in": "disk",
                "tilt_disk_target_group_out": "disk",
                "pin_to_plane_mode": "slide",
                "pin_to_plane_group": "disk_plane",
                "pin_to_plane_normal": [0.0, 0.0, 1.0],
                "pin_to_plane_point": [0.0, 0.0, 0.0],
            },
            "rim": {
                "constraints": ["pin_to_plane", "pin_to_circle"],
                "pin_to_plane_mode": "slide",
                "pin_to_plane_group": "disk_plane",
                "pin_to_plane_normal": [0.0, 0.0, 1.0],
                "pin_to_plane_point": [0.0, 0.0, 0.0],
                "pin_to_circle_group": "rim",
                "pin_to_circle_radius": 1.0,
                "pin_to_circle_normal": [0.0, 0.0, 1.0],
                "pin_to_circle_point": [0.0, 0.0, 0.0],
                "pin_to_circle_mode": "fixed",
                "rim_slope_match_group": "rim",
            },
            "outer_rim": {
                "constraints": ["pin_to_plane", "pin_to_circle"],
                "pin_to_plane_normal": [0.0, 0.0, 1.0],
                "pin_to_plane_point": [0.0, 0.0, 0.0],
                "pin_to_circle_group": "outer",
                "pin_to_circle_radius": 6.0,
                "pin_to_circle_normal": [0.0, 0.0, 1.0],
                "pin_to_circle_point": [0.0, 0.0, 0.0],
                "pin_to_circle_mode": "fixed",
                "tilt_fixed_in": True,
                "tilt_fixed_out": True,
            },
        },
        "energy_modules": [
            "bending_tilt_in",
            "bending_tilt_out",
            "tilt_in",
            "tilt_out",
            "tilt_smoothness_in",
            "tilt_smoothness_out",
            "tilt_disk_target_in",
            "tilt_disk_target_out",
            "rim_slope_match_out",
        ],
        "vertices": vertices,
        "edges": edges,
        "faces": faces,
    }


def _build_minimizer(mesh) -> Minimizer:
    return Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
    )


def _collect_group_rows(mesh, key: str, value: str) -> np.ndarray:
    rows: list[int] = []
    for vid in mesh.vertex_ids:
        opts = getattr(mesh.vertices[int(vid)], "options", None) or {}
        if opts.get(key) == value:
            rows.append(mesh.vertex_index_to_row[int(vid)])
    return np.asarray(rows, dtype=int)


def _radial_unit_vectors(positions: np.ndarray) -> np.ndarray:
    r = np.linalg.norm(positions[:, :2], axis=1)
    r_hat = np.zeros_like(positions)
    good = r > 1e-12
    r_hat[good, 0] = positions[good, 0] / r[good]
    r_hat[good, 1] = positions[good, 1] / r[good]
    return r_hat


def test_bilayer_profile_tilts_decay_in_outer_region() -> None:
    mesh = parse_geometry(_build_mesh())
    minim = _build_minimizer(mesh)
    minim.minimize(n_steps=60)

    positions = mesh.positions_view()
    r = np.linalg.norm(positions[:, :2], axis=1)
    rim_rows = _collect_group_rows(mesh, "rim_slope_match_group", "rim")
    assert rim_rows.size
    r_rim = float(np.mean(r[rim_rows]))

    # Exclude the rim itself; the asymmetric rim-matching condition allows
    # t_in != t_out at r=R even in the bilayer profile setup.
    mask = r >= (r_rim + 1e-3)
    rows = np.where(mask)[0]
    assert rows.size

    r_hat_outer = _radial_unit_vectors(positions[rows])
    theta_in_outer = np.einsum("ij,ij->i", mesh.tilts_in_view()[rows], r_hat_outer)
    theta_out_outer = np.einsum("ij,ij->i", mesh.tilts_out_view()[rows], r_hat_outer)

    inner_mask = r <= (r_rim + 1e-6)
    inner_rows = np.where(inner_mask)[0]
    assert inner_rows.size
    r_hat_inner = _radial_unit_vectors(positions[inner_rows])
    theta_in_inner = np.einsum(
        "ij,ij->i", mesh.tilts_in_view()[inner_rows], r_hat_inner
    )
    theta_out_inner = np.einsum(
        "ij,ij->i", mesh.tilts_out_view()[inner_rows], r_hat_inner
    )

    # Expect decay in both leaflets away from the disk.
    outer_in_p90 = float(np.quantile(np.abs(theta_in_outer), 0.9))
    outer_out_p90 = float(np.quantile(np.abs(theta_out_outer), 0.9))
    inner_in_p90 = float(np.quantile(np.abs(theta_in_inner), 0.9))
    inner_out_p90 = float(np.quantile(np.abs(theta_out_inner), 0.9))

    assert outer_in_p90 < 0.3 * (inner_in_p90 + 1e-12)
    assert outer_out_p90 < 0.3 * (inner_out_p90 + 1e-12)
