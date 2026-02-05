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
    """Build a minimal single-leaflet-profile input without loading from `meshes/`."""

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
            "bending_modulus_in": 2.0,
            "bending_modulus_out": 2.0,
            "tilt_modulus_in": 2.0,
            "tilt_modulus_out": 2.0,
            "tilt_disk_target_group_in": "disk",
            "tilt_disk_target_strength_in": 50.0,
            "tilt_disk_target_theta_B": 1.0,
            "tilt_disk_target_lambda": 1.0,
            "tilt_disk_target_center": [0.0, 0.0, 0.0],
            "tilt_disk_target_normal": [0.0, 0.0, 1.0],
            "rim_slope_match_group": "rim",
            "rim_slope_match_outer_group": "outer",
            "rim_slope_match_disk_group": "disk",
            "rim_slope_match_strength": 200.0,
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
        "constraint_modules": ["pin_to_plane", "pin_to_circle"],
        "definitions": {
            "disk": {
                "constraints": ["pin_to_plane"],
                "tilt_disk_target_group_in": "disk",
                "pin_to_plane_normal": [0.0, 0.0, 1.0],
                "pin_to_plane_point": [0.0, 0.0, 0.0],
            },
            "rim": {
                "constraints": ["pin_to_plane", "pin_to_circle"],
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


def _order_by_angle(positions: np.ndarray) -> np.ndarray:
    angles = np.arctan2(positions[:, 1], positions[:, 0])
    return np.argsort(angles)


def _radial_unit_vectors(positions: np.ndarray) -> np.ndarray:
    r = np.linalg.norm(positions[:, :2], axis=1)
    r_hat = np.zeros_like(positions)
    good = r > 1e-12
    r_hat[good, 0] = positions[good, 0] / r[good]
    r_hat[good, 1] = positions[good, 1] / r[good]
    return r_hat


def _outer_free_ring_rows(mesh, positions: np.ndarray) -> np.ndarray:
    rows: list[int] = []
    radii: list[float] = []
    for vid in mesh.vertex_ids:
        v = mesh.vertices[int(vid)]
        opts = getattr(v, "options", None) or {}
        if opts.get("pin_to_circle_group") == "outer":
            continue
        row = mesh.vertex_index_to_row[int(vid)]
        rows.append(row)
        radii.append(float(np.linalg.norm(positions[row, :2])))
    if not rows:
        return np.zeros(0, dtype=int)
    radii_arr = np.asarray(radii, dtype=float)
    r_max = float(np.max(radii_arr))
    tol = 1e-6
    rows_arr = np.asarray(rows, dtype=int)
    return rows_arr[np.abs(radii_arr - r_max) <= tol]


def test_single_leaflet_profile_behavior() -> None:
    mesh = parse_geometry(_build_mesh())
    minim = _build_minimizer(mesh)
    minim.minimize(n_steps=60)

    positions = mesh.positions_view()
    z_span = float(np.ptp(positions[:, 2]))
    assert z_span > 1e-4

    disk_rows = _collect_group_rows(mesh, "tilt_disk_target_group_in", "disk")
    assert disk_rows.size

    rim_rows = _collect_group_rows(mesh, "rim_slope_match_group", "rim")
    outer_rows = _collect_group_rows(mesh, "rim_slope_match_group", "outer")
    disk_ring_rows = _collect_group_rows(mesh, "rim_slope_match_group", "disk")
    assert rim_rows.size and outer_rows.size and disk_ring_rows.size

    rim_rows = rim_rows[_order_by_angle(positions[rim_rows])]
    outer_rows = outer_rows[_order_by_angle(positions[outer_rows])]

    rim_pos = positions[rim_rows]
    outer_pos = positions[outer_rows]
    r_rim = np.linalg.norm(rim_pos[:, :2], axis=1)
    r_outer = np.linalg.norm(outer_pos[:, :2], axis=1)
    dr = np.maximum(r_outer - r_rim, 1e-6)
    phi = float(np.mean((outer_pos[:, 2] - rim_pos[:, 2]) / dr))
    assert abs(phi) > 1e-4

    r_disk = np.linalg.norm(positions[disk_rows, :2], axis=1)
    r_max = float(np.max(r_disk))
    r_hat_disk = _radial_unit_vectors(positions[disk_rows])
    theta_disk = np.einsum(
        "ij,ij->i",
        mesh.tilts_in_view()[disk_rows],
        r_hat_disk,
    )
    inner_band = theta_disk[r_disk < 0.4 * r_max]
    outer_band = theta_disk[r_disk > 0.8 * r_max]
    assert float(np.mean(outer_band)) > float(np.mean(inner_band))

    rim_r_hat = _radial_unit_vectors(rim_pos)
    theta_in_rim = float(
        np.mean(np.einsum("ij,ij->i", mesh.tilts_in_view()[rim_rows], rim_r_hat))
    )
    theta_out_rim = float(
        np.mean(np.einsum("ij,ij->i", mesh.tilts_out_view()[rim_rows], rim_r_hat))
    )
    disk_ring_r_hat = _radial_unit_vectors(positions[disk_ring_rows])
    theta_disk_ring = float(
        np.mean(
            np.einsum(
                "ij,ij->i",
                mesh.tilts_in_view()[disk_ring_rows],
                disk_ring_r_hat,
            )
        )
    )
    denom = max(abs(theta_in_rim), abs(theta_disk_ring - phi), 1e-6)
    assert abs(theta_in_rim - (theta_disk_ring - phi)) / denom < 0.6
    assert abs(theta_out_rim) > 1e-4

    free_rows = _outer_free_ring_rows(mesh, positions)
    assert free_rows.size
    free_r_hat = _radial_unit_vectors(positions[free_rows])
    theta_out_far = float(
        np.mean(
            np.abs(
                np.einsum(
                    "ij,ij->i",
                    mesh.tilts_out_view()[free_rows],
                    free_r_hat,
                )
            )
        )
    )
    assert theta_out_far < 0.7 * abs(theta_out_rim)
