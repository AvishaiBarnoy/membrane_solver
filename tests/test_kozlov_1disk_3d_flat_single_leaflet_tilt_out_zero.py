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

pytestmark = pytest.mark.regression


def _build_mesh() -> dict:
    def ring_vertices(r: float, *, n: int, z: float = 0.0) -> list[list[float]]:
        out: list[list[float]] = []
        for k in range(n):
            ang = 2.0 * np.pi * k / n
            out.append(
                [float(r) * float(np.cos(ang)), float(r) * float(np.sin(ang)), float(z)]
            )
        return out

    n = 20
    r_disk_inner = 0.5
    r_disk = 1.0
    radii: list[tuple[float, dict | None]] = [
        (r_disk_inner, {"preset": "disk"}),
        (r_disk, {"preset": "disk"}),
        (2.0, None),
        (3.0, None),
        (4.0, {"preset": "outer_rim"}),
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
                "constraints": ["pin_to_plane"],
            },
        ]
    ]
    ring_vids: list[list[int]] = []
    vid = 1
    for r, opts in radii:
        vids: list[int] = []
        for x, y, zc in ring_vertices(r, n=n, z=0.0):
            if opts is None:
                vertices.append([x, y, zc, {"constraints": ["pin_to_plane"]}])
            else:
                entry = dict(opts)
                entry["constraints"] = ["pin_to_plane"]
                vertices.append([x, y, zc, entry])
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
            "tilt_modulus_in": 2.0,
            "tilt_modulus_out": 2.0,
            "tilt_disk_target_group_in": "disk",
            "tilt_disk_target_strength_in": 50.0,
            "tilt_disk_target_theta_B": 0.8,
            "tilt_disk_target_lambda": 1.0,
            "tilt_disk_target_center": [0.0, 0.0, 0.0],
            "tilt_disk_target_normal": [0.0, 0.0, 1.0],
            "tilt_solve_mode": "nested",
            "tilt_step_size": 0.1,
            "tilt_inner_steps": 200,
            "tilt_tol": 1.0e-10,
            "step_size": 0.0,
            "step_size_mode": "fixed",
            "pin_to_plane_normal": [0.0, 0.0, 1.0],
            "pin_to_plane_point": [0.0, 0.0, 0.0],
        },
        "constraint_modules": ["pin_to_plane"],
        "definitions": {
            "disk": {
                "constraints": ["pin_to_plane"],
                "tilt_disk_target_group_in": "disk",
                "pin_to_plane_normal": [0.0, 0.0, 1.0],
                "pin_to_plane_point": [0.0, 0.0, 0.0],
            },
            "outer_rim": {
                "constraints": ["pin_to_plane"],
                "pin_to_plane_normal": [0.0, 0.0, 1.0],
                "pin_to_plane_point": [0.0, 0.0, 0.0],
                "tilt_fixed_in": True,
                "tilt_fixed_out": True,
                "tilt_in": [0.0, 0.0, 0.0],
                "tilt_out": [0.0, 0.0, 0.0],
            },
        },
        "energy_modules": [
            "tilt_in",
            "tilt_out",
            "tilt_smoothness_in",
            "tilt_smoothness_out",
            "tilt_disk_target_in",
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


def test_flat_disk_single_leaflet_tilt_out_stays_zero() -> None:
    mesh = parse_geometry(_build_mesh())
    minim = _build_minimizer(mesh)
    minim.minimize(n_steps=1)

    positions = mesh.positions_view()
    z_span = float(np.ptp(positions[:, 2]))
    assert z_span < 1e-12

    r = np.linalg.norm(positions[:, :2], axis=1)
    r_hat = np.zeros_like(positions)
    good = r > 1e-12
    r_hat[good, 0] = positions[good, 0] / r[good]
    r_hat[good, 1] = positions[good, 1] / r[good]

    t_in = mesh.tilts_in_view()
    t_out = mesh.tilts_out_view()

    t_in_rad = np.einsum("ij,ij->i", t_in, r_hat)
    t_out_rad = np.einsum("ij,ij->i", t_out, r_hat)

    max_in = float(np.max(np.abs(t_in_rad)))
    max_out = float(np.max(np.abs(t_out_rad)))
    assert max_in > 1e-3
    assert max_out < 1e-2 * max_in

    r_disk = 1.0
    inner = r < 0.6 * r_disk
    near_rim = (r > 0.9 * r_disk) & (r < 1.1 * r_disk)
    outer_band = (r > 2.5 * r_disk) & (r < 3.5 * r_disk)

    inner_med = float(np.median(np.abs(t_in_rad[inner])))
    rim_med = float(np.median(np.abs(t_in_rad[near_rim])))
    outer_med = float(np.median(np.abs(t_in_rad[outer_band])))

    assert rim_med > inner_med
    assert outer_med < 0.7 * rim_med
