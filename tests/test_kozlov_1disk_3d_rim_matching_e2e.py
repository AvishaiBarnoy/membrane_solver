import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import parse_geometry
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.gradient_descent import GradientDescent


def _disk_outer_rim_match_data(
    *,
    n_theta: int = 12,
    n_disk_rings: int = 3,
    n_outer_rings: int = 6,
    r_disk: float = 1.0,
    r_out: float = 6.0,
    source_strength: float = 1.0,
    match_strength: float = 200.0,
    z_bump: float = 1e-3,
) -> dict:
    """Return a disk+outer setup for rim-matching (γ=0)."""
    disk_radii = np.linspace(0.0, float(r_disk), int(n_disk_rings) + 1)[1:]
    outer_radii = np.linspace(float(r_disk), float(r_out), int(n_outer_rings) + 1)[1:]
    radii = np.concatenate([disk_radii, outer_radii], axis=0)

    rim_ring = int(n_disk_rings)
    outer_ring = int(n_disk_rings + n_outer_rings)

    definitions = {
        "disk": {
            "constraints": ["pin_to_plane"],
            "pin_to_plane_normal": [0.0, 0.0, 1.0],
            "pin_to_plane_point": [0.0, 0.0, 0.0],
        },
        "rim": {
            "constraints": ["pin_to_plane", "pin_to_circle"],
            "pin_to_plane_normal": [0.0, 0.0, 1.0],
            "pin_to_plane_point": [0.0, 0.0, 0.0],
            "pin_to_circle_group": "rim",
            "pin_to_circle_radius": float(r_disk),
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
            "pin_to_circle_radius": float(r_out),
            "pin_to_circle_normal": [0.0, 0.0, 1.0],
            "pin_to_circle_point": [0.0, 0.0, 0.0],
            "pin_to_circle_mode": "fixed",
            "tilt_fixed_in": True,
            "tilt_fixed_out": True,
        },
    }

    vertices: list[list] = []
    # Center vertex (regularity).
    vertices.append(
        [
            0.0,
            0.0,
            0.0,
            {
                "preset": "disk",
                "tilt_fixed_in": True,
                "tilt_fixed_out": True,
                "tilt_in": [0.0, 0.0],
                "tilt_out": [0.0, 0.0],
                "fixed": True,
            },
        ]
    )

    for ring_idx, r in enumerate(radii, start=1):
        for i in range(n_theta):
            theta = 2.0 * np.pi * i / float(n_theta)
            x = float(r * np.cos(theta))
            y = float(r * np.sin(theta))
            z = 0.0
            opts: dict | None = None

            if ring_idx < rim_ring:
                opts = {"preset": "disk"}
                if ring_idx == rim_ring - 1:
                    opts["rim_slope_match_group"] = "disk"
            elif ring_idx == rim_ring:
                opts = {"preset": "rim"}
            elif ring_idx == rim_ring + 1:
                z = float(z_bump)
                opts = {"rim_slope_match_group": "outer"}
            elif ring_idx == outer_ring:
                opts = {
                    "preset": "outer_rim",
                    "tilt_in": [0.0, 0.0],
                    "tilt_out": [0.0, 0.0],
                }

            if opts is None:
                vertices.append([x, y, z])
            else:
                vertices.append([x, y, z, opts])

    def vid(ring: int, k: int) -> int:
        if ring == 0:
            return 0
        return 1 + (int(ring) - 1) * int(n_theta) + int(k)

    edges: list[list[int]] = []
    for ring in range(1, outer_ring + 1):
        for k in range(n_theta):
            edges.append([vid(ring, k), vid(ring, (k + 1) % n_theta)])
    for ring in range(1, outer_ring):
        for k in range(n_theta):
            edges.append([vid(ring, k), vid(ring + 1, k)])
    for ring in range(1, outer_ring):
        for k in range(n_theta):
            edges.append([vid(ring, k), vid(ring + 1, (k + 1) % n_theta)])
    for k in range(n_theta):
        edges.append([0, vid(1, k)])

    edge_index_by_pair: dict[tuple[int, int], int] = {}
    for idx, (tail, head, *_rest) in enumerate(edges):
        edge_index_by_pair[(int(tail), int(head))] = int(idx)

    def edge_ref(tail: int, head: int) -> int | str:
        forward = edge_index_by_pair.get((int(tail), int(head)))
        if forward is not None:
            return forward
        reverse = edge_index_by_pair.get((int(head), int(tail)))
        if reverse is not None:
            return f"r{reverse}"
        raise KeyError(f"Missing edge for face: {tail}->{head}")

    faces: list[list] = []
    for k in range(n_theta):
        k1 = (k + 1) % n_theta
        faces.append(
            [
                edge_ref(0, vid(1, k)),
                edge_ref(vid(1, k), vid(1, k1)),
                edge_ref(vid(1, k1), 0),
            ]
        )
    for ring in range(1, outer_ring):
        for k in range(n_theta):
            k1 = (k + 1) % n_theta
            v00 = vid(ring, k)
            v01 = vid(ring, k1)
            v10 = vid(ring + 1, k)
            v11 = vid(ring + 1, k1)
            faces.append([edge_ref(v00, v01), edge_ref(v01, v11), edge_ref(v11, v00)])
            faces.append([edge_ref(v00, v11), edge_ref(v11, v10), edge_ref(v10, v00)])

    return {
        "global_parameters": {
            "surface_tension": 0.0,
            "bending_energy_model": "helfrich",
            "spontaneous_curvature": 0.0,
            "bending_modulus_in": 2.0,
            "bending_modulus_out": 2.0,
            "tilt_modulus_in": 2.0,
            "tilt_modulus_out": 2.0,
            "tilt_rim_source_center": [0.0, 0.0, 0.0],
            "tilt_rim_source_edge_mode": "all",
            "tilt_rim_source_group": "rim",
            "tilt_rim_source_strength": float(source_strength),
            "rim_slope_match_group": "rim",
            "rim_slope_match_outer_group": "outer",
            "rim_slope_match_disk_group": "disk",
            "rim_slope_match_strength": float(match_strength),
            "rim_slope_match_center": [0.0, 0.0, 0.0],
            "rim_slope_match_normal": [0.0, 0.0, 1.0],
            "tilt_solve_mode": "nested",
            "tilt_step_size": 0.15,
            "tilt_inner_steps": 40,
            "tilt_tol": 1e-10,
            "step_size": 0.01,
            "step_size_mode": "fixed",
        },
        "constraint_modules": ["pin_to_plane", "pin_to_circle"],
        "definitions": definitions,
        "energy_modules": [
            "bending_tilt_in",
            "bending_tilt_out",
            "tilt_in",
            "tilt_out",
            "tilt_rim_source_bilayer",
            "rim_slope_match_out",
        ],
        "vertices": vertices,
        "edges": edges,
        "faces": faces,
        "instructions": [],
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


def _rim_slope_stats(
    mesh, *, n_theta: int, rim_ring: int
) -> tuple[float, float, float, float]:
    start = 1 + (int(rim_ring) - 1) * int(n_theta)
    rim_rows = np.arange(start, start + int(n_theta), dtype=int)
    outer_rows = rim_rows + int(n_theta)
    disk_rows = rim_rows - int(n_theta)
    pos = mesh.positions_view()
    r_rim = np.linalg.norm(pos[rim_rows, :2], axis=1)
    r_out = np.linalg.norm(pos[outer_rows, :2], axis=1)
    z_rim = pos[rim_rows, 2]
    z_out = pos[outer_rows, 2]
    dr = np.maximum(r_out - r_rim, 1e-6)
    phi = np.mean((z_out - z_rim) / dr)

    r_hat = np.zeros_like(pos[rim_rows])
    r_hat[:, 0] = pos[rim_rows, 0] / r_rim
    r_hat[:, 1] = pos[rim_rows, 1] / r_rim
    t_out = mesh.tilts_out_view()[rim_rows]
    theta_out = float(np.mean(np.einsum("ij,ij->i", t_out, r_hat)))
    t_in_rim = mesh.tilts_in_view()[rim_rows]
    theta_in_rim = float(np.mean(np.einsum("ij,ij->i", t_in_rim, r_hat)))
    t_in_disk = mesh.tilts_in_view()[disk_rows]
    theta_disk = float(np.mean(np.einsum("ij,ij->i", t_in_disk, r_hat)))
    return float(phi), float(theta_out), float(theta_disk), float(theta_in_rim)


def test_kozlov_1disk_3d_rim_matching_drives_outer_slope() -> None:
    mesh = parse_geometry(_disk_outer_rim_match_data())
    minim = _build_minimizer(mesh)
    minim.minimize(n_steps=30)

    z = mesh.positions_view()[:, 2]
    assert float(np.ptp(z)) > 1e-3

    phi, theta_out, theta_disk, theta_in_rim = _rim_slope_stats(
        mesh, n_theta=12, rim_ring=3
    )
    denom = max(abs(phi), abs(theta_out), 1e-6)
    assert abs(phi - theta_out) / denom < 0.4

    denom_in = max(abs(theta_disk - phi), abs(theta_in_rim), 1e-6)
    assert abs(theta_in_rim - (theta_disk - phi)) / denom_in < 0.4
