import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import parse_geometry
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.gradient_descent import GradientDescent


def _milestone_d_single_leaflet_hard_source_data(
    *, source_amplitude: float = 1.0
) -> dict:
    """Return a self-contained Milestone-D annulus benchmark geometry dict.

    Milestone D aims to demonstrate *indirect* leaflet coupling: a hard tilt
    boundary condition applied to only one leaflet induces tilt in the opposite
    leaflet via a shared, free midplane and the `bending_tilt_in/out` modules.
    """
    n = 8
    r_in = 1.0
    r_mid = 2.0
    r_out = 4.0

    vertices: list[list] = []

    definitions = {
        "inner_rim": {
            "constraints": ["pin_to_circle"],
            "pin_to_circle_group": "inner",
            "pin_to_circle_radius": r_in,
            "pin_to_circle_normal": [0.0, 0.0, 1.0],
            "pin_to_circle_point": [0.0, 0.0, 0.0],
            # The inner rim is a rigid ring that can translate/tilt in z so the
            # membrane can kink to relax the imposed tilt boundary condition.
            "pin_to_circle_mode": "fit",
            "tilt_fixed_in": True,
        },
        "outer_rim": {
            "constraints": ["pin_to_circle", "pin_to_plane"],
            "pin_to_circle_group": "outer",
            "pin_to_circle_radius": r_out,
            "pin_to_circle_normal": [0.0, 0.0, 1.0],
            "pin_to_circle_point": [0.0, 0.0, 0.0],
            "pin_to_circle_mode": "fixed",
            # Clamp far-field tilts in both leaflets.
            "tilt_fixed_in": True,
            "tilt_fixed_out": True,
        },
    }

    # Vertices are arranged CCW for +z triangle normals on the initial flat mesh.
    for i in range(n):
        theta = 2.0 * np.pi * i / n
        vertices.append(
            [
                r_in * np.cos(theta),
                r_in * np.sin(theta),
                0.0,
                {
                    "preset": "inner_rim",
                    # Hard source: clamp tilt only in the inner leaflet to the
                    # in-plane radial direction.
                    "tilt_in": [
                        float(source_amplitude * np.cos(theta)),
                        float(source_amplitude * np.sin(theta)),
                    ],
                },
            ]
        )
    for i in range(n):
        theta = 2.0 * np.pi * i / n
        vertices.append([r_mid * np.cos(theta), r_mid * np.sin(theta), 0.0])
    for i in range(n):
        theta = 2.0 * np.pi * i / n
        vertices.append(
            [
                r_out * np.cos(theta),
                r_out * np.sin(theta),
                0.0,
                {"preset": "outer_rim", "tilt_in": [0.0, 0.0], "tilt_out": [0.0, 0.0]},
            ]
        )

    edges: list[list] = []
    for base in (0, 8, 16):
        for i in range(n):
            edges.append([base + i, base + ((i + 1) % n)])

    for i in range(n):
        edges.append([i, 8 + i])
        edges.append([8 + i, 16 + i])

    for i in range(n):
        edges.append([i, 8 + ((i + 1) % n)])
        edges.append([8 + i, 16 + ((i + 1) % n)])

    edge_index_by_pair: dict[tuple[int, int], int] = {}
    for idx, (tail, head, *_rest) in enumerate(edges):
        edge_index_by_pair[(int(tail), int(head))] = int(idx)

    def edge_ref(tail: int, head: int) -> int | str:
        """Return an edge reference compatible with `parse_geometry` faces."""
        forward = edge_index_by_pair.get((int(tail), int(head)))
        if forward is not None:
            return forward
        reverse = edge_index_by_pair.get((int(head), int(tail)))
        if reverse is not None:
            return f"r{reverse}"
        raise KeyError(f"Missing edge for face: {tail}->{head}")

    faces: list[list] = []
    for i in range(n):
        i1 = (i + 1) % n
        v_i, v_i1 = i, i1
        m_i, m_i1 = 8 + i, 8 + i1
        o_i, o_i1 = 16 + i, 16 + i1

        # Inner ↔ mid quad.
        faces.append([edge_ref(v_i, v_i1), edge_ref(v_i1, m_i1), edge_ref(m_i1, v_i)])
        faces.append([edge_ref(v_i, m_i1), edge_ref(m_i1, m_i), edge_ref(m_i, v_i)])

        # Mid ↔ outer quad.
        faces.append([edge_ref(m_i, m_i1), edge_ref(m_i1, o_i1), edge_ref(o_i1, m_i)])
        faces.append([edge_ref(m_i, o_i1), edge_ref(o_i1, o_i), edge_ref(o_i, m_i)])

    return {
        "global_parameters": {
            "surface_tension": 0.0,
            "bending_energy_model": "helfrich",
            "spontaneous_curvature": 0.0,
            # Parameter regime chosen to make the induced opposite-leaflet tilt
            # clearly measurable on a coarse annulus.
            "bending_modulus_in": 10.0,
            "bending_modulus_out": 10.0,
            "tilt_modulus_in": 1.0,
            "tilt_modulus_out": 0.1,
            "tilt_solve_mode": "nested",
            "tilt_step_size": 0.15,
            "tilt_inner_steps": 15,
            "tilt_tol": 1e-10,
            # Allow midplane relaxation so leaflet coupling can act through shape.
            "step_size": 0.006,
            "pin_to_plane_normal": [0.0, 0.0, 1.0],
            "pin_to_plane_point": [0.0, 0.0, 0.0],
        },
        "constraint_modules": ["pin_to_plane", "pin_to_circle"],
        "definitions": definitions,
        "energy_modules": [
            "bending_tilt_in",
            "bending_tilt_out",
            "tilt_in",
            "tilt_out",
        ],
        "vertices": vertices,
        "edges": edges,
        "faces": faces,
    }


def _build_minimizer(mesh) -> Minimizer:
    """Return a minimizer instance for the provided mesh."""
    return Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
    )


def _break_up_down_symmetry(mesh, *, z_bump: float = 1e-3, target_radius: float = 2.0):
    """Perturb a mid-radius vertex in z to avoid the flat saddle-point solution."""
    mesh.increment_version()
    mesh._positions_cache = None
    mesh._positions_cache_version = -1
    positions = mesh.positions_view()
    radii = np.linalg.norm(positions[:, :2], axis=1)
    mid_row = int(np.argmin(np.abs(radii - float(target_radius))))
    mesh.vertices[int(mesh.vertex_ids[mid_row])].position[2] = float(z_bump)
    mesh.increment_version()
    mesh._positions_cache = None
    mesh._positions_cache_version = -1


def test_milestone_d_hard_in_source_induces_opposite_leaflet_tilt() -> None:
    """Milestone D: hard inner-leaflet source induces outer-leaflet tilt via curvature."""
    mesh = parse_geometry(_milestone_d_single_leaflet_hard_source_data())
    _break_up_down_symmetry(mesh)

    minim = _build_minimizer(mesh)
    minim.minimize(n_steps=10)

    t_in = mesh.tilts_in_view()
    t_out = mesh.tilts_out_view()
    max_in = float(np.max(np.linalg.norm(t_in, axis=1)))
    max_out = float(np.max(np.linalg.norm(t_out, axis=1)))
    assert max_in > 0.5
    assert max_out / max_in > 0.5

    z = mesh.positions_view()[:, 2]
    assert float(np.ptp(z)) > 1e-2


def test_milestone_d_without_bending_tilt_out_keeps_opposite_leaflet_zeroish() -> None:
    """Milestone D control: without bending_tilt_out there is no shape-mediated coupling."""
    data = _milestone_d_single_leaflet_hard_source_data()
    data["energy_modules"] = [
        m for m in data["energy_modules"] if m != "bending_tilt_out"
    ]
    mesh = parse_geometry(data)
    _break_up_down_symmetry(mesh)

    minim = _build_minimizer(mesh)
    minim.minimize(n_steps=10)

    t_out = mesh.tilts_out_view()
    assert float(np.max(np.linalg.norm(t_out, axis=1))) < 1e-6
