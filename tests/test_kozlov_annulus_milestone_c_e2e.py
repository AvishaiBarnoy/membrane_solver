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


def _milestone_c_soft_source_data(*, rim_source_strength: float = 25.0) -> dict:
    """Return a self-contained Milestone-C annulus benchmark geometry dict.

    This intentionally does *not* load from `meshes/` so unit tests are robust
    to users iterating on YAML benchmark files.
    """
    n = 8
    r_in = 1.0
    r_mid = 2.0
    r_out = 3.0

    vertices: list[list] = []

    # Inner rim: pinned to a circle but allowed to translate in z as a ring by
    # using `pin_to_circle_mode: fit` (outer rim stays fixed and pinned to z=0).
    definitions = {
        "inner_rim": {
            "constraints": ["pin_to_circle"],
            "pin_to_circle_group": "inner",
            "pin_to_circle_radius": r_in,
            "pin_to_circle_normal": [0.0, 0.0, 1.0],
            "pin_to_circle_point": [0.0, 0.0, 0.0],
            "pin_to_circle_mode": "fit",
        },
        "outer_rim": {
            "constraints": ["pin_to_circle", "pin_to_plane"],
            "pin_to_circle_group": "outer",
            "pin_to_circle_radius": r_out,
            "pin_to_circle_normal": [0.0, 0.0, 1.0],
            "pin_to_circle_point": [0.0, 0.0, 0.0],
            "pin_to_circle_mode": "fixed",
            "tilt_fixed_in": True,
            "tilt_fixed_out": True,
        },
    }

    # Vertices are arranged CCW for +z triangle normals on the initial flat mesh.
    for i in range(n):
        theta = 2.0 * np.pi * i / n
        vertices.append(
            [r_in * np.cos(theta), r_in * np.sin(theta), 0.0, {"preset": "inner_rim"}]
        )
    for i in range(n):
        theta = 2.0 * np.pi * i / n
        vertices.append(
            [
                r_mid * np.cos(theta),
                r_mid * np.sin(theta),
                0.0,
                {"ring": "mid"},
            ]
        )
    for i in range(n):
        theta = 2.0 * np.pi * i / n
        vertices.append(
            [
                r_out * np.cos(theta),
                r_out * np.sin(theta),
                0.0,
                {
                    "preset": "outer_rim",
                    "tilt_in": [0.0, 0.0],
                    "tilt_out": [0.0, 0.0],
                },
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
        faces.append(
            [
                edge_ref(v_i, v_i1),
                edge_ref(v_i1, m_i1),
                edge_ref(m_i1, v_i),
            ]
        )
        faces.append(
            [
                edge_ref(v_i, m_i1),
                edge_ref(m_i1, m_i),
                edge_ref(m_i, v_i),
            ]
        )

        # Mid ↔ outer quad.
        faces.append(
            [
                edge_ref(m_i, m_i1),
                edge_ref(m_i1, o_i1),
                edge_ref(o_i1, m_i),
            ]
        )
        faces.append(
            [
                edge_ref(m_i, o_i1),
                edge_ref(o_i1, o_i),
                edge_ref(o_i, m_i),
            ]
        )

    return {
        "global_parameters": {
            "surface_tension": 0.0,
            "bending_energy_model": "helfrich",
            "spontaneous_curvature": 0.0,
            "tilt_modulus_in": 0.1,
            "tilt_modulus_out": 0.1,
            "bending_modulus_in": 1.0,
            "bending_modulus_out": 1.0,
            "tilt_rim_source_center": [0.0, 0.0, 0.0],
            "tilt_rim_source_group_in": "inner",
            "tilt_rim_source_strength_in": float(rim_source_strength),
            "tilt_solve_mode": "nested",
            "tilt_step_size": 0.05,
            "tilt_inner_steps": 120,
            "tilt_tol": 1e-12,
            "step_size": 0.002,
            "pin_to_plane_normal": [0.0, 0.0, 1.0],
            "pin_to_plane_point": [0.0, 0.0, 0.0],
        },
        "constraint_modules": ["pin_to_plane", "pin_to_circle"],
        "definitions": definitions,
        "energy_modules": [
            "bending_tilt_in",
            "bending_tilt_out",
            "tilt_smoothness_in",
            "tilt_smoothness_out",
            "tilt_in",
            "tilt_out",
            "tilt_rim_source_in",
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


def _outer_rim_rows(mesh) -> list[int]:
    return [
        mesh.vertex_index_to_row[int(vid)]
        for vid in mesh.vertex_ids
        if mesh.vertices[int(vid)].options.get("pin_to_circle_group") == "outer"
    ]


def _inner_rim_rows(mesh) -> list[int]:
    """Return vertex rows belonging to the inner rim group."""
    return [
        mesh.vertex_index_to_row[int(vid)]
        for vid in mesh.vertex_ids
        if mesh.vertices[int(vid)].options.get("pin_to_circle_group") == "inner"
    ]


def _mid_ring_rows(mesh) -> list[int]:
    """Return vertex rows for the mid ring (topologically tagged)."""
    return [
        mesh.vertex_index_to_row[int(vid)]
        for vid in mesh.vertex_ids
        if mesh.vertices[int(vid)].options.get("ring") == "mid"
    ]


def _break_up_down_symmetry(
    mesh, *, z_bump: float = 1e-3, target_radius: float = 2.0
) -> None:
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


def test_milestone_c_soft_source_generates_curvature_and_outer_tilt() -> None:
    """Milestone C: soft rim source + bending_tilt generates curvature; outer leaflet responds."""
    mesh = parse_geometry(_milestone_c_soft_source_data())
    mesh.global_parameters.set("tilt_inner_steps", 20)
    mesh.global_parameters.set("tilt_tol", 1e-8)

    # Break the up/down symmetry to make the relaxed shape deterministic.
    _break_up_down_symmetry(mesh)

    minim = _build_minimizer(mesh)
    minim.minimize(n_steps=50)

    positions = mesh.positions_view()
    z = positions[:, 2]

    outer_rows = _outer_rim_rows(mesh)
    assert len(outer_rows) > 0
    assert float(np.max(np.abs(z[outer_rows]))) < 1e-8
    assert float(np.max(np.abs(z))) > 2e-4

    # Without explicit `tilt_coupling`, the outer leaflet can still become nonzero
    # because `bending_tilt_out` couples it to the shared shape curvature.
    t_out = mesh.tilts_out_view()
    assert float(np.max(np.linalg.norm(t_out, axis=1))) > 5e-4


def test_milestone_c_without_bending_tilt_out_keeps_outer_tilt_zeroish() -> None:
    """Milestone C control: removing bending_tilt_out leaves tilt_out near its zero init."""
    mesh = parse_geometry(_milestone_c_soft_source_data())
    mesh.energy_modules = [m for m in mesh.energy_modules if m != "bending_tilt_out"]
    mesh.global_parameters.set("tilt_inner_steps", 20)
    mesh.global_parameters.set("tilt_tol", 1e-8)

    _break_up_down_symmetry(mesh)

    minim = _build_minimizer(mesh)
    minim.minimize(n_steps=50)

    t_out = mesh.tilts_out_view()
    assert float(np.max(np.linalg.norm(t_out, axis=1))) < 5e-5


@pytest.mark.xfail(
    reason=(
        "Sign-flip regression: swapping rim-source leaflet currently fails to flip "
        "curvature direction; tracked in docs/ROADMAP.md."
    ),
    strict=False,
)
def test_milestone_c_swapping_source_leaflet_flips_curvature_direction() -> None:
    """Milestone C sign test: putting the same source on the other leaflet flips invagination."""
    mesh_in = parse_geometry(_milestone_c_soft_source_data())
    mesh_in.global_parameters.set("tilt_inner_steps", 20)
    mesh_in.global_parameters.set("tilt_tol", 1e-8)
    _break_up_down_symmetry(mesh_in)

    minim_in = _build_minimizer(mesh_in)
    minim_in.minimize(n_steps=50)
    inner_rows_in = _inner_rim_rows(mesh_in)
    mid_rows_in = _mid_ring_rows(mesh_in)
    assert len(inner_rows_in) > 0
    assert len(mid_rows_in) > 0
    positions_in = mesh_in.positions_view()
    delta_in = float(
        np.mean(positions_in[mid_rows_in, 2]) - np.mean(positions_in[inner_rows_in, 2])
    )

    mesh_out = parse_geometry(_milestone_c_soft_source_data())
    mesh_out.global_parameters.set("tilt_inner_steps", 20)
    mesh_out.global_parameters.set("tilt_tol", 1e-8)
    mesh_out.energy_modules = [
        m for m in mesh_out.energy_modules if m != "tilt_rim_source_in"
    ] + ["tilt_rim_source_out"]
    mesh_out.global_parameters.set(
        "tilt_rim_source_group_out",
        mesh_out.global_parameters.get("tilt_rim_source_group_in"),
    )
    mesh_out.global_parameters.set(
        "tilt_rim_source_strength_out",
        mesh_out.global_parameters.get("tilt_rim_source_strength_in"),
    )
    _break_up_down_symmetry(mesh_out)

    minim_out = _build_minimizer(mesh_out)
    minim_out.minimize(n_steps=30)
    inner_rows_out = _inner_rim_rows(mesh_out)
    mid_rows_out = _mid_ring_rows(mesh_out)
    assert len(inner_rows_out) > 0
    assert len(mid_rows_out) > 0
    positions_out = mesh_out.positions_view()
    delta_out = float(
        np.mean(positions_out[mid_rows_out, 2])
        - np.mean(positions_out[inner_rows_out, 2])
    )

    assert abs(delta_in) > 1e-6
    assert abs(delta_out) > 1e-6
    assert delta_in * delta_out < 0.0
