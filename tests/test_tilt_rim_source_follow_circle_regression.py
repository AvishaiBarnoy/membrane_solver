import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.parameters.resolver import ParameterResolver
from geometry.geom_io import parse_geometry
from modules.energy import tilt_rim_source_in


def _annulus_with_inner_group(
    *,
    center: np.ndarray,
    z: float,
    n: int = 12,
    r_in: float = 1.0,
    r_out: float = 2.0,
) -> dict:
    """Return a minimal annulus mesh dict with a fit-mode inner circle group."""
    center = np.asarray(center, dtype=float).reshape(3)
    center[2] = float(z)

    vertices: list[list] = []
    for i in range(n):
        theta = 2.0 * np.pi * i / n
        pos = center + np.array([r_in * np.cos(theta), r_in * np.sin(theta), 0.0])
        vertices.append(
            [
                float(pos[0]),
                float(pos[1]),
                float(pos[2]),
                {
                    "constraints": ["pin_to_circle"],
                    "pin_to_circle_group": "inner",
                    "pin_to_circle_mode": "fit",
                    "pin_to_circle_normal": [0.0, 0.0, 1.0],
                },
            ]
        )

    for i in range(n):
        theta = 2.0 * np.pi * i / n
        pos = center + np.array([r_out * np.cos(theta), r_out * np.sin(theta), 0.0])
        vertices.append([float(pos[0]), float(pos[1]), float(pos[2])])

    edges: list[list[int]] = []
    inner_edges = []
    outer_edges = []
    spokes = []

    # Inner loop edges.
    for i in range(n):
        a = i
        b = (i + 1) % n
        inner_edges.append(len(edges))
        edges.append([a, b])

    # Outer loop edges.
    for i in range(n):
        a = n + i
        b = n + ((i + 1) % n)
        outer_edges.append(len(edges))
        edges.append([a, b])

    # Spokes (inner i -> outer i).
    for i in range(n):
        a = i
        b = n + i
        spokes.append(len(edges))
        edges.append([a, b])

    faces: list[list] = []
    for i in range(n):
        i_next = (i + 1) % n
        # Quad: inner(i)->inner(i+1)->outer(i+1)->outer(i)->inner(i)
        e_inner = inner_edges[i]
        e_spoke_next = spokes[i_next]
        e_outer_rev = f"r{outer_edges[i]}"
        e_spoke_rev = f"r{spokes[i]}"
        faces.append([e_inner, e_spoke_next, e_outer_rev, e_spoke_rev])

    return {
        "global_parameters": {
            "tilt_rim_source_group_in": "inner",
            "tilt_rim_source_strength_in": 1.0,
            # Deliberately incorrect when the circle is translated; fit-mode
            # rims should ignore this and follow the fitted circle.
            "tilt_rim_source_center": [0.0, 0.0, 0.0],
        },
        "energy_modules": ["tilt_rim_source_in"],
        "vertices": vertices,
        "edges": edges,
        "faces": faces,
        "instructions": [],
    }


def _set_radial_tilt_in(mesh, *, center: np.ndarray) -> None:
    positions = mesh.positions_view()
    center = np.asarray(center, dtype=float).reshape(3)
    r = positions - center[None, :]
    r[:, 2] = 0.0
    rn = np.linalg.norm(r, axis=1)
    tilts = np.zeros_like(positions)
    good = rn > 1e-12
    tilts[good] = r[good] / rn[good][:, None]
    mesh.set_tilts_in_from_array(tilts)


def test_tilt_rim_source_follows_fit_circle_translation_invariance():
    base_center = np.array([0.0, 0.0, 0.0], dtype=float)
    shifted_center = np.array([0.6, -0.4, 0.0], dtype=float)
    z = 1.25

    mesh0 = parse_geometry(_annulus_with_inner_group(center=base_center, z=z))
    _set_radial_tilt_in(mesh0, center=np.array([0.0, 0.0, z], dtype=float))
    resolver0 = ParameterResolver(mesh0.global_parameters)
    E0 = tilt_rim_source_in.compute_energy_and_gradient_array(
        mesh0,
        mesh0.global_parameters,
        resolver0,
        positions=mesh0.positions_view(),
        index_map=mesh0.vertex_index_to_row,
        grad_arr=np.zeros_like(mesh0.positions_view()),
        tilt_in_grad_arr=None,
    )

    mesh1 = parse_geometry(_annulus_with_inner_group(center=shifted_center, z=z))
    _set_radial_tilt_in(
        mesh1, center=np.array([shifted_center[0], shifted_center[1], z])
    )
    resolver1 = ParameterResolver(mesh1.global_parameters)
    E1 = tilt_rim_source_in.compute_energy_and_gradient_array(
        mesh1,
        mesh1.global_parameters,
        resolver1,
        positions=mesh1.positions_view(),
        index_map=mesh1.vertex_index_to_row,
        grad_arr=np.zeros_like(mesh1.positions_view()),
        tilt_in_grad_arr=None,
    )

    # The rim source should be invariant to translating the fitted circle.
    assert np.isfinite(E0)
    assert np.isfinite(E1)
    assert abs(float(E0) - float(E1)) < 1e-6
