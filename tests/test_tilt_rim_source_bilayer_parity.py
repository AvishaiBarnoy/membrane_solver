import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.parameters.resolver import ParameterResolver
from geometry.geom_io import parse_geometry
from modules.energy import (
    tilt_rim_source_bilayer,
    tilt_rim_source_in,
    tilt_rim_source_out,
)


def _annulus_source_mesh(*, n: int = 10) -> dict:
    vertices: list[list] = []
    for i in range(n):
        theta = 2.0 * np.pi * i / n
        vertices.append(
            [
                float(np.cos(theta)),
                float(np.sin(theta)),
                0.0,
                {
                    "pin_to_circle_group": "inner",
                    "pin_to_circle_mode": "fit",
                    "pin_to_circle_normal": [0.0, 0.0, 1.0],
                },
            ]
        )
    for i in range(n):
        theta = 2.0 * np.pi * i / n
        vertices.append([float(2.0 * np.cos(theta)), float(2.0 * np.sin(theta)), 0.0])

    edges: list[list[int]] = []
    inner_edges = []
    outer_edges = []
    spokes = []
    for i in range(n):
        inner_edges.append(len(edges))
        edges.append([i, (i + 1) % n])
    for i in range(n):
        outer_edges.append(len(edges))
        edges.append([n + i, n + ((i + 1) % n)])
    for i in range(n):
        spokes.append(len(edges))
        edges.append([i, n + i])

    faces: list[list] = []
    for i in range(n):
        i_next = (i + 1) % n
        faces.append(
            [
                inner_edges[i],
                spokes[i_next],
                f"r{outer_edges[i]}",
                f"r{spokes[i]}",
            ]
        )

    return {
        "global_parameters": {
            "tilt_rim_source_center": [0.0, 0.0, 0.0],
            "tilt_rim_source_group": "inner",
            "tilt_rim_source_strength": 1.0,
            "tilt_rim_source_group_in": "inner",
            "tilt_rim_source_strength_in": 1.0,
            "tilt_rim_source_group_out": "inner",
            "tilt_rim_source_strength_out": 1.0,
        },
        "energy_modules": [],
        "vertices": vertices,
        "edges": edges,
        "faces": faces,
        "instructions": [],
    }


def _set_radial_tilts(mesh) -> None:
    positions = mesh.positions_view()
    r = positions.copy()
    r[:, 2] = 0.0
    rn = np.linalg.norm(r, axis=1)
    radial = np.zeros_like(positions)
    good = rn > 1e-12
    radial[good] = r[good] / rn[good][:, None]
    mesh.set_tilts_in_from_array(radial)
    mesh.set_tilts_out_from_array(2.0 * radial)


def test_tilt_rim_source_bilayer_matches_in_plus_out():
    mesh = parse_geometry(_annulus_source_mesh())
    _set_radial_tilts(mesh)
    resolver = ParameterResolver(mesh.global_parameters)

    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row
    grad_dummy = np.zeros_like(positions)

    grad_in_b = np.zeros_like(positions)
    grad_out_b = np.zeros_like(positions)
    e_b = tilt_rim_source_bilayer.compute_energy_and_gradient_array(
        mesh,
        mesh.global_parameters,
        resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=grad_dummy,
        tilts_in=mesh.tilts_in_view(),
        tilts_out=mesh.tilts_out_view(),
        tilt_in_grad_arr=grad_in_b,
        tilt_out_grad_arr=grad_out_b,
    )

    grad_in = np.zeros_like(positions)
    grad_out = np.zeros_like(positions)
    e_in = tilt_rim_source_in.compute_energy_and_gradient_array(
        mesh,
        mesh.global_parameters,
        resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=grad_dummy,
        tilts_in=mesh.tilts_in_view(),
        tilt_in_grad_arr=grad_in,
    )
    e_out = tilt_rim_source_out.compute_energy_and_gradient_array(
        mesh,
        mesh.global_parameters,
        resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=grad_dummy,
        tilts_out=mesh.tilts_out_view(),
        tilt_out_grad_arr=grad_out,
    )

    assert float(e_b) == float(e_in + e_out)
    assert np.allclose(grad_in_b, grad_in)
    assert np.allclose(grad_out_b, grad_out)
