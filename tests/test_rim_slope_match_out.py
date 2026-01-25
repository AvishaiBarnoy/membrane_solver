import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.parameters.resolver import ParameterResolver
from geometry.geom_io import parse_geometry
from modules.energy import rim_slope_match_out


def _annulus_two_ring_mesh(*, n: int = 8, r_rim: float = 1.0, r_out: float = 2.0):
    vertices: list[list] = []
    for i in range(n):
        theta = 2.0 * np.pi * i / n
        vertices.append(
            [
                float(r_rim * np.cos(theta)),
                float(r_rim * np.sin(theta)),
                0.0,
                {"rim_slope_match_group": "rim"},
            ]
        )
    for i in range(n):
        theta = 2.0 * np.pi * i / n
        vertices.append(
            [
                float(r_out * np.cos(theta)),
                float(r_out * np.sin(theta)),
                0.2,
                {"rim_slope_match_group": "outer"},
            ]
        )

    def vid(ring: int, k: int) -> int:
        return int(ring) * int(n) + int(k)

    edges: list[list[int]] = []
    for ring in range(2):
        for k in range(n):
            edges.append([vid(ring, k), vid(ring, (k + 1) % n)])
    for k in range(n):
        edges.append([vid(0, k), vid(1, k)])
    for k in range(n):
        edges.append([vid(0, k), vid(1, (k + 1) % n)])

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
    for k in range(n):
        k1 = (k + 1) % n
        v00 = vid(0, k)
        v01 = vid(0, k1)
        v10 = vid(1, k)
        v11 = vid(1, k1)
        faces.append([edge_ref(v00, v01), edge_ref(v01, v11), edge_ref(v11, v00)])
        faces.append([edge_ref(v00, v11), edge_ref(v11, v10), edge_ref(v10, v00)])

    return {
        "global_parameters": {
            "rim_slope_match_group": "rim",
            "rim_slope_match_outer_group": "outer",
            "rim_slope_match_strength": 10.0,
            "rim_slope_match_center": [0.0, 0.0, 0.0],
            "rim_slope_match_normal": [0.0, 0.0, 1.0],
        },
        "energy_modules": [],
        "vertices": vertices,
        "edges": edges,
        "faces": faces,
        "instructions": [],
    }


def test_rim_slope_match_out_zero_when_tilt_equals_slope() -> None:
    mesh = parse_geometry(_annulus_two_ring_mesh())
    resolver = ParameterResolver(mesh.global_parameters)

    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row
    grad_dummy = np.zeros_like(positions)

    # Compute desired slope: (z_out - z_rim) / (r_out - r_rim) = 0.2
    r = np.linalg.norm(positions[:, :2], axis=1)
    r_hat = np.zeros_like(positions)
    r_hat[r > 1e-12, 0] = positions[r > 1e-12, 0] / r[r > 1e-12]
    r_hat[r > 1e-12, 1] = positions[r > 1e-12, 1] / r[r > 1e-12]

    tilts_out = np.zeros_like(positions)
    rim_mask = np.arange(len(positions)) < (len(positions) // 2)
    tilts_out[rim_mask] = 0.2 * r_hat[rim_mask]
    mesh.set_tilts_out_from_array(tilts_out)

    e_match = rim_slope_match_out.compute_energy_and_gradient_array(
        mesh,
        mesh.global_parameters,
        resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=grad_dummy,
        tilts_out=mesh.tilts_out_view(),
        tilt_out_grad_arr=None,
    )
    assert abs(float(e_match)) < 1e-6

    tilts_out[rim_mask] = 0.0
    mesh.set_tilts_out_from_array(tilts_out)
    e_mismatch = rim_slope_match_out.compute_energy_and_gradient_array(
        mesh,
        mesh.global_parameters,
        resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=grad_dummy,
        tilts_out=mesh.tilts_out_view(),
        tilt_out_grad_arr=None,
    )
    assert float(e_mismatch) > 1e-3


def _annulus_three_ring_mesh(
    *, n: int = 8, r_disk: float = 0.5, r_rim: float = 1.0, r_out: float = 2.0
):
    vertices: list[list] = []
    for i in range(n):
        theta = 2.0 * np.pi * i / n
        vertices.append(
            [
                float(r_disk * np.cos(theta)),
                float(r_disk * np.sin(theta)),
                0.0,
                {"rim_slope_match_group": "disk"},
            ]
        )
    for i in range(n):
        theta = 2.0 * np.pi * i / n
        vertices.append(
            [
                float(r_rim * np.cos(theta)),
                float(r_rim * np.sin(theta)),
                0.0,
                {"rim_slope_match_group": "rim"},
            ]
        )
    for i in range(n):
        theta = 2.0 * np.pi * i / n
        vertices.append(
            [
                float(r_out * np.cos(theta)),
                float(r_out * np.sin(theta)),
                0.2,
                {"rim_slope_match_group": "outer"},
            ]
        )

    def vid(ring: int, k: int) -> int:
        return int(ring) * int(n) + int(k)

    edges: list[list[int]] = []
    for ring in range(3):
        for k in range(n):
            edges.append([vid(ring, k), vid(ring, (k + 1) % n)])
    for ring in range(2):
        for k in range(n):
            edges.append([vid(ring, k), vid(ring + 1, k)])
            edges.append([vid(ring, k), vid(ring + 1, (k + 1) % n)])

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
    for ring in range(2):
        for k in range(n):
            k1 = (k + 1) % n
            v00 = vid(ring, k)
            v01 = vid(ring, k1)
            v10 = vid(ring + 1, k)
            v11 = vid(ring + 1, k1)
            faces.append([edge_ref(v00, v01), edge_ref(v01, v11), edge_ref(v11, v00)])
            faces.append([edge_ref(v00, v11), edge_ref(v11, v10), edge_ref(v10, v00)])

    return {
        "global_parameters": {
            "rim_slope_match_group": "rim",
            "rim_slope_match_outer_group": "outer",
            "rim_slope_match_disk_group": "disk",
            "rim_slope_match_strength": 10.0,
            "rim_slope_match_center": [0.0, 0.0, 0.0],
            "rim_slope_match_normal": [0.0, 0.0, 1.0],
        },
        "energy_modules": [],
        "vertices": vertices,
        "edges": edges,
        "faces": faces,
        "instructions": [],
    }


def test_rim_slope_match_out_with_disk_group_zero_when_matching() -> None:
    mesh = parse_geometry(_annulus_three_ring_mesh())
    resolver = ParameterResolver(mesh.global_parameters)

    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row
    grad_dummy = np.zeros_like(positions)

    r = np.linalg.norm(positions[:, :2], axis=1)
    r_hat = np.zeros_like(positions)
    good = r > 1e-12
    r_hat[good, 0] = positions[good, 0] / r[good]
    r_hat[good, 1] = positions[good, 1] / r[good]

    n = len(positions) // 3
    disk_rows = np.arange(0, n, dtype=int)
    rim_rows = np.arange(n, 2 * n, dtype=int)

    # Desired: phi = 0.2, theta_disk = 0.6 => rim tilt_in = 0.4, tilt_out = 0.2.
    tilts_in = np.zeros_like(positions)
    tilts_out = np.zeros_like(positions)
    tilts_in[disk_rows] = 0.6 * r_hat[disk_rows]
    tilts_in[rim_rows] = 0.4 * r_hat[rim_rows]
    tilts_out[rim_rows] = 0.2 * r_hat[rim_rows]
    mesh.set_tilts_in_from_array(tilts_in)
    mesh.set_tilts_out_from_array(tilts_out)

    e_match = rim_slope_match_out.compute_energy_and_gradient_array(
        mesh,
        mesh.global_parameters,
        resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=grad_dummy,
        tilts_in=mesh.tilts_in_view(),
        tilts_out=mesh.tilts_out_view(),
        tilt_in_grad_arr=None,
        tilt_out_grad_arr=None,
    )
    assert abs(float(e_match)) < 1e-6
