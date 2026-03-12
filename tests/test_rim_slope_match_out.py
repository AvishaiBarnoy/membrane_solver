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
    normals = mesh.vertex_normals(positions=positions)
    radial_tangent = r_hat - np.einsum("ij,ij->i", r_hat, normals)[:, None] * normals
    radial_tangent_norm = np.linalg.norm(radial_tangent, axis=1)
    radial_tangent = np.divide(
        radial_tangent,
        radial_tangent_norm[:, None],
        out=np.zeros_like(radial_tangent),
        where=radial_tangent_norm[:, None] > 1.0e-12,
    )
    normals = mesh.vertex_normals(positions=positions)
    radial_tangent = r_hat - np.einsum("ij,ij->i", r_hat, normals)[:, None] * normals
    radial_tangent_norm = np.linalg.norm(radial_tangent, axis=1)
    radial_tangent = np.divide(
        radial_tangent,
        radial_tangent_norm[:, None],
        out=np.zeros_like(radial_tangent),
        where=radial_tangent_norm[:, None] > 1e-12,
    )

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
    mesh.set_tilts_out_from_array(np.zeros_like(positions))
    e_zero = rim_slope_match_out.compute_energy_and_gradient_array(
        mesh,
        mesh.global_parameters,
        resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=grad_dummy,
        tilts_out=mesh.tilts_out_view(),
        tilt_out_grad_arr=None,
    )

    assert float(e_match) < 1.0e-2
    assert float(e_match) < (0.01 * float(e_zero))

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


def test_rim_slope_match_out_shared_staggered_supports_interpolated_outer_pairing() -> (
    None
):
    mesh = parse_geometry(_annulus_two_ring_mesh(n=16))
    resolver = ParameterResolver(mesh.global_parameters)
    mesh.global_parameters.set("rim_slope_match_mode", "shared_rim_staggered_v1")

    # Keep only every other inner-ring vertex in the rim group so the shared-rim
    # matcher must interpolate on the denser outer ring.
    for row, vid in enumerate(sorted(mesh.vertices.keys())):
        opts = getattr(mesh.vertices[int(vid)], "options", None) or {}
        if row < 16 and (row % 2 == 1):
            opts.pop("rim_slope_match_group", None)
        mesh.vertices[int(vid)].options = opts

    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row
    grad_dummy = np.zeros_like(positions)

    r = np.linalg.norm(positions[:, :2], axis=1)
    r_hat = np.zeros_like(positions)
    r_hat[r > 1e-12, 0] = positions[r > 1e-12, 0] / r[r > 1e-12]
    r_hat[r > 1e-12, 1] = positions[r > 1e-12, 1] / r[r > 1e-12]
    normals = mesh.vertex_normals(positions=positions)
    radial_tangent = r_hat - np.einsum("ij,ij->i", r_hat, normals)[:, None] * normals
    radial_tangent_norm = np.linalg.norm(radial_tangent, axis=1)
    radial_tangent = np.divide(
        radial_tangent,
        radial_tangent_norm[:, None],
        out=np.zeros_like(radial_tangent),
        where=radial_tangent_norm[:, None] > 1e-12,
    )

    tilts_out = np.zeros_like(positions)
    outer_mask = np.array(
        [
            (getattr(mesh.vertices[int(vid)], "options", None) or {}).get(
                "rim_slope_match_group"
            )
            == "outer"
            for vid in sorted(mesh.vertices.keys())
        ],
        dtype=bool,
    )
    tilts_out[outer_mask] = 0.2 * radial_tangent[outer_mask]
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
    mesh.set_tilts_out_from_array(np.zeros_like(positions))
    e_zero = rim_slope_match_out.compute_energy_and_gradient_array(
        mesh,
        mesh.global_parameters,
        resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=grad_dummy,
        tilts_out=mesh.tilts_out_view(),
        tilt_out_grad_arr=None,
    )

    assert float(e_match) < 1.0e-2
    assert float(e_match) < (0.01 * float(e_zero))


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


def test_rim_slope_match_out_skips_disk_group_when_matching_rim() -> None:
    mesh = parse_geometry(_annulus_three_ring_mesh())
    mesh.global_parameters.set("rim_slope_match_disk_group", "rim")
    resolver = ParameterResolver(mesh.global_parameters)

    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row
    grad_dummy = np.zeros_like(positions)

    rim_rows = []
    outer_rows = []
    for vid in mesh.vertex_ids:
        opts = getattr(mesh.vertices[int(vid)], "options", None) or {}
        row = mesh.vertex_index_to_row[int(vid)]
        if opts.get("rim_slope_match_group") == "rim":
            rim_rows.append(row)
        elif opts.get("rim_slope_match_group") == "outer":
            outer_rows.append(row)

    rim_rows = np.asarray(rim_rows, dtype=int)
    outer_rows = np.asarray(outer_rows, dtype=int)
    rim_pos = positions[rim_rows]
    outer_pos = positions[outer_rows]

    rim_order = np.argsort(np.arctan2(rim_pos[:, 1], rim_pos[:, 0]))
    outer_order = np.argsort(np.arctan2(outer_pos[:, 1], outer_pos[:, 0]))
    rim_rows = rim_rows[rim_order]
    outer_rows = outer_rows[outer_order]
    rim_pos = positions[rim_rows]
    outer_pos = positions[outer_rows]

    r_rim = np.linalg.norm(rim_pos[:, :2], axis=1)
    r_out = np.linalg.norm(outer_pos[:, :2], axis=1)
    dr = np.maximum(r_out - r_rim, 1e-6)
    phi = (outer_pos[:, 2] - rim_pos[:, 2]) / dr

    r_hat = np.zeros_like(rim_pos)
    good = r_rim > 1e-12
    r_hat[good, 0] = rim_pos[good, 0] / r_rim[good]
    r_hat[good, 1] = rim_pos[good, 1] / r_rim[good]

    tilts_out = np.zeros_like(positions)
    tilts_out[rim_rows] = phi[:, None] * r_hat
    mesh.set_tilts_out_from_array(tilts_out)

    tilts_in = np.zeros_like(positions)
    mesh.set_tilts_in_from_array(tilts_in)

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


def test_rim_slope_match_out_shared_rim_staggered_mode_zero_when_outer_ring_matches():
    mesh = parse_geometry(_annulus_three_ring_mesh())
    mesh.global_parameters.set("rim_slope_match_mode", "shared_rim_staggered_v1")
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
    outer_rows = np.arange(2 * n, 3 * n, dtype=int)

    normals = mesh.vertex_normals(positions=positions)
    outer_normals = normals[outer_rows]
    outer_r_dir = (
        r_hat[outer_rows]
        - np.einsum("ij,ij->i", r_hat[outer_rows], outer_normals)[:, None]
        * outer_normals
    )
    outer_r_dir /= np.linalg.norm(outer_r_dir, axis=1)[:, None]

    tilts_in = np.zeros_like(positions)
    tilts_out = np.zeros_like(positions)
    tilts_in[disk_rows] = 0.6 * r_hat[disk_rows]
    tilts_in[outer_rows] = 0.4 * outer_r_dir
    tilts_out[outer_rows] = 0.2 * outer_r_dir
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
    assert np.allclose(grad_dummy[rim_rows], 0.0, atol=1e-6)
    assert np.allclose(grad_dummy[outer_rows], 0.0, atol=1e-6)


def test_rim_slope_match_out_shared_rim_staggered_mode_targets_outer_tilt_rows() -> (
    None
):
    mesh = parse_geometry(_annulus_three_ring_mesh())
    mesh.global_parameters.set("rim_slope_match_mode", "shared_rim_staggered_v1")
    resolver = ParameterResolver(mesh.global_parameters)

    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row
    grad_arr = np.zeros_like(positions)
    tilt_in_grad_arr = np.zeros_like(positions)
    tilt_out_grad_arr = np.zeros_like(positions)

    r = np.linalg.norm(positions[:, :2], axis=1)
    r_hat = np.zeros_like(positions)
    good = r > 1e-12
    r_hat[good, 0] = positions[good, 0] / r[good]
    r_hat[good, 1] = positions[good, 1] / r[good]

    n = len(positions) // 3
    disk_rows = np.arange(0, n, dtype=int)
    rim_rows = np.arange(n, 2 * n, dtype=int)
    outer_rows = np.arange(2 * n, 3 * n, dtype=int)

    tilts_in = np.zeros_like(positions)
    tilts_out = np.zeros_like(positions)
    tilts_in[disk_rows] = 0.6 * r_hat[disk_rows]
    mesh.set_tilts_in_from_array(tilts_in)
    mesh.set_tilts_out_from_array(tilts_out)

    energy = rim_slope_match_out.compute_energy_and_gradient_array(
        mesh,
        mesh.global_parameters,
        resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=grad_arr,
        tilts_in=mesh.tilts_in_view(),
        tilts_out=mesh.tilts_out_view(),
        tilt_in_grad_arr=tilt_in_grad_arr,
        tilt_out_grad_arr=tilt_out_grad_arr,
    )
    assert float(energy) > 0.0
    assert np.linalg.norm(tilt_out_grad_arr[outer_rows]) > 0.0
    assert np.linalg.norm(tilt_out_grad_arr[rim_rows]) == 0.0
    assert np.linalg.norm(tilt_in_grad_arr[outer_rows]) > 0.0
    assert np.linalg.norm(tilt_in_grad_arr[disk_rows]) > 0.0
    assert np.linalg.norm(grad_arr[rim_rows]) > 0.0
    assert np.linalg.norm(grad_arr[outer_rows]) > 0.0
