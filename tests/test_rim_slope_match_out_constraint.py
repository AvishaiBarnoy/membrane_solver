import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import parse_geometry


def _build_rim_match_geometry(
    *,
    n_theta: int = 8,
    r_disk_inner: float = 0.6,
    r_rim: float = 1.0,
    r_out: float = 2.0,
    z_bump: float = 0.1,
) -> dict:
    """Create a small disk+rim+outer mesh with labeled rings."""
    radii = [float(r_disk_inner), float(r_rim), float(r_out)]
    rim_ring = 2
    outer_ring = 3

    vertices: list[list] = []
    vertices.append([0.0, 0.0, 0.0])  # center

    for ring_idx, r in enumerate(radii, start=1):
        for i in range(n_theta):
            theta = 2.0 * np.pi * i / float(n_theta)
            x = float(r * np.cos(theta))
            y = float(r * np.sin(theta))
            z = 0.0
            opts: dict | None = None
            if ring_idx == rim_ring - 1:
                opts = {"rim_slope_match_group": "disk"}
            elif ring_idx == rim_ring:
                opts = {"rim_slope_match_group": "rim"}
            elif ring_idx == outer_ring:
                z = float(z_bump)
                opts = {"rim_slope_match_group": "outer"}
            vertices.append([x, y, z] if opts is None else [x, y, z, opts])

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
            "rim_slope_match_group": "rim",
            "rim_slope_match_outer_group": "outer",
            "rim_slope_match_disk_group": "disk",
            "rim_slope_match_center": [0.0, 0.0, 0.0],
            "rim_slope_match_normal": [0.0, 0.0, 1.0],
        },
        "constraint_modules": ["rim_slope_match_out"],
        "energy_modules": [],
        "vertices": vertices,
        "edges": edges,
        "faces": faces,
        "instructions": [],
    }


def _collect_group_rows(mesh, group: str) -> np.ndarray:
    rows: list[int] = []
    for vid in mesh.vertex_ids:
        opts = getattr(mesh.vertices[int(vid)], "options", None) or {}
        if opts.get("rim_slope_match_group") == group:
            row = mesh.vertex_index_to_row.get(int(vid))
            if row is not None:
                rows.append(int(row))
    return np.asarray(rows, dtype=int)


def _order_by_angle(positions: np.ndarray, normal: np.ndarray) -> np.ndarray:
    trial = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(np.dot(trial, normal)) > 0.9:
        trial = np.array([0.0, 1.0, 0.0], dtype=float)
    u = trial - np.dot(trial, normal) * normal
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    rel_plane = positions - np.einsum("ij,j->i", positions, normal)[:, None] * normal
    x = rel_plane @ u
    y = rel_plane @ v
    return np.argsort(np.arctan2(y, x))


def test_rim_slope_match_out_constraint_enforces_radial_tilts():
    data = _build_rim_match_geometry()
    mesh = parse_geometry(data)

    positions = mesh.positions_view()
    tilts_in = mesh.tilts_in_view().copy(order="F")
    tilts_out = mesh.tilts_out_view().copy(order="F")

    rim_rows = _collect_group_rows(mesh, "rim")
    disk_rows = _collect_group_rows(mesh, "disk")
    outer_rows = _collect_group_rows(mesh, "outer")

    normal = np.array([0.0, 0.0, 1.0], dtype=float)
    rim_order = _order_by_angle(positions[rim_rows], normal)
    disk_order = _order_by_angle(positions[disk_rows], normal)
    outer_order = _order_by_angle(positions[outer_rows], normal)

    rim_rows = rim_rows[rim_order]
    disk_rows = disk_rows[disk_order]
    outer_rows = outer_rows[outer_order]

    r_vec = positions[rim_rows].copy()
    r_vec[:, 2] = 0.0
    r_hat = r_vec / np.linalg.norm(r_vec, axis=1)[:, None]
    normals = mesh.vertex_normals(positions=positions)
    rim_normals = normals[rim_rows]
    r_dir = r_hat - np.einsum("ij,ij->i", r_hat, rim_normals)[:, None] * rim_normals
    r_dir /= np.linalg.norm(r_dir, axis=1)[:, None]

    r_vec_disk = positions[disk_rows].copy()
    r_vec_disk[:, 2] = 0.0
    r_hat_disk = r_vec_disk / np.linalg.norm(r_vec_disk, axis=1)[:, None]

    tilts_out[rim_rows] = 0.02 * r_dir
    tilts_in[rim_rows] = 0.01 * r_dir
    tilts_in[disk_rows] = 0.2 * r_hat_disk

    mesh.set_tilts_in_from_array(tilts_in)
    mesh.set_tilts_out_from_array(tilts_out)

    from modules.constraints import rim_slope_match_out as constraint

    constraint.enforce_tilt_constraint(mesh, global_params=mesh.global_parameters)

    tilts_in = mesh.tilts_in_view()
    tilts_out = mesh.tilts_out_view()

    h_rim = positions[rim_rows][:, 2]
    h_out = positions[outer_rows][:, 2]
    r_rim = np.linalg.norm(positions[rim_rows][:, :2], axis=1)
    r_out = np.linalg.norm(positions[outer_rows][:, :2], axis=1)
    phi = (h_out - h_rim) / (r_out - r_rim)

    theta_disk = np.einsum("ij,ij->i", tilts_in[disk_rows], r_hat_disk)
    target_in = theta_disk - phi

    t_out_rad = np.einsum("ij,ij->i", tilts_out[rim_rows], r_dir)
    t_in_rad = np.einsum("ij,ij->i", tilts_in[rim_rows], r_dir)

    assert np.allclose(t_out_rad, phi, atol=1e-6)
    assert np.allclose(t_in_rad, target_in, atol=1e-6)


def test_group_rows_cache_keeps_multiple_groups() -> None:
    data = _build_rim_match_geometry()
    mesh = parse_geometry(data)
    mesh.build_position_cache()

    from modules.constraints import rim_slope_match_out as constraint

    rim_rows_1 = constraint._collect_group_rows(mesh, "rim")
    outer_rows_1 = constraint._collect_group_rows(mesh, "outer")
    disk_rows_1 = constraint._collect_group_rows(mesh, "disk")
    rim_rows_2 = constraint._collect_group_rows(mesh, "rim")

    np.testing.assert_array_equal(rim_rows_1, rim_rows_2)
    assert outer_rows_1.size > 0
    assert disk_rows_1.size > 0

    cache = getattr(mesh, "_rim_slope_match_group_rows_cache", {})
    entries = cache.get("entries", {})
    assert len(entries) >= 3


def test_matching_data_cache_reuses_and_invalidates_on_version_change() -> None:
    data = _build_rim_match_geometry()
    mesh = parse_geometry(data)
    mesh.build_position_cache()
    gp = mesh.global_parameters
    positions = mesh.positions_view()

    from modules.constraints import rim_slope_match_out as constraint

    d1 = constraint._build_matching_data(mesh, gp, positions)
    d2 = constraint._build_matching_data(mesh, gp, positions)
    assert d1 is d2

    mesh.increment_version()
    positions2 = mesh.positions_view()
    d3 = constraint._build_matching_data(mesh, gp, positions2)
    assert d3 is not d2
