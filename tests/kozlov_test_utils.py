"""Shared construction helpers for Kozlov regression and E2E tests."""

from pathlib import Path

import numpy as np

from tests.minimizer_test_utils import build_minimizer as build_minimizer

FIXTURE_DIR = Path(__file__).with_name("fixtures")


def fixture_path(name: str) -> str:
    """Return an absolute path to a named test fixture."""
    return str(FIXTURE_DIR / name)


def collect_group_rows(mesh, key: str, value: str) -> np.ndarray:
    """Return mesh rows whose vertex option matches a group value."""
    rows = [
        mesh.vertex_index_to_row[int(vid)]
        for vid in mesh.vertex_ids
        if (getattr(mesh.vertices[int(vid)], "options", None) or {}).get(key) == value
    ]
    return np.asarray(rows, dtype=int)


def order_by_angle(positions: np.ndarray) -> np.ndarray:
    """Return XY angular order for a position array."""
    return np.argsort(np.arctan2(positions[:, 1], positions[:, 0]))


def radial_unit_vectors(positions: np.ndarray) -> np.ndarray:
    """Return XY radial unit vectors with zeros at the origin."""
    radii = np.linalg.norm(positions[:, :2], axis=1)
    radial = np.zeros_like(positions)
    good = radii > 1e-12
    radial[good, 0] = positions[good, 0] / radii[good]
    radial[good, 1] = positions[good, 1] / radii[good]
    return radial


def outer_free_ring_rows(mesh, positions: np.ndarray) -> np.ndarray:
    """Return outermost rows not pinned to the outer circle group."""
    rows: list[int] = []
    radii: list[float] = []
    for vid in mesh.vertex_ids:
        vertex = mesh.vertices[int(vid)]
        opts = getattr(vertex, "options", None) or {}
        if opts.get("pin_to_circle_group") == "outer":
            continue
        row = mesh.vertex_index_to_row[int(vid)]
        rows.append(row)
        radii.append(float(np.linalg.norm(positions[row, :2])))
    if not rows:
        return np.zeros(0, dtype=int)
    radii_arr = np.asarray(radii, dtype=float)
    rows_arr = np.asarray(rows, dtype=int)
    return rows_arr[np.abs(radii_arr - float(np.max(radii_arr))) <= 1e-6]


def build_1disk_profile_data(*, bilayer: bool) -> dict:
    """Return the shared synthetic 1-disk profile mesh and lane configuration."""

    def ring_vertices(radius: float, *, n: int, z: float = 0.0) -> list[list[float]]:
        vertices: list[list[float]] = []
        for index in range(n):
            angle = 2.0 * np.pi * index / n
            vertices.append(
                [
                    float(radius) * float(np.cos(angle)),
                    float(radius) * float(np.sin(angle)),
                    float(z),
                ]
            )
        return vertices

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
    vertex_id = 1
    for radius, options, z in radii:
        ids: list[int] = []
        for x, y, z_coord in ring_vertices(radius, n=n, z=z):
            if options is None:
                vertices.append([x, y, z_coord])
            else:
                vertices.append([x, y, z_coord, dict(options)])
            ids.append(vertex_id)
            vertex_id += 1
        ring_vids.append(ids)

    edges: list[list[int]] = []
    edge_map: dict[tuple[int, int], int] = {}

    def get_edge(tail: int, head: int) -> tuple[int, bool]:
        first, second = (tail, head) if tail < head else (head, tail)
        edge_index = edge_map.get((first, second))
        if edge_index is None:
            edge_index = len(edges)
            edges.append([first, second])
            edge_map[(first, second)] = edge_index
        stored_tail, stored_head = edges[edge_index]
        return edge_index, stored_tail == tail and stored_head == head

    def face_edges(first: int, second: int, third: int) -> list:
        result: list = []
        for tail, head in ((first, second), (second, third), (third, first)):
            edge_index, forward = get_edge(tail, head)
            result.append(edge_index if forward else f"r{edge_index}")
        return result

    faces: list[list] = []

    def add_triangle(first: int, second: int, third: int) -> None:
        faces.append(face_edges(first, second, third))

    disk_inner = ring_vids[0]
    for index in range(n):
        add_triangle(0, disk_inner[index], disk_inner[(index + 1) % n])
    for inner_ring, outer_ring in zip(ring_vids, ring_vids[1:]):
        for index in range(n):
            inner = inner_ring[index]
            inner_next = inner_ring[(index + 1) % n]
            outer = outer_ring[index]
            outer_next = outer_ring[(index + 1) % n]
            add_triangle(inner, inner_next, outer)
            add_triangle(outer, inner_next, outer_next)

    data = {
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
    if bilayer:
        data["global_parameters"].update(
            {
                "bending_modulus_in": 0.1,
                "bending_modulus_out": 0.1,
                "tilt_modulus_in": 1.0,
                "tilt_modulus_out": 1.0,
                "tilt_disk_target_strength_in": 200.0,
                "tilt_disk_target_group_out": "disk",
                "tilt_disk_target_strength_out": 200.0,
                "rim_slope_match_strength": 0.0,
            }
        )
        data["constraint_modules"].append("rim_slope_match_out")
        data["definitions"]["disk"].update(
            {
                "tilt_disk_target_group_out": "disk",
                "pin_to_plane_mode": "slide",
                "pin_to_plane_group": "disk_plane",
            }
        )
        data["definitions"]["rim"].update(
            {
                "pin_to_plane_mode": "slide",
                "pin_to_plane_group": "disk_plane",
            }
        )
        data["energy_modules"].insert(-1, "tilt_disk_target_out")
    return data
