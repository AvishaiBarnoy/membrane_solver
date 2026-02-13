import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import parse_geometry
from modules.constraints import pin_to_circle
from runtime.constraint_manager import ConstraintModuleManager


def test_pin_to_circle_projects_vertices_and_edges() -> None:
    radius = 2.0
    data = {
        "vertices": [
            [3.0, 0.0, 5.0, {"constraints": ["pin_to_circle"]}],
            [0.0, 3.0, -1.0],  # via edge constraint
            [0.0, 0.0, 0.0],  # via edge constraint
            [0.0, 0.0, 0.0],  # untouched
        ],
        "edges": [
            [1, 2, {"constraints": ["pin_to_circle"]}],
        ],
        "faces": [],
        "global_parameters": {
            "pin_to_circle_normal": [0.0, 0.0, 1.0],
            "pin_to_circle_point": [0.0, 0.0, 0.0],
            "pin_to_circle_radius": radius,
        },
    }
    mesh = parse_geometry(data)
    manager = ConstraintModuleManager(mesh.constraint_modules)
    manager.enforce_all(mesh)

    center = np.array([0.0, 0.0, 0.0])
    for vidx in (0, 1, 2):
        pos = mesh.vertices[vidx].position
        assert np.isclose(pos[2], 0.0)
        assert np.isclose(np.linalg.norm(pos - center), radius)

    assert np.allclose(mesh.vertices[3].position, np.array([0.0, 0.0, 0.0]))


def test_pin_to_circle_fit_allows_rim_to_move() -> None:
    radius = 2.0
    center = np.array([10.0, 5.0, -3.0], dtype=float)
    normal = np.array([0.0, 1.0, 0.0], dtype=float)  # y=const plane

    angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    points = np.stack(
        [
            center[0] + radius * np.cos(angles),
            center[1] + np.zeros_like(angles),
            center[2] + radius * np.sin(angles),
        ],
        axis=1,
    )

    # Add small noise so the fitter has something to correct.
    rng = np.random.default_rng(123)
    points = points + 0.02 * rng.normal(size=points.shape)

    data = {
        "vertices": [
            [
                float(p[0]),
                float(p[1]),
                float(p[2]),
                {
                    "constraints": ["pin_to_circle"],
                    "pin_to_circle_group": "rim",
                },
            ]
            for p in points
        ],
        "edges": [[0, 1]],
        "faces": [],
        "global_parameters": {
            "pin_to_circle_mode": "fit",
            "pin_to_circle_normal": normal.tolist(),
            "pin_to_circle_radius": radius,
        },
    }
    mesh = parse_geometry(data)
    manager = ConstraintModuleManager(mesh.constraint_modules)
    manager.enforce_all(mesh)

    # The circle is not pinned to the origin; it should remain near the input center.
    y_mean = float(np.mean([v.position[1] for v in mesh.vertices.values()]))
    assert np.isclose(y_mean, center[1], atol=0.1)

    # All constrained points should be at the requested radius around the fitted center.
    # Use the module's internal fitter so the test stays consistent with implementation.
    from modules.constraints import pin_to_circle as mod

    pts = np.array([v.position for v in mesh.vertices.values()], dtype=float)
    fitted = mod._fit_circle_in_plane(pts, normal, radius_fixed=radius)
    assert fitted is not None
    fitted_center, fitted_radius = fitted
    assert np.isclose(fitted_radius, radius, atol=1e-6)

    distances = np.linalg.norm(pts - fitted_center[None, :], axis=1)
    assert np.allclose(distances, radius, atol=1e-6)


def test_pin_to_circle_slide_allows_only_normal_translation() -> None:
    """Slide mode: the rim can move along the fixed normal but not rotate."""
    radius = 2.0
    base_center = np.array([1.0, -2.0, 0.0], dtype=float)
    normal = np.array([0.0, 0.0, 1.0], dtype=float)

    # Start with a circle at z=0 but shift all points up to z=3.0 (translation along normal).
    angles = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    points = np.stack(
        [
            base_center[0] + radius * np.cos(angles),
            base_center[1] + radius * np.sin(angles),
            np.full_like(angles, 3.0),
        ],
        axis=1,
    )
    # Add noise that would normally cause a full fit to rotate/translate in-plane.
    rng = np.random.default_rng(1)
    points[:, :2] += 0.05 * rng.normal(size=(len(points), 2))

    data = {
        "vertices": [
            [
                float(p[0]),
                float(p[1]),
                float(p[2]),
                {
                    "constraints": ["pin_to_circle"],
                    "pin_to_circle_group": "rim",
                    "pin_to_circle_mode": "slide",
                },
            ]
            for p in points
        ],
        "edges": [[0, 1]],
        "faces": [],
        "global_parameters": {
            "pin_to_circle_normal": normal.tolist(),
            "pin_to_circle_point": base_center.tolist(),
            "pin_to_circle_radius": radius,
        },
    }
    mesh = parse_geometry(data)
    manager = ConstraintModuleManager(mesh.constraint_modules)
    manager.enforce_all(mesh)

    pts = np.array([v.position for v in mesh.vertices.values()], dtype=float)
    # Slide keeps the rim planar with fixed normal.
    assert float(np.ptp(pts[:, 2])) < 1e-10
    # And should keep the in-plane center fixed (points lie on the circle
    # around base_center, but can be non-uniformly distributed in angle).
    z0 = float(np.mean(pts[:, 2]))
    center = np.array([base_center[0], base_center[1], z0], dtype=float)
    distances = np.linalg.norm(pts - center[None, :], axis=1)
    assert np.allclose(distances, radius, atol=1e-6)


def test_pin_to_circle_refine_preserves_planar_rim_z() -> None:
    """Refinement should preserve pin_to_circle constraints on rim midpoints."""
    from runtime.refinement import refine_triangle_mesh

    radius = 1.0
    n = 8
    r_mid = 2.0
    r_out = 3.0

    vertices: list[list] = []
    for i in range(n):
        theta = 2.0 * np.pi * i / n
        # Deliberately non-planar inner rim; constraint should re-project to z=const.
        z = 0.2 if (i % 2 == 0) else -0.1
        vertices.append(
            [
                radius * np.cos(theta),
                radius * np.sin(theta),
                z,
                {
                    "constraints": ["pin_to_circle"],
                    "pin_to_circle_group": "inner",
                    "pin_to_circle_mode": "slide",
                },
            ]
        )
    for i in range(n):
        theta = 2.0 * np.pi * i / n
        vertices.append([r_mid * np.cos(theta), r_mid * np.sin(theta), 0.0])
    for i in range(n):
        theta = 2.0 * np.pi * i / n
        vertices.append([r_out * np.cos(theta), r_out * np.sin(theta), 0.0])

    edges: list[list] = []
    for base in (0, n, 2 * n):
        for i in range(n):
            edges.append([base + i, base + ((i + 1) % n)])
    for i in range(n):
        edges.append([i, n + i])
        edges.append([n + i, 2 * n + i])
    for i in range(n):
        edges.append([i, n + ((i + 1) % n)])
        edges.append([n + i, 2 * n + ((i + 1) % n)])

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
    for i in range(n):
        i1 = (i + 1) % n
        v_i, v_i1 = i, i1
        m_i, m_i1 = n + i, n + i1
        o_i, o_i1 = 2 * n + i, 2 * n + i1

        faces.append([edge_ref(v_i, v_i1), edge_ref(v_i1, m_i1), edge_ref(m_i1, v_i)])
        faces.append([edge_ref(v_i, m_i1), edge_ref(m_i1, m_i), edge_ref(m_i, v_i)])
        faces.append([edge_ref(m_i, m_i1), edge_ref(m_i1, o_i1), edge_ref(o_i1, m_i)])
        faces.append([edge_ref(m_i, o_i1), edge_ref(o_i1, o_i), edge_ref(o_i, m_i)])

    data = {
        "vertices": vertices,
        "edges": edges,
        "faces": faces,
        "global_parameters": {
            "pin_to_circle_normal": [0.0, 0.0, 1.0],
            "pin_to_circle_point": [0.0, 0.0, 0.0],
            "pin_to_circle_radius": radius,
        },
        "constraint_modules": ["pin_to_circle"],
    }

    mesh = parse_geometry(data)
    manager = ConstraintModuleManager(mesh.constraint_modules)
    manager.enforce_all(mesh)

    def rim_z_range(m) -> float:
        z = [
            float(v.position[2])
            for v in m.vertices.values()
            if getattr(v, "options", {}).get("pin_to_circle_group") == "inner"
        ]
        return float(np.ptp(np.asarray(z, dtype=float)))

    assert rim_z_range(mesh) < 1e-10

    refined = refine_triangle_mesh(mesh)
    manager = ConstraintModuleManager(refined.constraint_modules)
    manager.enforce_all(refined)
    assert rim_z_range(refined) < 1e-10


def test_pin_to_circle_constraint_gradients_match_finite_difference() -> None:
    data = {
        "vertices": [
            [1.4, 0.6, 0.3, {"constraints": ["pin_to_circle"]}],
            [0.0, 0.0, 0.0],
        ],
        "edges": [[0, 1]],
        "faces": [],
        "global_parameters": {
            "pin_to_circle_normal": [0.0, 0.0, 1.0],
            "pin_to_circle_point": [0.0, 0.0, 0.0],
            "pin_to_circle_radius": 1.5,
        },
    }
    mesh = parse_geometry(data)
    grads = pin_to_circle.constraint_gradients(mesh, mesh.global_parameters)
    assert grads is not None
    g_for_v = [g[0] for g in grads if 0 in g]
    assert len(g_for_v) >= 2
    g_plane = g_for_v[0]
    g_radial = g_for_v[1]

    x0 = mesh.vertices[0].position.copy()
    n = np.array([0.0, 0.0, 1.0], dtype=float)
    c = np.array([0.0, 0.0, 0.0], dtype=float)
    r = 1.5

    def residual_plane(x: np.ndarray) -> float:
        return float(np.dot(x - c, n))

    def residual_radial(x: np.ndarray) -> float:
        x_plane = x - float(np.dot(x - c, n)) * n
        return float(np.linalg.norm(x_plane - c) - r)

    eps = 1e-7
    fd_plane = np.zeros(3, dtype=float)
    fd_radial = np.zeros(3, dtype=float)
    for axis in range(3):
        xp = x0.copy()
        xm = x0.copy()
        xp[axis] += eps
        xm[axis] -= eps
        fd_plane[axis] = (residual_plane(xp) - residual_plane(xm)) / (2.0 * eps)
        fd_radial[axis] = (residual_radial(xp) - residual_radial(xm)) / (2.0 * eps)

    assert np.allclose(g_plane, fd_plane, rtol=1e-6, atol=1e-8)
    assert np.allclose(g_radial, fd_radial, rtol=1e-5, atol=1e-7)


def test_pin_to_circle_constraint_gradients_array_matches_dict() -> None:
    data = {
        "vertices": [
            [1.4, 0.6, 0.3, {"constraints": ["pin_to_circle"]}],
            [0.3, 1.3, -0.2, {"constraints": ["pin_to_circle"]}],
            [0.0, 0.0, 0.0],
        ],
        "edges": [[0, 2, {"constraints": ["pin_to_circle"]}]],
        "faces": [],
        "global_parameters": {
            "pin_to_circle_normal": [0.0, 0.0, 1.0],
            "pin_to_circle_point": [0.0, 0.0, 0.0],
            "pin_to_circle_radius": 1.5,
        },
    }
    mesh = parse_geometry(data)
    mesh.build_position_cache()
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row

    g_dict = pin_to_circle.constraint_gradients(mesh, mesh.global_parameters)
    g_arr = pin_to_circle.constraint_gradients_array(
        mesh,
        mesh.global_parameters,
        positions=positions,
        index_map=index_map,
    )
    assert g_dict is not None and g_arr is not None
    assert len(g_dict) == len(g_arr)

    for gC, gA in zip(g_dict, g_arr):
        dense = np.zeros_like(positions)
        for vidx, vec in gC.items():
            dense[index_map[int(vidx)]] += vec
        assert np.allclose(dense, gA, atol=1e-12, rtol=0.0)
