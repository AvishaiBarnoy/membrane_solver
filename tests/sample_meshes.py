import copy
import json
import math

import numpy as np

from core.parameters.global_parameters import GlobalParameters
from geometry.entities import Body, Edge, Facet, Mesh, Vertex
from geometry.geom_io import parse_geometry

SAMPLE_GEOMETRY = {
    "vertices": [
        [0, 0, 0],
        [1, 0, 0],
        [1, 0, 1],
        [0, 0, 1],
        [0, 1, 1],
        [0, 1, 0],
        [1, 1, 0],
        [1, 1, 1],
    ],
    "edges": [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 5],
        [1, 6],
        [2, 7],
        [3, 4],
    ],
    "faces": [
        [0, 1, 2, 3],
        ["r0", 8, 5, "r9"],
        [9, 6, -10, -1],
        [-2, 10, 7, -11],
        [11, 4, -8, -3],
        [-5, -4, -7, -6],
    ],
    "bodies": {
        "faces": [[0, 1, 2, 3, 4, 5]],
        "target_volume": [1.0],
    },
    "global_parameters": {
        "surface_tension": 1.0,
        "intrinsic_curvature": 0.0,
        "bending_modulus": 0.0,
        "gaussian_modulus": 0.0,
        "volume_stiffness": 1e3,
        "volume_constraint_mode": "lagrange",
    },
    "instructions": [],
}


def write_sample_geometry(tmp_path, name="sample_geometry.json", data=None):
    """Write SAMPLE_GEOMETRY (or provided data) to tmp_path/name."""
    path = tmp_path / name
    with open(path, "w") as f:
        json.dump(data or SAMPLE_GEOMETRY, f)
    return str(path)


def cube_soft_volume_input(volume_mode: str = "penalty") -> dict:
    """Return a deep copy of the cube sample with requested volume mode."""
    data = copy.deepcopy(SAMPLE_GEOMETRY)
    data.setdefault("global_parameters", {})
    if volume_mode == "penalty":
        projection = True
    else:
        projection = False
    data["global_parameters"].update(
        {
            "surface_tension": 1.0,
            "volume_constraint_mode": volume_mode,
            "volume_projection_during_minimization": projection,
        }
    )
    return data


def single_triangle_mesh() -> Mesh:
    """Return the canonical cached triangle used by focused runtime tests."""
    mesh = Mesh()
    mesh.vertices = {
        0: Vertex(0, np.array([0.0, 0.0, 0.0])),
        1: Vertex(1, np.array([1.0, 0.0, 0.0])),
        2: Vertex(2, np.array([0.0, 1.0, 0.0])),
    }
    mesh.edges = {1: Edge(1, 0, 1), 2: Edge(2, 1, 2), 3: Edge(3, 2, 0)}
    mesh.facets = {0: Facet(0, edge_indices=[1, 2, 3])}
    mesh.build_facet_vertex_loops()
    mesh.build_position_cache()
    return mesh


def two_triangle_square_mesh() -> Mesh:
    """Return a cached unit square split along the 0-to-2 diagonal."""
    mesh = Mesh()
    mesh.vertices = {
        0: Vertex(0, np.array([0.0, 0.0, 0.0])),
        1: Vertex(1, np.array([1.0, 0.0, 0.0])),
        2: Vertex(2, np.array([1.0, 1.0, 0.0])),
        3: Vertex(3, np.array([0.0, 1.0, 0.0])),
    }
    mesh.edges = {
        1: Edge(1, 0, 1),
        2: Edge(2, 1, 2),
        3: Edge(3, 2, 3),
        4: Edge(4, 3, 0),
        5: Edge(5, 0, 2),
    }
    mesh.facets = {
        0: Facet(0, edge_indices=[1, 2, -5]),
        1: Facet(1, edge_indices=[5, 3, 4]),
    }
    mesh.build_facet_vertex_loops()
    mesh.build_position_cache()
    return mesh


def set_mesh_positions(mesh: Mesh, positions: np.ndarray) -> None:
    """Update vertex positions in row order and invalidate geometry caches."""
    mesh.build_position_cache()
    if positions.shape != (len(mesh.vertex_ids), 3):
        raise ValueError("positions must have shape (N_vertices, 3)")

    for row, vid in enumerate(mesh.vertex_ids):
        mesh.vertices[int(vid)].position[:] = positions[row]
    mesh.increment_version()


def tetra_mesh_with_body() -> Mesh:
    """Return the asymmetric tetrahedron used by array/dict parity tests."""
    mesh = Mesh()
    points = np.array(
        [
            [0.1, 0.2, 0.05],
            [1.1, -0.1, 0.3],
            [0.4, 1.2, -0.2],
            [0.5, 0.4, 1.5],
        ],
        dtype=float,
    )
    for index, point in enumerate(points):
        mesh.vertices[index] = Vertex(index, point)

    faces = [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]]
    edge_map: dict[tuple[int, int], int] = {}
    next_edge = 1
    for facet_id, (a, b, c) in enumerate(faces):
        edge_ids = []
        for tail, head in ((a, b), (b, c), (c, a)):
            key = (min(tail, head), max(tail, head))
            if key not in edge_map:
                edge_map[key] = next_edge
                mesh.edges[next_edge] = Edge(next_edge, tail, head)
                next_edge += 1
            edge_id = edge_map[key]
            edge = mesh.edges[edge_id]
            edge_ids.append(edge_id if edge.tail_index == tail else -edge_id)
        mesh.facets[facet_id] = Facet(facet_id, edge_ids)

    mesh.bodies[0] = Body(0, list(mesh.facets.keys()), target_volume=0.5)
    mesh.build_connectivity_maps()
    mesh.build_facet_vertex_loops()
    return mesh


def square_mesh_with_center(*, z_offset: float) -> Mesh:
    """Return a four-triangle unit square with an adjustable center height."""
    mesh = Mesh()
    mesh.vertices = {
        0: Vertex(0, np.array([0.0, 0.0, 0.0])),
        1: Vertex(1, np.array([1.0, 0.0, 0.0])),
        2: Vertex(2, np.array([1.0, 1.0, 0.0])),
        3: Vertex(3, np.array([0.0, 1.0, 0.0])),
        4: Vertex(4, np.array([0.5, 0.5, float(z_offset)])),
    }
    mesh.edges = {
        1: Edge(1, 0, 1),
        2: Edge(2, 1, 2),
        3: Edge(3, 2, 3),
        4: Edge(4, 3, 0),
        5: Edge(5, 0, 4),
        6: Edge(6, 1, 4),
        7: Edge(7, 2, 4),
        8: Edge(8, 3, 4),
    }
    mesh.facets = {
        1: Facet(1, [1, 6, -5]),
        2: Facet(2, [2, 7, -6]),
        3: Facet(3, [3, 8, -7]),
        4: Facet(4, [4, 5, -8]),
    }
    mesh.build_connectivity_maps()
    mesh.build_facet_vertex_loops()
    return mesh


def single_vertex_mesh() -> Mesh:
    """Return the initialized one-vertex mesh used by manager tests."""
    mesh = Mesh()
    mesh.vertices = {0: Vertex(0, np.zeros(3, dtype=float))}
    mesh.edges = {}
    mesh.facets = {}
    mesh.energy_modules = []
    mesh.constraint_modules = []
    mesh.global_parameters = GlobalParameters()
    mesh.build_position_cache()
    return mesh


def ring_vertices(n_theta: int, radius: float, z: float, options: dict) -> list[list]:
    """Return option-tagged vertices on a horizontal ring."""
    vertices: list[list] = []
    for index in range(n_theta):
        theta = 2.0 * np.pi * index / float(n_theta)
        x = float(radius * np.cos(theta))
        y = float(radius * np.sin(theta))
        vertices.append([x, y, z, dict(options)])
    return vertices


def parsed_two_triangle_square_mesh() -> Mesh:
    """Return the parsed two-triangle square used by row/operation tests."""
    mesh = parse_geometry(
        {
            "vertices": [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            "edges": [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]],
            "faces": [[0, 1, "r4"], [4, 2, 3]],
        }
    )
    mesh.build_facet_vertex_loops()
    mesh.build_position_cache()
    return mesh


SQUARE_PERIMETER_GEOMETRY = {
    "vertices": [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    "edges": [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
    ],
    "faces": [[0, 1, 2, 3]],
    "bodies": {
        "faces": [[0]],
        "energy": [{"constraints": ["body_area"], "target_area": 1.0}],
    },
    "constraint_modules": ["perimeter"],
    "global_parameters": {
        "surface_tension": 0.5,
        "volume_constraint_mode": "penalty",
        "volume_projection_during_minimization": True,
        "perimeter_constraints": [{"edges": [1, 2, 3, 4], "target_perimeter": 4.0}],
    },
    "instructions": [],
}


def square_perimeter_input(target_perimeter: float = 4.0) -> dict:
    data = copy.deepcopy(SQUARE_PERIMETER_GEOMETRY)
    data["global_parameters"]["perimeter_constraints"][0]["target_perimeter"] = (
        target_perimeter
    )
    return data


def generate_open_cylinder(radius=1.0, height=2.0, n_segments=16):
    """
    Generate a triangulated open cylinder mesh (walls only).
    Top and bottom rings are fixed.
    """
    mesh = Mesh()

    # Vertices
    # Bottom ring (z = -height/2)
    # Top ring (z = height/2)
    z_bottom = -height / 2.0
    z_top = height / 2.0

    for i in range(n_segments):
        theta = 2.0 * math.pi * i / n_segments
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)

        # Bottom vertex (fixed)
        mesh.vertices[i] = Vertex(i, np.array([x, y, z_bottom]), fixed=True)

        # Top vertex (fixed)
        mesh.vertices[i + n_segments] = Vertex(
            i + n_segments, np.array([x, y, z_top]), fixed=True
        )

    # Edges
    edge_idx = 1

    # 1. Bottom ring edges
    b_ring_edges = []  # stored as indices
    for i in range(n_segments):
        mesh.edges[edge_idx] = Edge(edge_idx, i, (i + 1) % n_segments, fixed=True)
        b_ring_edges.append(edge_idx)
        edge_idx += 1

    # 2. Top ring edges
    t_ring_edges = []
    for i in range(n_segments):
        mesh.edges[edge_idx] = Edge(
            edge_idx, i + n_segments, (i + 1) % n_segments + n_segments, fixed=True
        )
        t_ring_edges.append(edge_idx)
        edge_idx += 1

    # 3. Vertical edges (b_i -> t_i)
    v_edges = []
    for i in range(n_segments):
        mesh.edges[edge_idx] = Edge(edge_idx, i, i + n_segments)
        v_edges.append(edge_idx)
        edge_idx += 1

    # 4. Diagonal edges (b_i+1 -> t_i)
    d_edges = []
    for i in range(n_segments):
        mesh.edges[edge_idx] = Edge(edge_idx, (i + 1) % n_segments, i + n_segments)
        d_edges.append(edge_idx)
        edge_idx += 1

    # Create Facets
    # Quad i: b_i, b_{i+1}, t_{i+1}, t_i
    # Triangle 1: b_i, b_{i+1}, t_i  (Using edges: b_ring[i], d_edge[i], -v_edge[i])
    # Triangle 2: b_{i+1}, t_{i+1}, t_i (Using edges: v_edge[i+1], -t_ring[i], -d_edge[i])

    facet_idx = 1
    facet_list = []
    for i in range(n_segments):
        # Tri 1: b_i -> b_{i+1} -> t_i
        # Edges:
        # b_i -> b_{i+1} is b_ring_edges[i]
        # b_{i+1} -> t_i is d_edges[i]
        # t_i -> b_i is -v_edges[i]

        f1 = Facet(facet_idx, [b_ring_edges[i], d_edges[i], -v_edges[i]])
        mesh.facets[facet_idx] = f1
        facet_list.append(facet_idx)
        facet_idx += 1

        # Tri 2: b_{i+1} -> t_{i+1} -> t_i
        # Edges:
        # b_{i+1} -> t_{i+1} is v_edges[(i+1)%n]
        # t_{i+1} -> t_i is -t_ring_edges[i]
        # t_i -> b_{i+1} is -d_edges[i]

        f2 = Facet(
            facet_idx, [v_edges[(i + 1) % n_segments], -t_ring_edges[i], -d_edges[i]]
        )
        mesh.facets[facet_idx] = f2
        facet_list.append(facet_idx)
        facet_idx += 1

    # Body
    # Calculate initial volume to set as target
    # Note: Body volume is computed using divergence theorem. For open cylinder,
    # it gives volume of cone from origin.
    # We set target_volume to this computed value to keep it "constant".

    mesh.bodies[1] = Body(1, facet_list, target_volume=0.0)  # Will update target later

    mesh.build_connectivity_maps()
    mesh.build_facet_vertex_loops()

    # Calculate initial volume
    initial_vol = mesh.bodies[1].compute_volume(mesh)
    mesh.bodies[1].target_volume = initial_vol

    # Global params
    mesh.global_parameters.set("surface_tension", 1.0)
    mesh.global_parameters.set("volume_constraint_mode", "lagrange")
    mesh.global_parameters.set("volume_stiffness", 1000.0)

    return mesh


def square_annulus_mesh() -> Mesh:
    """Return a small planar annulus mesh (outer square minus inner square).

    The mesh is a triangulated topological cylinder with two boundary loops.
    It is useful for Gauss–Bonnet regression tests because the total invariant
    should be 0 (Euler characteristic χ=0), with opposite-signed boundary
    contributions from the outer and inner loops.
    """
    mesh = Mesh()

    outer_xy = [
        (0.0, 0.0),
        (1.0, 0.0),
        (2.0, 0.0),
        (2.0, 1.0),
        (2.0, 2.0),
        (1.0, 2.0),
        (0.0, 2.0),
        (0.0, 1.0),
    ]
    inner_xy = [
        (0.75, 0.75),
        (1.0, 0.75),
        (1.25, 0.75),
        (1.25, 1.0),
        (1.25, 1.25),
        (1.0, 1.25),
        (0.75, 1.25),
        (0.75, 1.0),
    ]

    for vid, (x, y) in enumerate([*outer_xy, *inner_xy]):
        mesh.vertices[vid] = Vertex(vid, np.array([x, y, 0.0]))

    triangles: list[tuple[int, int, int]] = []
    for k in range(8):
        o0 = k
        o1 = (k + 1) % 8
        i0 = 8 + k
        i1 = 8 + ((k + 1) % 8)
        triangles.append((o0, o1, i1))
        triangles.append((o0, i1, i0))

    edge_map: dict[tuple[int, int], int] = {}
    next_eid = 1
    for fidx, tri in enumerate(triangles):
        e_ids: list[int] = []
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            key = tuple(sorted((a, b)))
            if key not in edge_map:
                edge_map[key] = next_eid
                mesh.edges[next_eid] = Edge(next_eid, a, b)
                next_eid += 1
            eid = edge_map[key]
            edge = mesh.edges[eid]
            e_ids.append(eid if edge.tail_index == a and edge.head_index == b else -eid)
        mesh.facets[fidx] = Facet(fidx, e_ids)

    mesh.build_connectivity_maps()
    mesh.build_facet_vertex_loops()
    return mesh
