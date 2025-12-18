import copy
import json
import math

import numpy as np

from geometry.entities import Body, Edge, Facet, Mesh, Vertex

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
