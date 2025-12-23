import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from geometry.entities import Body, Edge, Facet, Mesh, Vertex
from runtime.refinement import refine_polygonal_facets, refine_triangle_mesh


def create_quad():
    mesh = Mesh()

    # A unit square in the XY plane
    v0 = Vertex(0, np.array([0, 0, 0]))
    v1 = Vertex(1, np.array([1, 0, 0]))
    v2 = Vertex(2, np.array([1, 1, 0]))
    v3 = Vertex(3, np.array([0, 1, 0]))
    vertices = [v0, v1, v2, v3]

    e0 = Edge(1, v0.index, v1.index)
    e1 = Edge(2, v1.index, v2.index)
    e2 = Edge(3, v2.index, v3.index)
    e3 = Edge(4, v3.index, v0.index)
    edges = [e0, e1, e2, e3]

    facet = Facet(0, [e0.index, e1.index, e2.index, e3.index])
    facets = [facet]

    body = Body(0, [facets[0].index], options={"target_volume": 0})
    bodies = [body]

    mesh = Mesh()
    for i in vertices:
        mesh.vertices[i.index] = i
    for i in edges:
        mesh.edges[i.index] = i
    for i in facets:
        mesh.facets[i.index] = i
    for i in bodies:
        mesh.bodies[i.index] = i

    return mesh


def test_triangle_refinement_updates_bodies():
    mesh = create_quad()

    # Testing polygonal refinement
    mesh_tri = refine_polygonal_facets(mesh)
    assert len(mesh_tri.vertices) == len(mesh.vertices) + len(mesh.facets), (
        "Initial triangulation of square should add a vertex at centroid, 5 total."
    )
    assert len(mesh_tri.edges) == len(mesh.edges) * 2, (
        "Initial triangulation of square should end with 8 edges."
    )
    assert all(
        len(mesh_tri.facets[f_idx].edge_indices) == 3
        for f_idx in mesh_tri.facets.keys()
    ), "All refined facets must be triangles"
    assert len(mesh_tri.facets) == len(mesh.vertices), (
        "Initial triangulation of square should end with 4 facets."
    )
    # assert all(isinstance(f, Facet) for f in mesh.b_tri[0].facets), "All body facets must be Facets"
    assert len(mesh_tri.bodies[0].facet_indices) == len(mesh_tri.facets), (
        "Body should include all refined facets"
    )

    # Testing triangular refinement
    mesh_ref = refine_triangle_mesh(mesh_tri)

    assert len(mesh_ref.vertices) == len(mesh_tri.vertices) + len(mesh_tri.edges), (
        "Refinemenet should add len(edges) new vertex per facet"
    )
    assert len(mesh_ref.edges) == 2 * len(mesh_tri.edges) + 3 * len(mesh_tri.facets), (
        "Refining splits edges and adds 3 more for each facet"
    )
    assert len(mesh_ref.facets) == 2 ** len(mesh_tri.facets), (
        "Refiningt increases number of facets by factor of 2^k"
    )
    assert all(
        len(mesh_ref.facets[f_idx].edge_indices) == 3
        for f_idx in mesh_ref.facets.keys()
    ), "All refined facets must be triangles"
    assert len(mesh_ref.bodies[0].facet_indices) == len(mesh_ref.facets), (
        "Body should include all refined facets"
    )


def test_child_facets_are_closed_loops():
    mesh = create_quad()

    # 1. check loop on initial triangulation
    mesh2 = refine_polygonal_facets(mesh)
    for facet_idx in mesh2.facets.keys():
        # grab the three edges in order
        # check chaining: edge.head == next_edge.tail (mod 3)
        facet = mesh2.facets[facet_idx]

        for i in range(3):
            e_curr = mesh2.get_edge(facet.edge_indices[i])
            e_next = mesh2.get_edge(facet.edge_indices[(i + 1) % 3])
            assert e_curr.head_index == e_next.tail_index, (
                f"Facet {facet.index} is not a closed loop: "
                f"edge {e_curr.index}.head={e_curr.head_index!r} ≠ "
                f"edge {e_next.index}.tail={e_next.tail_index!r}"
            )

    # 2. check loop in runtime triangulation
    mesh3 = refine_polygonal_facets(mesh2)

    for facet_idx in mesh3.facets.keys():
        # grab the three edges in order
        # check chaining: edge.head == next_edge.tail (mod 3)
        facet = mesh2.facets[facet_idx]
        for i in range(3):
            e_curr = mesh3.get_edge(facet.edge_indices[i])
            e_next = mesh3.get_edge(facet.edge_indices[(i + 1) % 3])
            assert e_curr.head_index == e_next.tail_index, (
                f"Facet {facet.index} is not a closed loop: "
                f"edge {e_curr.index}.head={e_curr.head_index!r} ≠ "
                f"edge {e_next.index}.tail={e_next.tail_index!r}"
            )


def test_edge_and_vertex_options_inheritance_triangle_refinement():
    mesh = create_quad()
    # Set constraints and fixed on an edge and its vertices
    mesh.edges[1].options["constraints"] = ["test_constraint"]
    mesh.edges[1].fixed = True
    mesh.vertices[0].options["constraints"] = ["test_constraint"]
    mesh.vertices[1].options["constraints"] = ["test_constraint"]
    mesh.vertices[0].fixed = True
    mesh.vertices[1].fixed = True

    mesh_tri = refine_polygonal_facets(mesh)
    mesh_ref = refine_triangle_mesh(mesh_tri)

    # Find the midpoint vertex created on edge 1 (between v0 and v1)
    v0, v1 = mesh.vertices[0], mesh.vertices[1]
    midpoint = None
    for v in mesh_ref.vertices.values():
        if np.allclose(v.position, 0.5 * (v0.position + v1.position)):
            midpoint = v
            break
    assert midpoint is not None, "Midpoint vertex not found"
    # Should inherit the constraint and be fixed
    assert "test_constraint" in midpoint.options.get("constraints", []), (
        "Midpoint should inherit constraint"
    )
    # Should be fixed if the edge is fixed, regardless of parent vertices
    assert midpoint.fixed, (
        "Midpoint should be fixed if edge is fixed, regardless of parent vertices"
    )

    # Check that new edges splitting edge 1 inherit options and fixed
    found = False
    for e in mesh_ref.edges.values():
        if (e.tail_index, e.head_index) in [
            (v0.index, midpoint.index),
            (midpoint.index, v1.index),
        ] or (e.head_index, e.tail_index) in [
            (v0.index, midpoint.index),
            (midpoint.index, v1.index),
        ]:
            found = True
            assert "test_constraint" in e.options.get("constraints", []), (
                "Child edge should inherit constraint"
            )
            assert e.fixed, "Child edge should be fixed"
    assert found, "Child edges of split should be present"


def test_facet_options_inheritance_polygonal_refinement():
    mesh = create_quad()
    # Set a constraint and energy on the facet
    mesh.facets[0].options["constraints"] = ["facet_constraint"]
    mesh.facets[0].options["energy"] = ["facet_energy"]

    mesh_tri = refine_polygonal_facets(mesh)
    # All child facets should inherit the parent's options
    for f in mesh_tri.facets.values():
        assert "facet_constraint" in f.options.get("constraints", []), (
            "Child facet should inherit constraint"
        )
        assert "facet_energy" in f.options.get("energy", []), (
            "Child facet should inherit energy"
        )


def test_middle_edge_inherits_facet_constraint_polygonal_refinement():
    mesh = create_quad()
    mesh.facets[0].options["constraints"] = ["facet_constraint"]

    mesh_tri = refine_polygonal_facets(mesh)
    # Middle edges are those that connect to the centroid (index 4)
    centroid_idx = 4
    for e in mesh_tri.edges.values():
        if centroid_idx in (e.tail_index, e.head_index):
            assert "facet_constraint" in e.options.get("constraints", []), (
                "Middle edge should inherit facet constraint"
            )


def test_centroid_vertex_inherits_facet_constraint_options():
    mesh = create_quad()
    mesh.facets[0].options["constraints"] = ["pin_to_circle"]
    mesh.facets[0].options["pin_to_circle_normal"] = [0.0, 1.0, 0.0]
    mesh.facets[0].options["pin_to_circle_point"] = [0.5, 0.0, 0.5]
    mesh.facets[0].options["pin_to_circle_radius"] = 0.2

    mesh_tri = refine_polygonal_facets(mesh)

    centroid = None
    for vertex in mesh_tri.vertices.values():
        if vertex.index not in mesh.vertices:
            centroid = vertex
            break

    assert centroid is not None, "Centroid vertex not found"
    assert "pin_to_circle" in centroid.options.get("constraints", [])
    assert centroid.options.get("pin_to_circle_normal") == [0.0, 1.0, 0.0]
    assert centroid.options.get("pin_to_circle_point") == [0.5, 0.0, 0.5]
    assert centroid.options.get("pin_to_circle_radius") == 0.2


def test_midpoint_fixed_if_edge_fixed_even_if_vertices_not_fixed():
    mesh = create_quad()
    # Only the edge is fixed, not the vertices
    mesh.edges[1].options["constraints"] = ["test_constraint"]
    mesh.edges[1].fixed = True
    mesh.vertices[0].fixed = False
    mesh.vertices[1].fixed = False

    mesh_tri = refine_polygonal_facets(mesh)
    mesh_ref = refine_triangle_mesh(mesh_tri)

    v0, v1 = mesh.vertices[0], mesh.vertices[1]
    midpoint = None
    for v in mesh_ref.vertices.values():
        if np.allclose(v.position, 0.5 * (v0.position + v1.position)):
            midpoint = v
            break
    assert midpoint is not None, "Midpoint vertex not found"
    assert midpoint.fixed, (
        "Midpoint should be fixed if edge is fixed, even if parent vertices are not"
    )


def test_connectivity_maps_after_polygonal_refinement():
    mesh = create_quad()

    # Run polygonal refinement (quad → 4 triangles)
    mesh_tri = refine_polygonal_facets(mesh)

    # Ensure connectivity maps were built
    mesh_tri.build_connectivity_maps()

    # 1. Each vertex maps to the correct number of facets
    for v_id, facets in mesh_tri.vertex_to_facets.items():
        for f_id in facets:
            facet = mesh_tri.facets[f_id]
            # Reconstruct all vertex IDs used by the facet
            v_used = set()
            for signed_ei in facet.edge_indices:
                edge = mesh_tri.get_edge(signed_ei)
                v_used.add(edge.tail_index)
                v_used.add(edge.head_index)
            assert v_id in v_used, f"Vertex {v_id} not actually in facet {f_id}"

    # 2. Each edge maps to a facet that includes it
    for e_id, facets in mesh_tri.edge_to_facets.items():
        for f_id in facets:
            assert e_id in [abs(i) for i in mesh_tri.facets[f_id].edge_indices], (
                f"Edge {e_id} not found in facet {f_id}'s edge list"
            )

    # 3. Each vertex's edge map is consistent
    for v_id, edges in mesh_tri.vertex_to_edges.items():
        for e_id in edges:
            edge = mesh_tri.edges[e_id]
            assert v_id in (
                edge.tail_index,
                edge.head_index,
            ), f"Vertex {v_id} not in edge {e_id}"


def create_two_triangles():
    mesh = Mesh()

    # First triangle (will be skipped in refinement)
    v0 = Vertex(0, np.array([0.0, 0.0, 0.0]))
    v1 = Vertex(1, np.array([1.0, 0.0, 0.0]))
    v2 = Vertex(2, np.array([0.0, 1.0, 0.0]))

    e0 = Edge(1, 0, 1)
    e1 = Edge(2, 1, 2)
    e2 = Edge(3, 2, 0)

    f0 = Facet(0, [1, 2, 3], options={"no_refine": True})

    # Second triangle (will be refined)
    v3 = Vertex(3, np.array([2.0, 0.0, 0.0]))
    v4 = Vertex(4, np.array([3.0, 0.0, 0.0]))
    v5 = Vertex(5, np.array([2.0, 1.0, 0.0]))

    e3 = Edge(4, 3, 4)
    e4 = Edge(5, 4, 5)
    e5 = Edge(6, 5, 3)

    f1 = Facet(1, [4, 5, 6])

    for v in [v0, v1, v2, v3, v4, v5]:
        mesh.vertices[v.index] = v
    for e in [e0, e1, e2, e3, e4, e5]:
        mesh.edges[e.index] = e
    for f in [f0, f1]:
        mesh.facets[f.index] = f
    mesh.bodies[0] = Body(0, [0, 1], options={"target_volume": 0})

    return mesh


def test_polygonal_facets_triangulated_even_with_no_refine():
    mesh = create_quad()
    mesh.facets[0].options["no_refine"] = True
    mesh_tri = refine_polygonal_facets(mesh)
    assert all(len(f.edge_indices) == 3 for f in mesh_tri.facets.values())


def test_no_refine_skips_triangle_refinement():
    mesh = create_two_triangles()
    mesh_tri = refine_polygonal_facets(mesh)  # should do nothing
    mesh_ref = refine_triangle_mesh(mesh_tri)

    # Expect one facet unchanged and the other split into four
    assert len(mesh_ref.facets) == 5
    # Unrefined facet should remain a triangle
    assert any(
        f.options.get("no_refine") and len(f.edge_indices) == 3
        for f in mesh_ref.facets.values()
    )
    # Vertex and edge counts correspond to refining only one triangle
    assert len(mesh_ref.vertices) == 9
    assert len(mesh_ref.edges) == 12
