import logging

import numpy as np

from geometry.entities import Edge, Facet, Mesh

logger = logging.getLogger("membrane_solver")


def equiangulate_mesh(mesh: Mesh, max_iterations: int = 100) -> Mesh:
    """
    Performs equiangulation on a triangulated mesh using the Delaunay criterion.

    For any edge with two adjacent triangular facets, we switch the edge to the other
    diagonal of the quadrilateral if the sum of the angles at the off vertices is more
    than π. This is equivalent to cos θ₁ + cos θ₂ < 0.

    Args:
        mesh: The input triangulated mesh
        max_iterations: Maximum number of iterations to prevent infinite loops

    Returns:
        A new mesh with improved triangulation
    """
    logger.info("Starting equiangulation process...")

    # Sanity‑check the input mesh before we start modifying anything.
    try:
        mesh.build_connectivity_maps()
        if hasattr(mesh, "build_facet_vertex_loops"):
            mesh.build_facet_vertex_loops()
        if hasattr(mesh, "full_mesh_validate"):
            mesh.full_mesh_validate()
    except Exception as exc:  # defensive guardrail
        logger.warning(
            "Skipping equiangulation: mesh validation failed before start: %s",
            exc,
        )
        return mesh

    current_mesh = mesh
    iteration = 0

    for iteration in range(max_iterations):
        # Try one iteration of edge flips
        new_mesh, changes_made = equiangulate_iteration(current_mesh)

        if not changes_made:
            logger.info(f"Equiangulation converged in {iteration} iterations")
            # Validate final mesh; if broken, fall back to the original.
            try:
                if hasattr(new_mesh, "full_mesh_validate"):
                    new_mesh.full_mesh_validate()
            except Exception as exc:  # defensive guardrail
                logger.error(
                    "Mesh validation failed after equiangulation "
                    "(returning original mesh): %s",
                    exc,
                )
                return mesh
            return new_mesh

        current_mesh = new_mesh
        logger.debug(f"Iteration {iteration + 1}: performed edge flips")

    logger.warning(f"Equiangulation reached maximum iterations ({max_iterations})")
    # Final validation even if we hit the iteration cap.
    try:
        if hasattr(current_mesh, "full_mesh_validate"):
            current_mesh.full_mesh_validate()
    except Exception as exc:  # defensive guardrail
        logger.error(
            "Mesh validation failed after equiangulation (returning original mesh): %s",
            exc,
        )
        return mesh
    return current_mesh


def equiangulate_iteration(mesh: Mesh) -> tuple[Mesh, bool]:
    """
    Perform one iteration of equiangulation, returning a new mesh and whether changes were made.
    """
    # Create new mesh by copying the current one
    new_mesh = Mesh()
    new_mesh.vertices = {idx: v.copy() for idx, v in mesh.vertices.items()}
    new_mesh.edges = {idx: e.copy() for idx, e in mesh.edges.items()}
    new_mesh.facets = {idx: f.copy() for idx, f in mesh.facets.items()}
    new_mesh.bodies = {idx: b.copy() for idx, b in mesh.bodies.items()}
    new_mesh.global_parameters = mesh.global_parameters
    new_mesh.energy_modules = mesh.energy_modules[:]
    new_mesh.constraint_modules = mesh.constraint_modules[:]
    new_mesh.instructions = mesh.instructions[:]
    new_mesh.macros = getattr(mesh, "macros", {}).copy()

    # Build connectivity and facet loops for the new mesh
    new_mesh.build_connectivity_maps()
    if hasattr(new_mesh, "build_facet_vertex_loops"):
        new_mesh.build_facet_vertex_loops()

    changes_made = False
    next_edge_idx = max(new_mesh.edges.keys()) + 1 if new_mesh.edges else 1

    # Check all edges for potential flips
    edges_to_check = list(new_mesh.edges.keys())

    for edge_idx in edges_to_check:
        if edge_idx not in new_mesh.edges:
            continue  # Edge may have been removed

        edge = new_mesh.edges[edge_idx]

        # Respect fixed edges, as in Evolver.
        if getattr(edge, "fixed", False):
            continue

        # Get facets adjacent to this edge
        adjacent_facets = new_mesh.get_facets_of_edge(edge_idx)

        # Only process edges with exactly 2 adjacent triangular facets
        if len(adjacent_facets) != 2:
            continue

        facet1, facet2 = adjacent_facets

        # Ensure both facets are triangles
        if len(facet1.edge_indices) != 3 or len(facet2.edge_indices) != 3:
            continue

        # Check if edge should be flipped using Delaunay criterion
        if should_flip_edge(new_mesh, edge, facet1, facet2):
            # Perform the edge flip
            if flip_edge_safe(new_mesh, edge_idx, facet1, facet2, next_edge_idx):
                changes_made = True
                next_edge_idx += 1
                logger.debug(f"Flipped edge {edge_idx}")
                # Rebuild connectivity after each flip to ensure consistency
                new_mesh.build_connectivity_maps()
                if hasattr(new_mesh, "build_facet_vertex_loops"):
                    new_mesh.build_facet_vertex_loops()

    return new_mesh, changes_made


def should_flip_edge(mesh: Mesh, edge: Edge, facet1: Facet, facet2: Facet) -> bool:
    """
    Determines if an edge should be flipped using the Delaunay criterion.

    Returns True if cos θ₁ + cos θ₂ < 0, where θ₁ and θ₂ are the angles
    at the off vertices (vertices not on the shared edge).
    """
    # Get vertices of the quadrilateral
    v1, v2 = edge.tail_index, edge.head_index

    # Find the off vertices (vertices not on the shared edge)
    off_vertex1 = get_off_vertex(mesh, facet1, edge)
    off_vertex2 = get_off_vertex(mesh, facet2, edge)

    if off_vertex1 is None or off_vertex2 is None:
        return False

    # Get vertex positions
    pos1 = mesh.vertices[v1].position
    pos2 = mesh.vertices[v2].position
    pos_off1 = mesh.vertices[off_vertex1].position
    pos_off2 = mesh.vertices[off_vertex2].position

    # Calculate side lengths for both triangles
    # Triangle 1: (v1, v2, off_vertex1)
    a1 = np.linalg.norm(pos2 - pos_off1)  # edge opposite to v1
    b1 = np.linalg.norm(pos1 - pos_off1)  # edge from off_vertex1 to v1
    c1 = np.linalg.norm(pos2 - pos1)  # shared edge (common edge)

    # Triangle 2: (v1, v2, off_vertex2)
    a2 = np.linalg.norm(pos2 - pos_off2)  # edge opposite to v1
    d2 = np.linalg.norm(pos1 - pos_off2)  # edge from off_vertex2 to v1
    e2 = np.linalg.norm(pos2 - pos1)  # shared edge (common edge)

    # Calculate cos θ₁ and cos θ₂ using law of cosines
    # For triangle 1, angle at off_vertex1: cos θ₁ = (b1² + c1² - a1²) / (2*b1*c1)
    # For triangle 2, angle at off_vertex2: cos θ₂ = (d2² + e2² - a2²) / (2*d2*e2)

    # Avoid division by zero / degenerate configurations
    if b1 * c1 <= 0.0 or d2 * e2 <= 0.0:
        return False

    cos_theta1 = (b1**2 + c1**2 - a1**2) / (2 * b1 * c1)
    cos_theta2 = (d2**2 + e2**2 - a2**2) / (2 * d2 * e2)

    # Apply Delaunay/equiangular criterion with a small negative margin,
    # similar in spirit to Evolver's -0.001 tolerance, to avoid cycling
    # on nearly-flat quadrilaterals.
    delaunay_margin = 1e-3
    return (cos_theta1 + cos_theta2) < -delaunay_margin


def get_off_vertex(mesh: Mesh, facet: Facet, edge: Edge) -> int | None:
    """
    Find the vertex in ``facet`` that is not on ``edge``.

    Returns ``None`` (and skips flipping) if the facet is not a proper
    triangle or if the configuration around the edge is inconsistent.
    """
    # We only support equiangulation on triangles; anything else is skipped.
    if len(facet.edge_indices) != 3:
        logger.warning(
            "get_off_vertex: facet %d is not a triangle (has %d edges); "
            "skipping equiangulation on this facet.",
            facet.index,
            len(facet.edge_indices),
        )
        return None

    edge_vertices = {edge.tail_index, edge.head_index}

    # Collect all distinct vertices referenced by the facet's edges.
    facet_vertices: set[int] = set()
    for signed_edge_idx in facet.edge_indices:
        base_edge = mesh.edges[abs(signed_edge_idx)]
        facet_vertices.add(base_edge.tail_index)
        facet_vertices.add(base_edge.head_index)

    if len(facet_vertices) != 3:
        logger.warning(
            "get_off_vertex: facet %d does not have exactly 3 unique vertices "
            "(found %d); skipping.",
            facet.index,
            len(facet_vertices),
        )
        return None

    # The off-vertex is the one not lying on the shared edge.
    off_vertices = facet_vertices - edge_vertices

    if len(off_vertices) != 1:
        logger.warning(
            "Expected 1 off vertex, found %d in facet %d",
            len(off_vertices),
            facet.index,
        )
        return None

    return off_vertices.pop()


def flip_edge_safe(
    mesh: Mesh, edge_idx: int, facet1: Facet, facet2: Facet, new_edge_idx: int
) -> bool:
    """
    Safely flip an edge by replacing it with the other diagonal of the
    quadrilateral.  Returns True if the flip was successful, False otherwise.

    On failure, the mesh must be left completely unchanged.
    """
    try:
        edge = mesh.edges[edge_idx]
        v1, v2 = edge.tail_index, edge.head_index

        # Get the off vertices
        off_vertex1 = get_off_vertex(mesh, facet1, edge)
        off_vertex2 = get_off_vertex(mesh, facet2, edge)

        if off_vertex1 is None or off_vertex2 is None:
            return False

        # Store original facet normals for validation
        try:
            normal1_orig = facet1.normal(mesh)
            normal2_orig = facet2.normal(mesh)
        except ValueError:
            # Skip degenerate facets
            return False

        # Find the other edges of each triangle (excluding the edge being flipped)
        facet1_other_edges = [ei for ei in facet1.edge_indices if abs(ei) != edge_idx]
        facet2_other_edges = [ei for ei in facet2.edge_indices if abs(ei) != edge_idx]

        # Find edges connecting the shared edge vertices to the off vertices
        edge_v1_off1 = find_connecting_edge(mesh, v1, off_vertex1, facet1_other_edges)
        edge_v2_off1 = find_connecting_edge(mesh, v2, off_vertex1, facet1_other_edges)
        edge_v1_off2 = find_connecting_edge(mesh, v1, off_vertex2, facet2_other_edges)
        edge_v2_off2 = find_connecting_edge(mesh, v2, off_vertex2, facet2_other_edges)

        if None in [edge_v1_off1, edge_v2_off1, edge_v1_off2, edge_v2_off2]:
            return False

        # At this point, all edges are guaranteed to be not None
        assert edge_v1_off1 is not None and edge_v2_off1 is not None
        assert edge_v1_off2 is not None and edge_v2_off2 is not None

        # Build the new diagonal edge (off_vertex1 -- off_vertex2).
        new_edge = Edge(
            index=new_edge_idx,
            tail_index=off_vertex1,
            head_index=off_vertex2,
            fixed=edge.fixed,
            options=edge.options.copy(),
        )

        # Construct the new oriented edge lists for the two replacement facets.
        # Triangle 1: (v1, off_vertex1, off_vertex2)
        new_facet1_edges = [
            get_oriented_edge(mesh, v1, off_vertex1, edge_v1_off1),
            new_edge_idx,  # off_vertex1 to off_vertex2
            get_oriented_edge(mesh, off_vertex2, v1, edge_v1_off2),
        ]

        # Triangle 2: (v2, off_vertex2, off_vertex1)
        new_facet2_edges = [
            get_oriented_edge(mesh, v2, off_vertex2, edge_v2_off2),
            -new_edge_idx,  # off_vertex2 to off_vertex1 (reversed)
            get_oriented_edge(mesh, off_vertex1, v2, edge_v2_off1),
        ]

        # Save original state so we can revert if validation fails.
        old_edge = edge
        old_facet1_edges = list(facet1.edge_indices)
        old_facet2_edges = list(facet2.edge_indices)

        # Apply the flip.
        del mesh.edges[edge_idx]
        mesh.edges[new_edge_idx] = new_edge
        facet1.edge_indices = new_facet1_edges
        facet2.edge_indices = new_facet2_edges

        # Validate that the new triangles have consistent normals.
        try:
            new_normal1 = facet1.normal(mesh)
            new_normal2 = facet2.normal(mesh)

            # Normals should be roughly in the same hemisphere as originals.
            if (
                np.dot(new_normal1, normal1_orig) < -0.5
                or np.dot(new_normal2, normal2_orig) < -0.5
            ):
                logger.warning(
                    "Edge flip created inverted normals, reverting edge %d",
                    edge_idx,
                )
                # Revert and report failure.
                del mesh.edges[new_edge_idx]
                mesh.edges[edge_idx] = old_edge
                facet1.edge_indices = old_facet1_edges
                facet2.edge_indices = old_facet2_edges
                return False

        except ValueError:
            # New triangles are degenerate; revert and report failure.
            del mesh.edges[new_edge_idx]
            mesh.edges[edge_idx] = old_edge
            facet1.edge_indices = old_facet1_edges
            facet2.edge_indices = old_facet2_edges
            return False

        return True

    except Exception as e:  # defensive
        logger.warning(f"Edge flip failed for edge {edge_idx}: {e}")
        return False


def find_connecting_edge(
    mesh: Mesh, v1: int, v2: int, candidate_edges: list
) -> int | None:
    """
    Find an edge that connects two vertices from a list of candidate signed edge indices.
    """
    for signed_edge_idx in candidate_edges:
        edge = mesh.get_edge(signed_edge_idx)
        if (edge.tail_index == v1 and edge.head_index == v2) or (
            edge.tail_index == v2 and edge.head_index == v1
        ):
            return abs(signed_edge_idx)

    return None


def get_oriented_edge(
    mesh: Mesh, from_vertex: int, to_vertex: int, edge_idx: int
) -> int:
    """
    Get the correctly oriented edge index for going from from_vertex to to_vertex.
    Returns positive index if edge goes from->to, negative if edge goes to->from.
    """
    edge = mesh.edges[edge_idx]

    if edge.tail_index == from_vertex and edge.head_index == to_vertex:
        return edge_idx
    elif edge.tail_index == to_vertex and edge.head_index == from_vertex:
        return -edge_idx
    else:
        logger.error(
            f"Edge {edge_idx} does not connect vertices {from_vertex} and {to_vertex}"
        )
        return edge_idx  # fallback
