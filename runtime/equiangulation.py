import numpy as np
import logging
from geometry.entities import Mesh, Edge, Facet

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
        The mesh with improved triangulation
    """
    logger.info("Starting equiangulation process...")
    
    # Ensure mesh has connectivity maps
    mesh.build_connectivity_maps()
    
    iteration = 0
    changes_made = True
    
    while changes_made and iteration < max_iterations:
        changes_made = False
        iteration += 1
        
        # Get list of edges to check (copy to avoid modification during iteration)
        edges_to_check = list(mesh.edges.keys())
        
        for edge_idx in edges_to_check:
            if edge_idx not in mesh.edges:
                continue  # Edge may have been modified/removed
                
            edge = mesh.edges[edge_idx]
            
            # Get facets adjacent to this edge
            adjacent_facets = mesh.get_facets_of_edge(edge_idx)
            
            # Only process edges with exactly 2 adjacent triangular facets
            if len(adjacent_facets) != 2:
                continue
                
            facet1, facet2 = adjacent_facets
            
            # Ensure both facets are triangles
            if len(facet1.edge_indices) != 3 or len(facet2.edge_indices) != 3:
                continue
                
            # Check if edge should be flipped using Delaunay criterion
            if should_flip_edge(mesh, edge, facet1, facet2):
                flip_edge(mesh, edge_idx, facet1, facet2)
                changes_made = True
                logger.debug(f"Flipped edge {edge_idx}")
    
    if iteration >= max_iterations:
        logger.warning(f"Equiangulation reached maximum iterations ({max_iterations})")
    else:
        logger.info(f"Equiangulation completed in {iteration} iterations")
    
    # Rebuild connectivity maps after modifications
    mesh.build_connectivity_maps()
    return mesh


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
    c1 = np.linalg.norm(pos2 - pos1)      # shared edge (common edge)
    
    # Triangle 2: (v1, v2, off_vertex2)
    a2 = np.linalg.norm(pos2 - pos_off2)  # edge opposite to v1
    d2 = np.linalg.norm(pos1 - pos_off2)  # edge from off_vertex2 to v1
    e2 = np.linalg.norm(pos2 - pos1)      # shared edge (common edge)
    
    # Calculate cos θ₁ and cos θ₂ using law of cosines
    # For triangle 1, angle at off_vertex1: cos θ₁ = (b1² + c1² - a1²) / (2*b1*c1)
    # For triangle 2, angle at off_vertex2: cos θ₂ = (d2² + e2² - a2²) / (2*d2*e2)
    
    # Avoid division by zero
    if b1 * c1 == 0 or d2 * e2 == 0:
        return False
    
    cos_theta1 = (b1**2 + c1**2 - a1**2) / (2 * b1 * c1)
    cos_theta2 = (d2**2 + e2**2 - a2**2) / (2 * d2 * e2)
    
    # Apply Delaunay criterion: flip if cos θ₁ + cos θ₂ < 0
    return cos_theta1 + cos_theta2 < 0


def get_off_vertex(mesh: Mesh, facet: Facet, edge: Edge) -> int | None:
    """
    Find the vertex in the facet that is not on the given edge.
    """
    edge_vertices = {edge.tail_index, edge.head_index}
    
    # Get all vertices of the facet
    facet_vertices = set()
    for signed_edge_idx in facet.edge_indices:
        e = mesh.get_edge(signed_edge_idx)
        facet_vertices.add(e.tail_index)
        facet_vertices.add(e.head_index)
    
    # Find the vertex not on the edge
    off_vertices = facet_vertices - edge_vertices
    
    if len(off_vertices) != 1:
        logger.warning(f"Expected 1 off vertex, found {len(off_vertices)} in facet {facet.index}")
        return None
    
    return off_vertices.pop()


def flip_edge(mesh: Mesh, edge_idx: int, facet1: Facet, facet2: Facet):
    """
    Flip an edge by replacing it with the other diagonal of the quadrilateral.
    
    This removes the current edge and two triangular facets, then creates a new edge
    and two new triangular facets.
    """
    edge = mesh.edges[edge_idx]
    v1, v2 = edge.tail_index, edge.head_index
    
    # Get the off vertices
    off_vertex1 = get_off_vertex(mesh, facet1, edge)
    off_vertex2 = get_off_vertex(mesh, facet2, edge)
    
    if off_vertex1 is None or off_vertex2 is None:
        logger.warning(f"Cannot flip edge {edge_idx}: invalid off vertices")
        return
    
    # Create new edge connecting the off vertices
    new_edge_idx = max(mesh.edges.keys()) + 1
    new_edge = Edge(
        index=new_edge_idx,
        tail_index=off_vertex1,
        head_index=off_vertex2,
        fixed=edge.fixed,
        options=edge.options.copy()
    )
    
    # Remove the old edge
    del mesh.edges[edge_idx]
    
    # Add the new edge
    mesh.edges[new_edge_idx] = new_edge
    
    # Create new facets
    # We need to find the correct edges for the new triangles
    
    # Get all edges of both original facets
    edges1 = [abs(ei) for ei in facet1.edge_indices if abs(ei) != edge_idx]
    edges2 = [abs(ei) for ei in facet2.edge_indices if abs(ei) != edge_idx]
    
    # Find edges connecting v1 and v2 to the off vertices
    edge_v1_off1 = find_edge_between_vertices(mesh, v1, off_vertex1, edges1)
    edge_v2_off1 = find_edge_between_vertices(mesh, v2, off_vertex1, edges1)
    edge_v1_off2 = find_edge_between_vertices(mesh, v1, off_vertex2, edges2)
    edge_v2_off2 = find_edge_between_vertices(mesh, v2, off_vertex2, edges2)
    
    # Check if all required edges were found
    if None in [edge_v1_off1, edge_v2_off1, edge_v1_off2, edge_v2_off2]:
        logger.warning(f"Cannot flip edge {edge_idx}: missing required edges")
        return
    
    # Create new triangle 1: (v1, off_vertex1, off_vertex2)
    new_facet1_edges = [
        get_oriented_edge_index(mesh, v1, off_vertex1, edge_v1_off1),
        new_edge_idx,  # off_vertex1 to off_vertex2
        get_oriented_edge_index(mesh, off_vertex2, v1, edge_v1_off2)
    ]
    
    # Create new triangle 2: (v2, off_vertex2, off_vertex1) 
    new_facet2_edges = [
        get_oriented_edge_index(mesh, v2, off_vertex2, edge_v2_off2),
        -new_edge_idx,  # off_vertex2 to off_vertex1 (reversed)
        get_oriented_edge_index(mesh, off_vertex1, v2, edge_v2_off1)
    ]
    
    # Replace the old facets with new ones
    facet1.edge_indices = new_facet1_edges
    facet2.edge_indices = new_facet2_edges


def find_edge_between_vertices(mesh: Mesh, v1: int, v2: int, candidate_edges: list) -> int | None:
    """
    Find an edge that connects two vertices from a list of candidate edges.
    """
    for edge_idx in candidate_edges:
        edge = mesh.edges[edge_idx]
        if (edge.tail_index == v1 and edge.head_index == v2) or \
           (edge.tail_index == v2 and edge.head_index == v1):
            return edge_idx
    
    logger.error(f"Could not find edge between vertices {v1} and {v2}")
    return None


def get_oriented_edge_index(mesh: Mesh, from_vertex: int, to_vertex: int, edge_idx: int) -> int:
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
        logger.error(f"Edge {edge_idx} does not connect vertices {from_vertex} and {to_vertex}")
        return edge_idx  # fallback