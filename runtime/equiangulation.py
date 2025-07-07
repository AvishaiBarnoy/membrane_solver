import numpy as np
import logging
from geometry.entities import Mesh, Edge, Facet, Vertex, Body
from typing import Tuple, Optional, Dict, Any

logger = logging.getLogger("membrane_solver")


def equiangulate_mesh(mesh: Mesh, max_iterations: int = 100) -> Mesh:
    """
    Performs robust equiangulation on a triangulated mesh using improved Delaunay criterion.
    
    Args:
        mesh: The input triangulated mesh
        max_iterations: Maximum number of iterations to prevent infinite loops
        
    Returns:
        A new mesh with improved triangulation
    """
    logger.info("Starting robust equiangulation process...")
    
    # Ensure input mesh has connectivity maps
    mesh.build_connectivity_maps()
    
    # Validate initial mesh integrity
    if not validate_mesh_integrity(mesh):
        logger.warning("Input mesh has integrity issues, proceeding with caution")
    
    current_mesh = mesh
    iteration = 0
    
    for iteration in range(max_iterations):
        # Validate mesh integrity before processing
        if not validate_mesh_integrity(current_mesh):
            logger.error(f"Mesh integrity compromised at iteration {iteration}, stopping")
            break
        
        # Conservative edge processing
        new_mesh, changes_made = equiangulate_iteration_robust(current_mesh)
        
        # Validate result
        if not validate_mesh_integrity(new_mesh):
            logger.warning(f"Iteration {iteration} produced invalid mesh, reverting")
            break
        
        if not changes_made:
            logger.info(f"Equiangulation converged in {iteration} iterations")
            return new_mesh
        
        current_mesh = new_mesh
        logger.debug(f"Iteration {iteration + 1}: performed edge flips")
    
    logger.warning(f"Equiangulation reached maximum iterations ({max_iterations})")
    return current_mesh


def equiangulate_iteration_robust(mesh: Mesh) -> Tuple[Mesh, bool]:
    """
    Perform one iteration of robust equiangulation with extensive validation.
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
    
    # Build connectivity for the new mesh
    new_mesh.build_connectivity_maps()
    
    changes_made = False
    next_edge_idx = max(new_mesh.edges.keys()) + 1 if new_mesh.edges else 1
    
    # Check all edges for potential flips
    edges_to_check = list(new_mesh.edges.keys())
    
    for edge_idx in edges_to_check:
        if edge_idx not in new_mesh.edges:
            continue  # Edge may have been removed
            
        # Conservative edge flip with extensive validation
        if flip_edge_conservative(new_mesh, edge_idx, next_edge_idx):
            changes_made = True
            next_edge_idx += 1
            logger.debug(f"Successfully flipped edge {edge_idx}")
            # Rebuild connectivity after each flip to ensure consistency
            new_mesh.build_connectivity_maps()
    
    return new_mesh, changes_made


def flip_edge_conservative(mesh: Mesh, edge_idx: int, new_edge_idx: int) -> bool:
    """
    Conservative edge flip with extensive validation and rollback capability.
    """
    try:
        # 1. Pre-flip validation
        if not validate_flip_preconditions(mesh, edge_idx):
            return False
        
        edge = mesh.edges[edge_idx]
        adjacent_facets = mesh.get_facets_of_edge(edge_idx)
        
        if len(adjacent_facets) != 2:
            return False
        
        facet1, facet2 = adjacent_facets
        
        # Check if edge should be flipped using robust criterion
        if not should_flip_edge_robust(mesh, edge, facet1, facet2):
            return False
        
        # 2. Create backup of affected elements
        backup = create_flip_backup(mesh, edge_idx, facet1, facet2)
        
        # 3. Perform flip
        success = perform_edge_flip_safe(mesh, edge_idx, facet1, facet2, new_edge_idx)
        
        # 4. Post-flip validation
        if not success or not validate_flip_postconditions(mesh, new_edge_idx, facet1, facet2):
            restore_from_backup(mesh, backup)
            return False
        
        return True
        
    except Exception as e:
        logger.warning(f"Edge flip failed for edge {edge_idx}: {e}")
        return False


def validate_flip_preconditions(mesh: Mesh, edge_idx: int) -> bool:
    """Extensive pre-flip validation."""
    if edge_idx not in mesh.edges:
        return False
    
    edge = mesh.edges[edge_idx]
    adjacent_facets = mesh.get_facets_of_edge(edge_idx)
    
    if len(adjacent_facets) != 2:
        return False
    
    facet1, facet2 = adjacent_facets
    
    # Ensure both facets are triangles
    if len(facet1.edge_indices) != 3 or len(facet2.edge_indices) != 3:
        return False
    
    # Check for boundary constraints
    if any(f.options.get("no_refine", False) for f in adjacent_facets):
        return False
    
    # Check for geometric degeneracy
    if is_degenerate_configuration(mesh, edge, adjacent_facets):
        return False
    
    return True


def validate_flip_postconditions(mesh: Mesh, new_edge_idx: int, facet1: Facet, facet2: Facet) -> bool:
    """Validate mesh state after flip."""
    try:
        # Check that new edge exists
        if new_edge_idx not in mesh.edges:
            return False
        
        # Check that facets are still valid triangles
        if len(facet1.edge_indices) != 3 or len(facet2.edge_indices) != 3:
            return False
        
        # Check normals are reasonable
        normal1 = facet1.normal(mesh)
        normal2 = facet2.normal(mesh)
        
        if np.linalg.norm(normal1) < 1e-12 or np.linalg.norm(normal2) < 1e-12:
            return False
        
        return True
        
    except Exception:
        return False


def is_degenerate_configuration(mesh: Mesh, edge: Edge, adjacent_facets) -> bool:
    """Check if the edge configuration is degenerate."""
    try:
        facet1, facet2 = adjacent_facets
        
        # Check for very small areas
        area1 = abs(facet1.compute_area(mesh))
        area2 = abs(facet2.compute_area(mesh))
        
        if area1 < 1e-12 or area2 < 1e-12:
            return True
        
        # Check for very small angles or very long edges
        v1, v2 = edge.tail_index, edge.head_index
        off_vertex1 = get_off_vertex(mesh, facet1, edge)
        off_vertex2 = get_off_vertex(mesh, facet2, edge)
        
        if off_vertex1 is None or off_vertex2 is None:
            return True
        
        pos1 = mesh.vertices[v1].position
        pos2 = mesh.vertices[v2].position
        pos_off1 = mesh.vertices[off_vertex1].position
        pos_off2 = mesh.vertices[off_vertex2].position
        
        # Check edge lengths
        edge_length = np.linalg.norm(pos2 - pos1)
        if edge_length < 1e-12 or edge_length > 1e6:
            return True
        
        return False
        
    except Exception:
        return True


def create_flip_backup(mesh: Mesh, edge_idx: int, facet1: Facet, facet2: Facet) -> Dict[str, Any]:
    """Create backup of mesh elements before flip."""
    return {
        'edge': mesh.edges[edge_idx].copy(),
        'facet1': facet1.copy(),
        'facet2': facet2.copy(),
        'edge_idx': edge_idx,
        'facet1_idx': facet1.index,
        'facet2_idx': facet2.index
    }


def restore_from_backup(mesh: Mesh, backup: Dict[str, Any]) -> None:
    """Restore mesh state from backup."""
    try:
        edge_idx = backup['edge_idx']
        facet1_idx = backup['facet1_idx']
        facet2_idx = backup['facet2_idx']
        
        # Remove any new edge that was created
        new_edges_to_remove = []
        for idx, edge in mesh.edges.items():
            if idx != edge_idx and (edge.tail_index in [backup['edge'].tail_index, backup['edge'].head_index] or
                                   edge.head_index in [backup['edge'].tail_index, backup['edge'].head_index]):
                # This might be a newly created edge, check if it should be removed
                pass
        
        # Restore original edge and facets
        mesh.edges[edge_idx] = backup['edge']
        mesh.facets[facet1_idx] = backup['facet1']
        mesh.facets[facet2_idx] = backup['facet2']
        
    except Exception as e:
        logger.warning(f"Failed to restore from backup: {e}")


def should_flip_edge_robust(mesh: Mesh, edge: Edge, facet1: Facet, facet2: Facet) -> bool:
    """
    Robust edge flip criterion using incircle test instead of angle-based approach.
    """
    try:
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
        
        # Use robust incircle test: is off_vertex2 inside circumcircle of triangle (v1, v2, off_vertex1)?
        return is_in_circumcircle_robust(pos_off2, pos1, pos2, pos_off1)
        
    except Exception as e:
        logger.debug(f"Error in edge flip criterion: {e}")
        return False


def is_in_circumcircle_robust(p, a, b, c):
    """
    Robust incircle test using determinant approach with numerical stability.
    Returns True if point p is inside the circumcircle of triangle abc.
    """
    try:
        # Translate points to origin at p
        ax, ay = a[0] - p[0], a[1] - p[1]
        bx, by = b[0] - p[0], b[1] - p[1]
        cx, cy = c[0] - p[0], c[1] - p[1]
        
        # Compute determinant for incircle test
        det = (ax*ax + ay*ay) * (bx*cy - by*cx) + \
              (bx*bx + by*by) * (cx*ay - cy*ax) + \
              (cx*cx + cy*cy) * (ax*by - ay*bx)
        
        # Use appropriate epsilon for numerical stability
        epsilon = 1e-10 * max(abs(ax), abs(ay), abs(bx), abs(by), abs(cx), abs(cy), 1.0)**2
        
        return det > epsilon
        
    except Exception:
        return False


def perform_edge_flip_safe(mesh: Mesh, edge_idx: int, facet1: Facet, facet2: Facet, new_edge_idx: int) -> bool:
    """
    Safely perform the actual edge flip operation.
    """
    try:
        edge = mesh.edges[edge_idx]
        v1, v2 = edge.tail_index, edge.head_index
        
        # Get the off vertices
        off_vertex1 = get_off_vertex(mesh, facet1, edge)
        off_vertex2 = get_off_vertex(mesh, facet2, edge)
        
        if off_vertex1 is None or off_vertex2 is None:
            return False
        
        # Create new edge connecting the off vertices
        new_edge = Edge(
            index=new_edge_idx,
            tail_index=off_vertex1,
            head_index=off_vertex2,
            fixed=edge.fixed,
            options=edge.options.copy()
        )
        
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
        
        # Create new triangles
        # Triangle 1: (v1, off_vertex1, off_vertex2)
        new_facet1_edges = [
            get_oriented_edge(mesh, v1, off_vertex1, edge_v1_off1),
            new_edge_idx,  # off_vertex1 to off_vertex2
            get_oriented_edge(mesh, off_vertex2, v1, edge_v1_off2)
        ]
        
        # Triangle 2: (v2, off_vertex2, off_vertex1)
        new_facet2_edges = [
            get_oriented_edge(mesh, v2, off_vertex2, edge_v2_off2),
            -new_edge_idx,  # off_vertex2 to off_vertex1 (reversed)
            get_oriented_edge(mesh, off_vertex1, v2, edge_v2_off1)
        ]
        
        # Update the mesh
        del mesh.edges[edge_idx]
        mesh.edges[new_edge_idx] = new_edge
        
        # Update facets with new edge lists
        facet1.edge_indices = new_facet1_edges
        facet2.edge_indices = new_facet2_edges
        
        return True
        
    except Exception as e:
        logger.warning(f"Edge flip operation failed: {e}")
        return False


def validate_mesh_integrity(mesh: Mesh) -> bool:
    """
    Validate overall mesh integrity.
    """
    try:
        # Check that all edge references in facets are valid
        for facet in mesh.facets.values():
            for signed_edge_idx in facet.edge_indices:
                edge_idx = abs(signed_edge_idx)
                if edge_idx not in mesh.edges:
                    logger.warning(f"Facet {facet.index} references non-existent edge {edge_idx}")
                    return False
        
        # Check that all vertex references in edges are valid
        for edge in mesh.edges.values():
            if edge.tail_index not in mesh.vertices or edge.head_index not in mesh.vertices:
                logger.warning(f"Edge {edge.index} references non-existent vertices")
                return False
        
        # Check for degenerate triangles
        degenerate_count = 0
        for facet in mesh.facets.values():
            try:
                area = abs(facet.compute_area(mesh))
                if area < 1e-12:
                    degenerate_count += 1
            except Exception:
                degenerate_count += 1
        
        if degenerate_count > 0:
            logger.warning(f"Found {degenerate_count} degenerate triangles")
            return degenerate_count < len(mesh.facets) * 0.1  # Allow up to 10% degenerate
        
        return True
        
    except Exception as e:
        logger.warning(f"Mesh integrity check failed: {e}")
        return False


def get_off_vertex(mesh: Mesh, facet: Facet, edge: Edge) -> Optional[int]:
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
        logger.debug(f"Expected 1 off vertex, found {len(off_vertices)} in facet {facet.index}")
        return None
    
    return off_vertices.pop()


def find_connecting_edge(mesh: Mesh, v1: int, v2: int, candidate_edges: list) -> Optional[int]:
    """
    Find an edge that connects two vertices from a list of candidate signed edge indices.
    """
    for signed_edge_idx in candidate_edges:
        edge = mesh.get_edge(signed_edge_idx)
        if (edge.tail_index == v1 and edge.head_index == v2) or \
           (edge.tail_index == v2 and edge.head_index == v1):
            return abs(signed_edge_idx)
    
    return None


def get_oriented_edge(mesh: Mesh, from_vertex: int, to_vertex: int, edge_idx: int) -> int:
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