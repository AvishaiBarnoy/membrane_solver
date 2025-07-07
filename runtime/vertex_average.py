# runtime/vertex_average.py

import numpy as np
import logging
from typing import Set, List, Dict, Optional

logger = logging.getLogger("membrane_solver")

def compute_facet_centroid(mesh, facet, vertices):
    """Compute centroid of a facet."""
    v_ids = set()
    for signed_ei in facet.edge_indices:
        edge = mesh.get_edge(signed_ei)
        v_ids.update([edge.tail_index, edge.head_index])
    coords = np.array([vertices[i].position for i in v_ids])
    return np.mean(coords, axis=0)

def compute_facet_normal(mesh, facet, vertices):
    """Compute area-weighted normal of a facet."""
    # Use the first 3 vertices to compute a normal
    v_ids = []
    for signed_ei in facet.edge_indices:
        edge = mesh.get_edge(signed_ei)
        v_ids.append(edge.tail_index)
        if len(set(v_ids)) >= 3:
            break
    a = vertices[v_ids[1]].position - vertices[v_ids[0]].position
    b = vertices[v_ids[2]].position - vertices[v_ids[0]].position
    return 0.5 * np.cross(a, b)  # area-weighted normal

def vertex_average(mesh):
    """
    Enhanced volume-conserving vertex averaging with quality awareness.
    Uses adaptive smoothing based on local mesh quality.
    """
    logger.info("Starting adaptive vertex averaging...")
    
    # First pass: analyze local quality for all vertices
    quality_threshold = 0.2
    problematic_vertices = []
    
    for v_id, vertex in mesh.vertices.items():
        if vertex.fixed:
            continue
        
        local_quality = analyze_local_quality(mesh, v_id)
        if local_quality < quality_threshold:
            problematic_vertices.append(v_id)
    
    logger.info(f"Found {len(problematic_vertices)} vertices with poor local quality")
    
    # Second pass: apply appropriate smoothing
    for v_id, vertex in mesh.vertices.items():
        if vertex.fixed:
            continue
        
        if v_id in problematic_vertices:
            # Use more aggressive smoothing for poor quality areas
            success = laplacian_smoothing_cotangent(mesh, v_id, damping=0.3)
            if not success:
                # Fallback to volume-conserving if cotangent fails
                vertex_average_single(mesh, v_id)
        else:
            # Use volume-conserving for stable areas
            vertex_average_single(mesh, v_id)
    
    logger.info("Adaptive vertex averaging completed.")

def vertex_average_single(mesh, v_id):
    """
    Perform volume-conserving vertex averaging on a single vertex.
    This is the original implementation for a single vertex.
    """
    vertex = mesh.vertices[v_id]
    if vertex.fixed:
        return
    
    facet_ids = mesh.vertex_to_facets.get(v_id, [])
    if not facet_ids:
        return

    total_area = 0.0
    weighted_sum = np.zeros(3)
    total_normal = np.zeros(3)

    for f_id in facet_ids:
        facet = mesh.facets[f_id]
        centroid = compute_facet_centroid(mesh, facet, mesh.vertices)
        normal = compute_facet_normal(mesh, facet, mesh.vertices)
        area = np.linalg.norm(normal)

        weighted_sum += area * centroid
        total_normal += normal
        total_area += area

    if total_area < 1e-12 or np.linalg.norm(total_normal) < 1e-12:
        return

    v_avg = weighted_sum / total_area
    v = vertex.position
    lambda_ = (np.dot(v_avg, total_normal) - np.dot(v, total_normal)) / np.dot(total_normal, total_normal)
    v_new = v_avg - lambda_ * total_normal

    mesh.vertices[v_id].position = v_new

def laplacian_smoothing_cotangent(mesh, vertex_id: int, damping: float = 0.5) -> bool:
    """
    Cotangent-weighted Laplacian smoothing for a single vertex.
    Returns True if successful, False if failed.
    """
    try:
        vertex = mesh.vertices[vertex_id]
        if vertex.fixed:
            return True
        
        # Get neighboring vertices
        neighbors = get_vertex_neighbors(mesh, vertex_id)
        if len(neighbors) < 3:
            return False
        
        weighted_sum = np.zeros(3)
        total_weight = 0.0
        
        for neighbor_id in neighbors:
            # Compute cotangent weights from adjacent triangles
            weight = compute_cotangent_weight(mesh, vertex_id, neighbor_id)
            if weight <= 0:
                continue
            
            weighted_sum += weight * mesh.vertices[neighbor_id].position
            total_weight += weight
        
        if total_weight > 1e-12:
            new_pos = vertex.position * (1 - damping) + (weighted_sum / total_weight) * damping
            mesh.vertices[vertex_id].position = new_pos
        
        return True
        
    except Exception as e:
        logger.debug(f"Cotangent smoothing failed for vertex {vertex_id}: {e}")
        return False

def compute_cotangent_weight(mesh, vi: int, vj: int) -> float:
    """Compute cotangent weight between two adjacent vertices."""
    try:
        shared_faces = find_shared_faces(mesh, vi, vj)
        total_weight = 0.0
        
        for face in shared_faces:
            # Find the third vertex in the triangle
            vk = find_third_vertex(mesh, face, vi, vj)
            if vk is None:
                continue
            
            # Compute cotangent of angle at vk
            pi = mesh.vertices[vi].position
            pj = mesh.vertices[vj].position
            pk = mesh.vertices[vk].position
            
            edge_ki = pi - pk
            edge_kj = pj - pk
            
            norm_ki = np.linalg.norm(edge_ki)
            norm_kj = np.linalg.norm(edge_kj)
            
            if norm_ki < 1e-12 or norm_kj < 1e-12:
                continue
            
            cos_angle = np.dot(edge_ki, edge_kj) / (norm_ki * norm_kj)
            cos_angle = np.clip(cos_angle, -0.999, 0.999)
            
            if abs(cos_angle) < 0.999:  # Avoid division by zero
                sin_angle = np.sqrt(1 - cos_angle**2)
                if sin_angle > 1e-12:
                    cot_weight = cos_angle / sin_angle
                    total_weight += cot_weight
        
        return max(total_weight, 1e-6)  # Avoid negative or zero weights
        
    except Exception:
        return 1e-6

def analyze_local_quality(mesh, vertex_id: int) -> float:
    """
    Analyze the quality of triangles around a vertex.
    Returns a quality score between 0 and 1 (higher is better).
    """
    try:
        facet_ids = mesh.vertex_to_facets.get(vertex_id, [])
        if not facet_ids:
            return 1.0
        
        total_quality = 0.0
        valid_facets = 0
        
        for facet_id in facet_ids:
            facet = mesh.facets[facet_id]
            quality = compute_triangle_quality(mesh, facet)
            if quality >= 0:  # Valid quality measurement
                total_quality += quality
                valid_facets += 1
        
        if valid_facets == 0:
            return 0.0
        
        return total_quality / valid_facets
        
    except Exception:
        return 0.0

def compute_triangle_quality(mesh, facet) -> float:
    """
    Compute mean ratio quality metric for a triangle.
    Returns value between 0 and 1 (1 = equilateral triangle).
    """
    try:
        vertices = get_facet_vertices(mesh, facet)
        if len(vertices) != 3:
            return 0.0
        
        p1, p2, p3 = vertices
        
        # Edge lengths
        a = np.linalg.norm(p2 - p3)
        b = np.linalg.norm(p1 - p3)
        c = np.linalg.norm(p1 - p2)
        
        # Area using cross product
        area = 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))
        
        if area < 1e-12 or (a + b + c) < 1e-12:
            return 0.0
        
        # Mean ratio quality: 4*sqrt(3)*area / (a² + b² + c²)
        quality = (4 * np.sqrt(3) * area) / (a*a + b*b + c*c)
        return min(quality, 1.0)  # Clamp to [0, 1]
        
    except Exception:
        return 0.0

def get_facet_vertices(mesh, facet) -> List[np.ndarray]:
    """Get vertex positions for a facet."""
    v_ids = []
    for signed_ei in facet.edge_indices:
        edge = mesh.get_edge(signed_ei)
        v_ids.append(edge.tail_index)
    
    # Remove duplicates while preserving order
    unique_v_ids = []
    seen = set()
    for v_id in v_ids:
        if v_id not in seen:
            unique_v_ids.append(v_id)
            seen.add(v_id)
    
    return [mesh.vertices[v_id].position for v_id in unique_v_ids]

def get_vertex_neighbors(mesh, vertex_id: int) -> Set[int]:
    """Get all vertices connected to the given vertex by an edge."""
    neighbors = set()
    edge_ids = mesh.vertex_to_edges.get(vertex_id, set())
    
    for edge_id in edge_ids:
        edge = mesh.edges[edge_id]
        if edge.tail_index == vertex_id:
            neighbors.add(edge.head_index)
        elif edge.head_index == vertex_id:
            neighbors.add(edge.tail_index)
    
    return neighbors

def find_shared_faces(mesh, vi: int, vj: int) -> List:
    """Find faces that contain both vertices vi and vj."""
    faces_vi = mesh.vertex_to_facets.get(vi, set())
    faces_vj = mesh.vertex_to_facets.get(vj, set())
    
    shared_face_ids = faces_vi.intersection(faces_vj)
    return [mesh.facets[fid] for fid in shared_face_ids]

def find_third_vertex(mesh, facet, vi: int, vj: int) -> Optional[int]:
    """Find the third vertex in a triangle given two vertices."""
    v_ids = set()
    for signed_ei in facet.edge_indices:
        edge = mesh.get_edge(signed_ei)
        v_ids.add(edge.tail_index)
        v_ids.add(edge.head_index)
    
    other_vertices = v_ids - {vi, vj}
    
    if len(other_vertices) == 1:
        return other_vertices.pop()
    return None

# Alternative smoothing methods for comparison

def mean_curvature_flow(mesh, vertex_id: int, time_step: float = 0.01) -> bool:
    """
    Move vertex along mean curvature normal.
    Returns True if successful, False if failed.
    """
    try:
        vertex = mesh.vertices[vertex_id]
        if vertex.fixed:
            return True
        
        # Compute discrete mean curvature vector
        mean_curvature_vector = compute_discrete_mean_curvature(mesh, vertex_id)
        
        if np.linalg.norm(mean_curvature_vector) > 1e-12:
            # Move vertex along curvature direction
            new_pos = vertex.position + time_step * mean_curvature_vector
            mesh.vertices[vertex_id].position = new_pos
        
        return True
        
    except Exception as e:
        logger.debug(f"Mean curvature flow failed for vertex {vertex_id}: {e}")
        return False

def compute_discrete_mean_curvature(mesh, vertex_id: int) -> np.ndarray:
    """Compute discrete mean curvature using Meyer et al. formula."""
    try:
        neighbors = get_vertex_neighbors(mesh, vertex_id)
        curvature_vector = np.zeros(3)
        
        for neighbor_id in neighbors:
            edge_vector = mesh.vertices[neighbor_id].position - mesh.vertices[vertex_id].position
            cotangent_weight = compute_cotangent_weight(mesh, vertex_id, neighbor_id)
            curvature_vector += cotangent_weight * edge_vector
        
        # Normalize by approximate Voronoi area
        voronoi_area = compute_voronoi_area_approx(mesh, vertex_id)
        if voronoi_area > 1e-12:
            curvature_vector /= (2 * voronoi_area)
        
        return curvature_vector
        
    except Exception:
        return np.zeros(3)

def compute_voronoi_area_approx(mesh, vertex_id: int) -> float:
    """Compute approximate Voronoi area for a vertex."""
    try:
        facet_ids = mesh.vertex_to_facets.get(vertex_id, [])
        total_area = 0.0
        
        for facet_id in facet_ids:
            facet = mesh.facets[facet_id]
            area = facet.compute_area(mesh)
            total_area += area / 3.0  # Approximate as 1/3 of each adjacent triangle
        
        return max(total_area, 1e-12)
        
    except Exception:
        return 1e-12

# Export the main function
__all__ = ['vertex_average', 'vertex_average_single', 'laplacian_smoothing_cotangent', 
           'mean_curvature_flow', 'analyze_local_quality', 'compute_triangle_quality']
