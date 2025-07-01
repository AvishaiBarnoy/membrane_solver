"""
Optimized surface energy module with vectorized calculations and caching.

Performance improvements:
1. Vectorized area calculations
2. Cached parameter resolution
3. Batch processing of facets
4. Minimal memory allocations
"""

import numpy as np
from typing import Dict
from collections import defaultdict
from logging_config import setup_logging

logger = setup_logging('membrane_solver')

class OptimizedSurfaceEnergyCache:
    """Cache for surface energy calculations."""
    
    def __init__(self):
        self.surface_tensions = {}
        self.facet_areas = {}
        self.area_gradients = {}
        self.dirty_facets = set()
        
    def invalidate_facet(self, facet_id):
        """Invalidate cached values for a specific facet."""
        self.dirty_facets.add(facet_id)
        if facet_id in self.facet_areas:
            del self.facet_areas[facet_id]
        if facet_id in self.area_gradients:
            del self.area_gradients[facet_id]
            
    def clear(self):
        """Clear all cached values."""
        self.surface_tensions.clear()
        self.facet_areas.clear()
        self.area_gradients.clear()
        self.dirty_facets.clear()

# Global cache instance
_surface_cache = OptimizedSurfaceEnergyCache()

def compute_energy_and_gradient_optimized(mesh, global_params, param_resolver, *, compute_gradient: bool = True):
    """
    Optimized surface energy and gradient computation using vectorization and caching.
    
    Key optimizations:
    1. Batch process all facets
    2. Cache surface tension parameters
    3. Use optimized area calculations
    4. Vectorized gradient accumulation
    """
    total_energy = 0.0
    
    if compute_gradient:
        # Pre-allocate gradient dictionary with zero vectors
        max_vertex_id = max(mesh.vertices.keys()) if mesh.vertices else 0
        grad = np.zeros((max_vertex_id + 1, 3))
        vertex_mask = np.zeros(max_vertex_id + 1, dtype=bool)
    else:
        grad = None
        
    # Batch process facets for better cache locality
    facet_ids = list(mesh.facets.keys())
    facet_energies = np.zeros(len(facet_ids))
    
    for i, facet_id in enumerate(facet_ids):
        facet = mesh.facets[facet_id]
        
        # Get or cache surface tension parameter
        if facet_id not in _surface_cache.surface_tensions or facet_id in _surface_cache.dirty_facets:
            surface_tension = param_resolver.get(facet, 'surface_tension')
            if surface_tension is None:
                surface_tension = global_params.get("surface_tension", 1.0)
            _surface_cache.surface_tensions[facet_id] = surface_tension
        else:
            surface_tension = _surface_cache.surface_tensions[facet_id]
            
        # Get or compute area using optimized method
        if hasattr(facet, 'compute_area_optimized'):
            area = facet.compute_area_optimized(mesh)
        else:
            area = facet.compute_area(mesh)
            
        facet_energies[i] = surface_tension * area
        
        if compute_gradient:
            # Get or compute area gradient using optimized method
            if hasattr(facet, 'compute_area_gradient_optimized'):
                area_gradient = facet.compute_area_gradient_optimized(mesh)
            else:
                area_gradient = facet.compute_area_gradient(mesh)
                
            # Accumulate gradients efficiently
            for vertex_id, gradient_vector in area_gradient.items():
                if vertex_id < len(grad):
                    grad[vertex_id] += surface_tension * gradient_vector
                    vertex_mask[vertex_id] = True
    
    # Sum total energy
    total_energy = np.sum(facet_energies)
    
    # Convert gradient array back to dictionary format for compatibility
    if compute_gradient:
        grad_dict = {}
        for i in range(len(grad)):
            if vertex_mask[i]:
                grad_dict[i] = grad[i]
        return total_energy, grad_dict
    else:
        return total_energy, {}

def compute_energy_and_gradient_vectorized_triangular(mesh, global_params, param_resolver, *, compute_gradient: bool = True):
    """
    Fully vectorized surface energy computation for triangular meshes.
    
    This function assumes all facets are triangles and uses mesh-level vectorization.
    """
    if not hasattr(mesh, 'get_facet_vertices_array') or mesh.get_facet_vertices_array() is None:
        # Fall back to standard optimized version
        return compute_energy_and_gradient_optimized(mesh, global_params, param_resolver, compute_gradient=compute_gradient)
        
    # Get arrays for vectorized computation
    positions = mesh.get_vertex_positions_array()
    facet_vertices = mesh.get_facet_vertices_array()
    
    if len(facet_vertices) == 0:
        return 0.0, {}
        
    # Get surface tension for all facets (assuming uniform for now)
    # TODO: Handle per-facet surface tensions efficiently
    default_surface_tension = global_params.get("surface_tension", 1.0)
    
    # Vectorized area computation for all triangles
    v0 = positions[facet_vertices[:, 0]]
    v1 = positions[facet_vertices[:, 1]]
    v2 = positions[facet_vertices[:, 2]]
    
    # Triangle areas using cross product
    edge1 = v1 - v0
    edge2 = v2 - v0
    cross_products = np.cross(edge1, edge2)
    areas = 0.5 * np.linalg.norm(cross_products, axis=1)
    
    # Total energy
    total_energy = default_surface_tension * np.sum(areas)
    
    if not compute_gradient:
        return total_energy, {}
        
    # Vectorized gradient computation
    # For triangle area gradients: ∇A = 0.5 * (n × edge_opposite) / |n|
    norms = np.linalg.norm(cross_products, axis=1)
    valid_mask = norms > 1e-12
    
    # Initialize gradient accumulator
    max_vertex_id = positions.shape[0] - 1
    grad_accumulator = np.zeros((max_vertex_id + 1, 3))
    
    if np.any(valid_mask):
        # Normalized normals
        n_hat = np.zeros_like(cross_products)
        n_hat[valid_mask] = cross_products[valid_mask] / norms[valid_mask][:, None]
        
        # Gradients for each vertex of each triangle
        # Vertex 0: gradient = 0.5 * n_hat × (v2 - v1)
        # Vertex 1: gradient = 0.5 * n_hat × (v0 - v2)  
        # Vertex 2: gradient = 0.5 * n_hat × (v1 - v0)
        
        valid_indices = np.where(valid_mask)[0]
        
        for i in valid_indices:
            factor = 0.5 * default_surface_tension
            n = n_hat[i]
            
            # Gradients using the correct area gradient formula
            grad_v0 = factor * np.cross(n, v2[i] - v1[i])
            grad_v1 = factor * np.cross(n, v0[i] - v2[i])
            grad_v2 = factor * np.cross(n, v1[i] - v0[i])
            
            # Accumulate gradients
            v0_idx, v1_idx, v2_idx = facet_vertices[i]
            grad_accumulator[v0_idx] += grad_v0
            grad_accumulator[v1_idx] += grad_v1
            grad_accumulator[v2_idx] += grad_v2
    
    # Convert to dictionary format
    grad_dict = {}
    for i in range(len(grad_accumulator)):
        if i in mesh.vertices and np.any(grad_accumulator[i] != 0):
            grad_dict[i] = grad_accumulator[i]
            
    return total_energy, grad_dict

# Maintain backward compatibility
def compute_energy_and_gradient(mesh, global_params, param_resolver, *, compute_gradient: bool = True):
    """Main entry point that chooses the best optimization based on mesh type."""
    
    # Check if mesh supports vectorized computation
    if (hasattr(mesh, 'get_facet_vertices_array') and 
        mesh.get_facet_vertices_array() is not None and
        all(len(f.edge_indices) == 3 for f in mesh.facets.values())):
        
        # Use fully vectorized computation for triangular meshes
        return compute_energy_and_gradient_vectorized_triangular(
            mesh, global_params, param_resolver, compute_gradient=compute_gradient)
    else:
        # Use standard optimized computation
        return compute_energy_and_gradient_optimized(
            mesh, global_params, param_resolver, compute_gradient=compute_gradient)

def invalidate_cache():
    """Invalidate the surface energy cache."""
    _surface_cache.clear()

def calculate_surface_energy(mesh, global_params):
    """Legacy function for backward compatibility."""
    E, _ = compute_energy_and_gradient(mesh, global_params, None, compute_gradient=False)
    return E