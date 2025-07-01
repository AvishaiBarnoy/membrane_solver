# optimized_entities.py
"""
Optimized version of geometry entities with performance improvements.

Key optimizations:
1. Vectorized calculations using NumPy
2. Caching of expensive computations
3. Memory-efficient data structures
4. Reduced redundant operations
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from geometry.entities import Vertex, Edge, Facet, Body, Mesh

class OptimizedFacetMixin:
    """Performance optimizations for Facet class."""
    
    def __init__(self):
        self._cached_area = None
        self._cached_normal = None
        self._cached_vertex_loop = None
        self._dirty = True
        
    def invalidate_cache(self):
        """Mark cached values as invalid."""
        self._dirty = True
        self._cached_area = None
        self._cached_normal = None
        self._cached_vertex_loop = None
        
    def _get_vertex_loop_cached(self, mesh) -> List[int]:
        """Get vertex loop with caching."""
        if self._cached_vertex_loop is None or self._dirty:
            v_ids = []
            for signed_ei in self.edge_indices:
                edge = mesh.get_edge(signed_ei)
                tail = edge.tail_index
                if not v_ids or v_ids[-1] != tail:
                    v_ids.append(tail)
            self._cached_vertex_loop = v_ids
        return self._cached_vertex_loop
        
    def compute_area_optimized(self, mesh: "Mesh") -> float:
        """Optimized area computation with caching and vectorization."""
        if self._cached_area is not None and not self._dirty:
            return self._cached_area
            
        vertex_ids = self._get_vertex_loop_cached(mesh)
        if len(vertex_ids) < 3:
            self._cached_area = 0.0
            return self._cached_area
            
        # Get all vertex positions at once
        positions = np.array([mesh.vertices[i].position for i in vertex_ids])
        
        # Vectorized fan triangulation from first vertex
        v0 = positions[0]
        va = positions[1:-1] - v0  # vectors from v0 to vertices 1...n-2
        vb = positions[2:] - v0    # vectors from v0 to vertices 2...n-1
        
        # Cross products for all triangles at once
        cross_products = np.cross(va, vb)
        triangle_areas = 0.5 * np.linalg.norm(cross_products, axis=1)
        
        self._cached_area = np.sum(triangle_areas)
        self._dirty = False
        return self._cached_area
        
    def compute_area_gradient_optimized(self, mesh: "Mesh") -> Dict[int, np.ndarray]:
        """Optimized area gradient computation with vectorization."""
        vertex_ids = self._get_vertex_loop_cached(mesh)
        if len(vertex_ids) < 3:
            return {i: np.zeros(3) for i in vertex_ids}
            
        positions = np.array([mesh.vertices[i].position for i in vertex_ids])
        grad = {i: np.zeros(3) for i in vertex_ids}
        
        v0 = positions[0]
        va = positions[1:-1] - v0
        vb = positions[2:] - v0
        
        # Vectorized normal calculations
        normals = np.cross(va, vb)
        areas = np.linalg.norm(normals, axis=1)
        
        # Avoid division by zero
        valid_mask = areas >= 1e-12
        if not np.any(valid_mask):
            return grad
            
        # Normalized normals for valid triangles
        n_hat = np.zeros_like(normals)
        n_hat[valid_mask] = normals[valid_mask] / areas[valid_mask][:, None]
        
        # Vectorized gradient computation
        # Gradient for v0
        grad[vertex_ids[0]] = 0.5 * np.sum(np.cross(va[valid_mask], n_hat[valid_mask]), axis=0)
        
        # Gradients for other vertices
        for i, (a_idx, b_idx) in enumerate(zip(vertex_ids[1:-1], vertex_ids[2:])):
            if valid_mask[i]:
                grad[a_idx] += 0.5 * np.cross(vb[i] - va[i], n_hat[i])
                grad[b_idx] += 0.5 * np.cross(-vb[i], n_hat[i])
                
        return grad
        
    def normal_optimized(self, mesh: "Mesh") -> np.ndarray:
        """Optimized normal computation with caching."""
        if self._cached_normal is not None and not self._dirty:
            return self._cached_normal
            
        vertex_ids = self._get_vertex_loop_cached(mesh)
        if len(vertex_ids) < 3:
            raise ValueError("Cannot compute normal with fewer than 3 vertices.")
            
        positions = np.array([mesh.vertices[i].position for i in vertex_ids[:3]])
        
        u = positions[1] - positions[0]
        v = positions[2] - positions[0]
        normal = np.cross(u, v)
        
        norm = np.linalg.norm(normal)
        if norm == 0:
            raise ValueError("Degenerate facet with zero normal.")
            
        self._cached_normal = normal / norm
        return self._cached_normal


class OptimizedBodyMixin:
    """Performance optimizations for Body class."""
    
    def __init__(self):
        self._cached_volume = None
        self._cached_surface_area = None
        self._dirty = True
        
    def invalidate_cache(self):
        """Mark cached values as invalid."""
        self._dirty = True
        self._cached_volume = None
        self._cached_surface_area = None
        
    def compute_volume_optimized(self, mesh: "Mesh") -> float:
        """Optimized volume computation with vectorization."""
        if self._cached_volume is not None and not self._dirty:
            return self._cached_volume
            
        total_volume = 0.0
        
        for facet_idx in self.facet_indices:
            facet = mesh.facets[facet_idx]
            
            # Get vertex loop
            v_ids = []
            for signed_ei in facet.edge_indices:
                edge = mesh.edges[abs(signed_ei)]
                tail = edge.tail_index if signed_ei > 0 else edge.head_index
                if not v_ids or v_ids[-1] != tail:
                    v_ids.append(tail)
                    
            if len(v_ids) < 3:
                continue
                
            # Vectorized volume calculation
            positions = np.array([mesh.vertices[i].position for i in v_ids])
            v0 = positions[0]
            v1 = positions[1:-1]
            v2 = positions[2:]
            
            # Tetrahedron volumes: V = (1/6) * dot(v0, cross(v1, v2))
            cross_products = np.cross(v1, v2)
            volumes = np.dot(cross_products, v0) / 6.0
            total_volume += np.sum(volumes)
            
        self._cached_volume = total_volume
        self._dirty = False
        return self._cached_volume
        
    def compute_volume_gradient_optimized(self, mesh: "Mesh") -> Dict[int, np.ndarray]:
        """Optimized volume gradient computation."""
        # Collect all vertex indices in this body
        vertex_indices = set()
        for facet_idx in self.facet_indices:
            facet = mesh.facets[facet_idx]
            for signed_ei in facet.edge_indices:
                edge = mesh.edges[abs(signed_ei)]
                tail = edge.tail_index if signed_ei > 0 else edge.head_index
                vertex_indices.add(tail)
                
        grad = {i: np.zeros(3) for i in vertex_indices}
        
        for facet_idx in self.facet_indices:
            facet = mesh.facets[facet_idx]
            
            # Get vertex loop
            v_ids = []
            for signed_ei in facet.edge_indices:
                edge = mesh.edges[abs(signed_ei)]
                tail = edge.tail_index if signed_ei > 0 else edge.head_index
                if not v_ids or v_ids[-1] != tail:
                    v_ids.append(tail)
                    
            if len(v_ids) < 3:
                continue
                
            positions = np.array([mesh.vertices[i].position for i in v_ids])
            v0 = positions[0]
            va = positions[1:-1]
            vb = positions[2:]
            
            # Vectorized gradient computation
            cross_va_vb = np.cross(va, vb)
            grad[v_ids[0]] += np.sum(cross_va_vb, axis=0) / 6
            
            cross_vb_v0 = np.cross(vb, v0)
            cross_v0_va = np.cross(v0, va)
            
            for i, (a, b) in enumerate(zip(v_ids[1:-1], v_ids[2:])):
                grad[a] += cross_vb_v0[i] / 6
                grad[b] += cross_v0_va[i] / 6
                
        return grad


class OptimizedMesh(Mesh):
    """Optimized mesh with better data structures and caching."""
    
    def __init__(self):
        super().__init__()
        self._vertex_position_array = None
        self._facet_vertex_array = None
        self._connectivity_dirty = True
        
    def build_optimized_arrays(self):
        """Build optimized NumPy arrays for fast access."""
        if not self._connectivity_dirty:
            return
            
        # Vertex positions as contiguous array
        max_vertex_id = max(self.vertices.keys()) if self.vertices else 0
        self._vertex_position_array = np.zeros((max_vertex_id + 1, 3))
        
        for vid, vertex in self.vertices.items():
            self._vertex_position_array[vid] = vertex.position
            
        # For triangular meshes, store facet connectivity
        if all(len(f.edge_indices) == 3 for f in self.facets.values()):
            self._facet_vertex_array = np.zeros((len(self.facets), 3), dtype=int)
            
            for i, facet in enumerate(self.facets.values()):
                # Get the three vertices of the triangle
                v_ids = []
                for signed_ei in facet.edge_indices:
                    edge = self.get_edge(signed_ei)
                    tail = edge.tail_index
                    if not v_ids or v_ids[-1] != tail:
                        v_ids.append(tail)
                        
                if len(v_ids) >= 3:
                    self._facet_vertex_array[i] = v_ids[:3]
                    
        self._connectivity_dirty = False
        
    def get_vertex_positions_array(self) -> np.ndarray:
        """Get vertex positions as a contiguous NumPy array."""
        if self._connectivity_dirty:
            self.build_optimized_arrays()
        return self._vertex_position_array
        
    def get_facet_vertices_array(self) -> np.ndarray:
        """Get facet vertex connectivity as array (for triangular meshes)."""
        if self._connectivity_dirty:
            self.build_optimized_arrays()
        return self._facet_vertex_array
        
    def compute_total_surface_area_optimized(self) -> float:
        """Optimized total surface area computation."""
        if self._facet_vertex_array is not None and not self._connectivity_dirty:
            # Vectorized computation for triangular meshes
            positions = self.get_vertex_positions_array()
            facet_vertices = self.get_facet_vertices_array()
            
            v0 = positions[facet_vertices[:, 0]]
            v1 = positions[facet_vertices[:, 1]]
            v2 = positions[facet_vertices[:, 2]]
            
            # Triangle areas using cross product
            edge1 = v1 - v0
            edge2 = v2 - v0
            cross_products = np.cross(edge1, edge2)
            areas = 0.5 * np.linalg.norm(cross_products, axis=1)
            
            return np.sum(areas)
        else:
            # Fall back to standard computation
            return sum(facet.compute_area(self) for facet in self.facets.values())
            
    def invalidate_geometry_cache(self):
        """Invalidate all cached geometric values when mesh changes."""
        self._connectivity_dirty = True
        
        # Invalidate facet caches
        for facet in self.facets.values():
            if hasattr(facet, 'invalidate_cache'):
                facet.invalidate_cache()
                
        # Invalidate body caches  
        for body in self.bodies.values():
            if hasattr(body, 'invalidate_cache'):
                body.invalidate_cache()


# Monkey patch optimizations onto existing classes
def patch_optimizations():
    """Apply optimizations to existing geometry classes."""
    
    # Add optimized methods to Facet
    Facet.compute_area_optimized = OptimizedFacetMixin.compute_area_optimized
    Facet.compute_area_gradient_optimized = OptimizedFacetMixin.compute_area_gradient_optimized
    Facet.normal_optimized = OptimizedFacetMixin.normal_optimized
    Facet._get_vertex_loop_cached = OptimizedFacetMixin._get_vertex_loop_cached
    Facet.invalidate_cache = OptimizedFacetMixin.invalidate_cache
    
    # Add optimized methods to Body
    Body.compute_volume_optimized = OptimizedBodyMixin.compute_volume_optimized
    Body.compute_volume_gradient_optimized = OptimizedBodyMixin.compute_volume_gradient_optimized
    Body.invalidate_cache = OptimizedBodyMixin.invalidate_cache
    
    # Add optimized methods to Mesh
    Mesh.build_optimized_arrays = OptimizedMesh.build_optimized_arrays
    Mesh.get_vertex_positions_array = OptimizedMesh.get_vertex_positions_array
    Mesh.get_facet_vertices_array = OptimizedMesh.get_facet_vertices_array
    Mesh.compute_total_surface_area_optimized = OptimizedMesh.compute_total_surface_area_optimized
    Mesh.invalidate_geometry_cache = OptimizedMesh.invalidate_geometry_cache
    
    # Initialize cache fields
    def init_facet_cache(self):
        if not hasattr(self, '_cached_area'):
            self._cached_area = None
            self._cached_normal = None
            self._cached_vertex_loop = None
            self._dirty = True
            
    def init_body_cache(self):
        if not hasattr(self, '_cached_volume'):
            self._cached_volume = None
            self._cached_surface_area = None
            self._dirty = True
            
    def init_mesh_cache(self):
        if not hasattr(self, '_vertex_position_array'):
            self._vertex_position_array = None
            self._facet_vertex_array = None
            self._connectivity_dirty = True
    
    # Patch __init__ methods to add cache initialization
    original_facet_init = Facet.__init__
    original_body_init = Body.__init__
    original_mesh_init = Mesh.__init__
    
    def patched_facet_init(self, *args, **kwargs):
        original_facet_init(self, *args, **kwargs)
        init_facet_cache(self)
        
    def patched_body_init(self, *args, **kwargs):
        original_body_init(self, *args, **kwargs)
        init_body_cache(self)
        
    def patched_mesh_init(self, *args, **kwargs):
        original_mesh_init(self, *args, **kwargs)
        init_mesh_cache(self)
    
    Facet.__init__ = patched_facet_init
    Body.__init__ = patched_body_init
    Mesh.__init__ = patched_mesh_init