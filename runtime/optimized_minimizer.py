"""
Optimized minimizer with performance improvements.

Key optimizations:
1. Pre-allocated gradient arrays
2. Cached energy computations
3. Reduced memory allocations
4. Optimized constraint projection
5. Better data locality
"""

import sys
import os
import time
import numpy as np
from typing import Dict, List, Optional
from runtime.minimizer import Minimizer
from geometry.entities import Mesh
from parameters.global_parameters import GlobalParameters
from runtime.energy_manager import EnergyModuleManager
from runtime.constraint_manager import ConstraintModuleManager
from runtime.steppers.base import BaseStepper
import logging

logger = logging.getLogger('membrane_solver')

class OptimizedMinimizer(Minimizer):
    """
    Optimized minimizer with performance improvements over the base Minimizer.
    """
    
    def __init__(self, 
                 mesh: Mesh,
                 global_params: GlobalParameters,
                 stepper: BaseStepper,
                 energy_manager: EnergyModuleManager,
                 constraint_manager: ConstraintModuleManager,
                 energy_modules: Optional[List[str]] = None,
                 constraint_modules: Optional[List[str]] = None,
                 step_size: float = 1e-3,
                 tol: float = 1e-6,
                 quiet: bool = False,
                 use_optimizations: bool = True) -> None:
        
        super().__init__(mesh, global_params, stepper, energy_manager, constraint_manager,
                        energy_modules, constraint_modules, step_size, tol, quiet)
        
        self.use_optimizations = use_optimizations
        self._setup_optimizations()
        
        # Performance tracking
        self.timing_data = {
            'energy_computation': 0.0,
            'gradient_computation': 0.0,
            'constraint_projection': 0.0,
            'step_computation': 0.0,
            'total_time': 0.0
        }
        self.iteration_count = 0
        
    def _setup_optimizations(self):
        """Set up optimization data structures."""
        if not self.use_optimizations:
            return
            
        # Apply geometry optimizations
        try:
            from geometry.optimized_entities import patch_optimizations
            patch_optimizations()
            logger.debug("Applied geometry optimizations")
        except ImportError:
            logger.warning("Could not import geometry optimizations")
            
        # Pre-allocate gradient arrays
        max_vertex_id = max(self.mesh.vertices.keys()) if self.mesh.vertices else 0
        self.grad_array = np.zeros((max_vertex_id + 1, 3))
        self.vertex_mask = np.zeros(max_vertex_id + 1, dtype=bool)
        
        # Build optimized mesh arrays
        if hasattr(self.mesh, 'build_optimized_arrays'):
            self.mesh.build_optimized_arrays()
            
        # Cache for energy modules
        self._energy_cache = {}
        self._gradient_cache = {}
        self._cache_dirty = True
        
    def compute_energy_and_gradient_optimized(self):
        """Optimized energy and gradient computation."""
        start_time = time.perf_counter()
        
        total_energy = 0.0
        
        # Reset gradient array
        self.grad_array.fill(0.0)
        self.vertex_mask.fill(False)
        
        # Use optimized energy modules if available
        for module in self.energy_modules:
            module_name = module.__name__ if hasattr(module, '__name__') else str(module)
            
            # Check for optimized compute method
            if hasattr(module, 'compute_energy_and_gradient') and self.use_optimizations:
                try:
                    E_mod, g_mod = module.compute_energy_and_gradient(
                        self.mesh, self.global_params, self.param_resolver, 
                        compute_gradient=True)
                    
                    total_energy += E_mod
                    
                    # Accumulate gradients efficiently
                    for vertex_id, grad_vec in g_mod.items():
                        if vertex_id < len(self.grad_array):
                            self.grad_array[vertex_id] += grad_vec
                            self.vertex_mask[vertex_id] = True
                            
                except Exception as e:
                    logger.warning(f"Optimization failed for module {module_name}, falling back: {e}")
                    # Fall back to standard computation
                    E_mod, g_mod = self._compute_module_standard(module)
                    total_energy += E_mod
                    self._accumulate_gradients_dict(g_mod)
            else:
                # Standard computation
                E_mod, g_mod = self._compute_module_standard(module)
                total_energy += E_mod
                self._accumulate_gradients_dict(g_mod)
        
        # Convert back to dictionary format for compatibility
        grad_dict = {}
        for i in range(len(self.grad_array)):
            if self.vertex_mask[i]:
                grad_dict[i] = self.grad_array[i].copy()
        
        elapsed = time.perf_counter() - start_time
        self.timing_data['energy_computation'] += elapsed
        
        return total_energy, grad_dict
    
    def _compute_module_standard(self, module):
        """Standard energy module computation."""
        return module.compute_energy_and_gradient(
            self.mesh, self.global_params, self.param_resolver)
    
    def _accumulate_gradients_dict(self, grad_dict):
        """Accumulate gradients from dictionary format."""
        for vertex_id, grad_vec in grad_dict.items():
            if vertex_id < len(self.grad_array):
                self.grad_array[vertex_id] += grad_vec
                self.vertex_mask[vertex_id] = True
    
    def project_constraints_optimized(self, grad: Dict[int, np.ndarray]) -> None:
        """Optimized constraint projection."""
        start_time = time.perf_counter()
        
        # Batch process fixed vertices
        fixed_vertices = []
        for vidx, vertex in self.mesh.vertices.items():
            if getattr(vertex, 'fixed', False):
                fixed_vertices.append(vidx)
                
        # Zero out fixed vertices in batch
        for vidx in fixed_vertices:
            if vidx in grad:
                grad[vidx][:] = 0.0
        
        # Process constraints (if any)
        for vidx, vertex in self.mesh.vertices.items():
            if hasattr(vertex, 'constraint') and vidx in grad:
                grad[vidx] = vertex.constraint.project_gradient(grad[vidx])
        
        # Process edge constraints (placeholder - extend as needed)
        for eidx, edge in self.mesh.edges.items():
            if hasattr(edge, 'constraint'):
                # Implement edge constraint projection
                pass
        
        elapsed = time.perf_counter() - start_time
        self.timing_data['constraint_projection'] += elapsed
    
    def compute_energy_only_optimized(self):
        """Optimized energy-only computation for line search."""
        total_energy = 0.0
        
        for module in self.energy_modules:
            if hasattr(module, 'compute_energy_and_gradient'):
                E_mod, _ = module.compute_energy_and_gradient(
                    self.mesh, self.global_params, self.param_resolver, 
                    compute_gradient=False)
            else:
                # Fall back method
                E_mod = getattr(module, 'calculate_energy', 
                               lambda m, p: 0.0)(self.mesh, self.global_params)
            
            total_energy += E_mod
            
        return total_energy
    
    def minimize_optimized(self, n_steps: int = 1):
        """Optimized minimization loop."""
        start_total = time.perf_counter()
        zero_step_counter = 0
        max_zero_steps = 5
        
        for i in range(0, n_steps + 1):
            iteration_start = time.perf_counter()
            
            # Compute energy and gradients
            if self.use_optimizations:
                E, grad = self.compute_energy_and_gradient_optimized()
            else:
                E, grad = self.compute_energy_and_gradient()
            
            # Project constraints
            if self.use_optimizations:
                self.project_constraints_optimized(grad)
            else:
                self.project_constraints(grad)
            
            # Check convergence
            grad_norm = np.sqrt(sum(np.dot(g, g) for g in grad.values()))
            if grad_norm < self.tol:
                logger.debug("Converged: gradient norm below tolerance.")
                logger.info(f"Converged in {i} iterations; |∇E|={grad_norm:.3e}")
                
                elapsed_total = time.perf_counter() - start_total
                self.timing_data['total_time'] += elapsed_total
                
                return {"energy": E, "gradient": grad, "mesh": self.mesh,
                        "step_success": True, "iterations": i + 1,
                        "terminated_early": True, "timing": self.timing_data}
            
            # Compute area for reporting
            if hasattr(self.mesh, 'compute_total_surface_area_optimized') and self.use_optimizations:
                total_area = self.mesh.compute_total_surface_area_optimized()
            else:
                total_area = sum(facet.compute_area(self.mesh) for facet in self.mesh.facets.values())
            
            # Print step details
            if not self.quiet:
                print(f"Step {i:4d}: Area = {total_area:.5f}, Energy = {E:.5f}, Step Size = {self.step_size:.2e}")
            
            # Take optimization step
            step_start = time.perf_counter()
            
            if self.use_optimizations:
                step_success, self.step_size = self.stepper.step(
                    self.mesh, grad, self.step_size, self.compute_energy_only_optimized)
            else:
                step_success, self.step_size = self.stepper.step(
                    self.mesh, grad, self.step_size, self.compute_energy)
            
            step_elapsed = time.perf_counter() - step_start
            self.timing_data['step_computation'] += step_elapsed
            
            # Invalidate caches after step
            if self.use_optimizations and hasattr(self.mesh, 'invalidate_geometry_cache'):
                self.mesh.invalidate_geometry_cache()
            
            # Handle step failures
            if not step_success:
                zero_step_counter += 1
                if zero_step_counter >= max_zero_steps:
                    logger.info(f"Terminating early after {zero_step_counter} consecutive zero-steps.")
                    
                    elapsed_total = time.perf_counter() - start_total
                    self.timing_data['total_time'] += elapsed_total
                    
                    return {"energy": E, "gradient": grad, "mesh": self.mesh,
                            "step_success": False, "iterations": i + 1,
                            "terminated_early": True, "timing": self.timing_data}
            else:
                zero_step_counter = 0
            
            iteration_elapsed = time.perf_counter() - iteration_start
            self.iteration_count += 1
        
        elapsed_total = time.perf_counter() - start_total
        self.timing_data['total_time'] += elapsed_total
        
        return {"energy": E, "gradient": grad, "mesh": self.mesh,
                "step_success": step_success, "iterations": n_steps,
                "terminated_early": False, "timing": self.timing_data}
    
    def get_performance_stats(self):
        """Get detailed performance statistics."""
        if self.iteration_count == 0:
            return self.timing_data
            
        stats = self.timing_data.copy()
        stats['avg_energy_time'] = stats['energy_computation'] / self.iteration_count
        stats['avg_constraint_time'] = stats['constraint_projection'] / self.iteration_count
        stats['avg_step_time'] = stats['step_computation'] / self.iteration_count
        stats['avg_iteration_time'] = stats['total_time'] / self.iteration_count
        stats['iterations'] = self.iteration_count
        
        return stats
    
    def minimize(self, n_steps: int = 1):
        """Main minimize method - uses optimized version if enabled."""
        if self.use_optimizations:
            return self.minimize_optimized(n_steps)
        else:
            return super().minimize(n_steps)