#!/usr/bin/env python3
"""
Benchmark script to compare performance before and after optimizations.

This script compares:
1. Original vs optimized area/volume calculations
2. Original vs optimized energy computations
3. Original vs optimized minimization loops
4. Memory usage patterns
"""

import time
import json
import sys
import os
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from geometry.geom_io import load_data, parse_geometry
from runtime.minimizer import Minimizer
from runtime.optimized_minimizer import OptimizedMinimizer
from parameters.resolver import ParameterResolver
from runtime.steppers.gradient_descent import GradientDescent
from runtime.energy_manager import EnergyModuleManager
from runtime.constraint_manager import ConstraintModuleManager
from runtime.refinement import refine_triangle_mesh
from logging_config import setup_logging

class PerformanceBenchmark:
    """Performance benchmarking suite for membrane solver optimizations."""
    
    def __init__(self, mesh_file="meshes/cube.json"):
        self.mesh_file = mesh_file
        self.logger = setup_logging('benchmark.log', quiet=True)
        self.results = {}
        
    def load_test_mesh(self):
        """Load and prepare the test mesh."""
        print(f"Loading test mesh: {self.mesh_file}")
        data = load_data(self.mesh_file)
        mesh = parse_geometry(data)
        
        print(f"Mesh info:")
        print(f"  Vertices: {len(mesh.vertices)}")
        print(f"  Edges: {len(mesh.edges)}")
        print(f"  Facets: {len(mesh.facets)}")
        print(f"  Bodies: {len(mesh.bodies)}")
        print(f"  Energy modules: {mesh.energy_modules}")
        
        return mesh
    
    def benchmark_area_calculations(self, mesh, iterations=1000):
        """Benchmark area calculation performance."""
        print(f"\nBenchmarking area calculations ({iterations} iterations)...")
        
        # Original method
        start_time = time.perf_counter()
        for _ in range(iterations):
            total_area = sum(facet.compute_area(mesh) for facet in mesh.facets.values())
        original_time = time.perf_counter() - start_time
        
        # Try optimized method
        optimized_time = None
        try:
            # Apply optimizations
            from geometry.optimized_entities import patch_optimizations
            patch_optimizations()
            
            # Build optimized arrays
            if hasattr(mesh, 'build_optimized_arrays'):
                mesh.build_optimized_arrays()
            
            # Test individual facet optimization
            start_time = time.perf_counter()
            for _ in range(iterations):
                total_area_opt = sum(facet.compute_area_optimized(mesh) if hasattr(facet, 'compute_area_optimized') 
                                   else facet.compute_area(mesh) for facet in mesh.facets.values())
            optimized_time = time.perf_counter() - start_time
            
            # Test mesh-level optimization
            mesh_optimized_time = None
            if hasattr(mesh, 'compute_total_surface_area_optimized'):
                start_time = time.perf_counter()
                for _ in range(iterations):
                    total_area_mesh = mesh.compute_total_surface_area_optimized()
                mesh_optimized_time = time.perf_counter() - start_time
                
        except Exception as e:
            print(f"Could not test optimized area calculations: {e}")
        
        results = {
            'original_time': original_time,
            'optimized_time': optimized_time,
            'mesh_optimized_time': mesh_optimized_time,
            'speedup': optimized_time / original_time if optimized_time else None,
            'mesh_speedup': mesh_optimized_time / original_time if mesh_optimized_time else None
        }
        
        print(f"  Original method: {original_time:.4f}s ({original_time/iterations*1000:.2f}ms per iteration)")
        if optimized_time:
            print(f"  Optimized method: {optimized_time:.4f}s ({optimized_time/iterations*1000:.2f}ms per iteration)")
            print(f"  Speedup: {original_time/optimized_time:.2f}x")
        if mesh_optimized_time:
            print(f"  Mesh-optimized: {mesh_optimized_time:.4f}s ({mesh_optimized_time/iterations*1000:.2f}ms per iteration)")
            print(f"  Mesh speedup: {original_time/mesh_optimized_time:.2f}x")
        
        return results
    
    def benchmark_volume_calculations(self, mesh, iterations=1000):
        """Benchmark volume calculation performance."""
        print(f"\nBenchmarking volume calculations ({iterations} iterations)...")
        
        # Original method
        start_time = time.perf_counter()
        for _ in range(iterations):
            total_volume = sum(body.compute_volume(mesh) for body in mesh.bodies.values())
        original_time = time.perf_counter() - start_time
        
        # Try optimized method
        optimized_time = None
        try:
            start_time = time.perf_counter()
            for _ in range(iterations):
                total_volume_opt = sum(body.compute_volume_optimized(mesh) if hasattr(body, 'compute_volume_optimized')
                                     else body.compute_volume(mesh) for body in mesh.bodies.values())
            optimized_time = time.perf_counter() - start_time
        except Exception as e:
            print(f"Could not test optimized volume calculations: {e}")
        
        results = {
            'original_time': original_time,
            'optimized_time': optimized_time,
            'speedup': optimized_time / original_time if optimized_time else None
        }
        
        print(f"  Original method: {original_time:.4f}s ({original_time/iterations*1000:.2f}ms per iteration)")
        if optimized_time:
            print(f"  Optimized method: {optimized_time:.4f}s ({optimized_time/iterations*1000:.2f}ms per iteration)")
            print(f"  Speedup: {original_time/optimized_time:.2f}x")
        
        return results
    
    def benchmark_energy_computation(self, mesh, iterations=50):
        """Benchmark energy and gradient computation."""
        print(f"\nBenchmarking energy computation ({iterations} iterations)...")
        
        # Set up original minimizer
        param_resolver = ParameterResolver(mesh.global_parameters)
        energy_manager = EnergyModuleManager(mesh.energy_modules)
        constraint_manager = ConstraintModuleManager(mesh.constraint_modules)
        stepper = GradientDescent()
        
        original_minimizer = Minimizer(
            mesh, mesh.global_parameters, stepper, energy_manager, constraint_manager, quiet=True)
        
        # Benchmark original energy computation
        start_time = time.perf_counter()
        for _ in range(iterations):
            E, grad = original_minimizer.compute_energy_and_gradient()
        original_time = time.perf_counter() - start_time
        
        # Try optimized minimizer
        optimized_time = None
        try:
            optimized_minimizer = OptimizedMinimizer(
                mesh, mesh.global_parameters, stepper, energy_manager, constraint_manager, 
                quiet=True, use_optimizations=True)
            
            start_time = time.perf_counter()
            for _ in range(iterations):
                E_opt, grad_opt = optimized_minimizer.compute_energy_and_gradient_optimized()
            optimized_time = time.perf_counter() - start_time
            
        except Exception as e:
            print(f"Could not test optimized energy computation: {e}")
        
        results = {
            'original_time': original_time,
            'optimized_time': optimized_time,
            'speedup': optimized_time / original_time if optimized_time else None
        }
        
        print(f"  Original method: {original_time:.4f}s ({original_time/iterations*1000:.2f}ms per iteration)")
        if optimized_time:
            print(f"  Optimized method: {optimized_time:.4f}s ({optimized_time/iterations*1000:.2f}ms per iteration)")
            print(f"  Speedup: {original_time/optimized_time:.2f}x")
        
        return results
    
    def benchmark_minimization_loop(self, mesh, steps=10):
        """Benchmark full minimization loop."""
        print(f"\nBenchmarking minimization loop ({steps} steps)...")
        
        # Set up minimizers
        param_resolver = ParameterResolver(mesh.global_parameters)
        energy_manager = EnergyModuleManager(mesh.energy_modules)
        constraint_manager = ConstraintModuleManager(mesh.constraint_modules)
        stepper = GradientDescent()
        
        # Original minimizer
        original_minimizer = Minimizer(
            mesh, mesh.global_parameters, stepper, energy_manager, constraint_manager, quiet=True)
        
        start_time = time.perf_counter()
        original_result = original_minimizer.minimize(n_steps=steps)
        original_time = time.perf_counter() - start_time
        
        # Try optimized minimizer
        optimized_time = None
        optimized_result = None
        try:
            # Reload mesh for fair comparison
            mesh_copy = parse_geometry(load_data(self.mesh_file))
            
            optimized_minimizer = OptimizedMinimizer(
                mesh_copy, mesh_copy.global_parameters, GradientDescent(), 
                EnergyModuleManager(mesh_copy.energy_modules),
                ConstraintModuleManager(mesh_copy.constraint_modules),
                quiet=True, use_optimizations=True)
            
            start_time = time.perf_counter()
            optimized_result = optimized_minimizer.minimize(n_steps=steps)
            optimized_time = time.perf_counter() - start_time
            
        except Exception as e:
            print(f"Could not test optimized minimization: {e}")
        
        results = {
            'original_time': original_time,
            'original_energy': original_result.get('energy', 0),
            'optimized_time': optimized_time,
            'optimized_energy': optimized_result.get('energy', 0) if optimized_result else None,
            'speedup': optimized_time / original_time if optimized_time else None
        }
        
        print(f"  Original method: {original_time:.4f}s, final energy: {original_result.get('energy', 0):.6f}")
        if optimized_time:
            print(f"  Optimized method: {optimized_time:.4f}s, final energy: {optimized_result.get('energy', 0):.6f}")
            print(f"  Speedup: {original_time/optimized_time:.2f}x")
        
        return results
    
    def benchmark_mesh_refinement(self, mesh):
        """Benchmark mesh refinement performance."""
        print(f"\nBenchmarking mesh refinement...")
        
        original_vertices = len(mesh.vertices)
        original_facets = len(mesh.facets)
        
        start_time = time.perf_counter()
        refined_mesh = refine_triangle_mesh(mesh)
        refinement_time = time.perf_counter() - start_time
        
        results = {
            'time': refinement_time,
            'original_vertices': original_vertices,
            'refined_vertices': len(refined_mesh.vertices),
            'original_facets': original_facets,
            'refined_facets': len(refined_mesh.facets),
            'vertex_ratio': len(refined_mesh.vertices) / original_vertices,
            'facet_ratio': len(refined_mesh.facets) / original_facets
        }
        
        print(f"  Refinement time: {refinement_time:.4f}s")
        print(f"  Vertices: {original_vertices} → {len(refined_mesh.vertices)} ({results['vertex_ratio']:.1f}x)")
        print(f"  Facets: {original_facets} → {len(refined_mesh.facets)} ({results['facet_ratio']:.1f}x)")
        
        return results, refined_mesh
    
    def run_full_benchmark(self):
        """Run complete benchmark suite."""
        print("=" * 60)
        print("MEMBRANE SOLVER PERFORMANCE BENCHMARK")
        print("=" * 60)
        
        # Load test mesh
        mesh = self.load_test_mesh()
        
        # Run benchmarks
        self.results['area_calculations'] = self.benchmark_area_calculations(mesh)
        self.results['volume_calculations'] = self.benchmark_volume_calculations(mesh)
        self.results['energy_computation'] = self.benchmark_energy_computation(mesh)
        self.results['minimization_loop'] = self.benchmark_minimization_loop(mesh)
        self.results['mesh_refinement'], refined_mesh = self.benchmark_mesh_refinement(mesh)
        
        # Test on refined mesh for scaling analysis
        if len(refined_mesh.vertices) > len(mesh.vertices) * 2:
            print(f"\nBenchmarking on refined mesh ({len(refined_mesh.vertices)} vertices)...")
            self.results['refined_area_calculations'] = self.benchmark_area_calculations(refined_mesh, iterations=200)
            self.results['refined_energy_computation'] = self.benchmark_energy_computation(refined_mesh, iterations=10)
        
        # Summary
        self.print_summary()
        
        return self.results
    
    def print_summary(self):
        """Print benchmark summary."""
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        
        for test_name, results in self.results.items():
            if isinstance(results, dict) and 'speedup' in results and results['speedup']:
                speedup = results['speedup']
                improvement = (1/speedup - 1) * 100 if speedup < 1 else (speedup - 1) * 100
                status = "SLOWER" if speedup < 1 else "FASTER"
                print(f"{test_name:25s}: {speedup:.2f}x ({improvement:+.1f}% {status})")
        
        print("\nKey optimizations verified:")
        if self.results.get('area_calculations', {}).get('speedup', 0) > 1:
            print("✓ Area calculations optimized")
        if self.results.get('volume_calculations', {}).get('speedup', 0) > 1:
            print("✓ Volume calculations optimized")
        if self.results.get('energy_computation', {}).get('speedup', 0) > 1:
            print("✓ Energy computation optimized")
        if self.results.get('minimization_loop', {}).get('speedup', 0) > 1:
            print("✓ Minimization loop optimized")
    
    def save_results(self, filename="benchmark_results.json"):
        """Save benchmark results to file."""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nBenchmark results saved to {filename}")

def main():
    """Main benchmark entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark membrane solver optimizations")
    parser.add_argument('--mesh', default='meshes/cube.json', help='Mesh file to benchmark')
    parser.add_argument('--output', default='benchmark_results.json', help='Output file for results')
    args = parser.parse_args()
    
    benchmark = PerformanceBenchmark(args.mesh)
    results = benchmark.run_full_benchmark()
    benchmark.save_results(args.output)

if __name__ == "__main__":
    main()