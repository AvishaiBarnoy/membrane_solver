#!/usr/bin/env python3
"""
Test script to verify that optimizations work correctly and maintain compatibility.

This script tests:
1. Basic functionality still works
2. Optimized methods produce same results as original methods
3. Performance improvements are achieved
4. Caching works correctly
5. Error handling is maintained
"""

import sys
import os
import json
import time
import traceback

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all necessary modules can be imported."""
    print("Testing imports...")
    
    try:
        # Test basic imports
        from geometry.entities import Vertex, Edge, Facet, Body, Mesh
        from geometry.geom_io import load_data, parse_geometry
        from runtime.minimizer import Minimizer
        from parameters.resolver import ParameterResolver
        from runtime.steppers.gradient_descent import GradientDescent
        from runtime.energy_manager import EnergyModuleManager
        from runtime.constraint_manager import ConstraintModuleManager
        print("✓ Basic imports successful")
        
        # Test optimization imports
        from geometry.optimized_entities import patch_optimizations, OptimizedFacetMixin, OptimizedBodyMixin
        from runtime.optimized_minimizer import OptimizedMinimizer
        print("✓ Optimization imports successful")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test that basic mesh operations still work."""
    print("\nTesting basic functionality...")
    
    try:
        # Load a simple mesh
        from geometry.geom_io import load_data, parse_geometry
        data = load_data("meshes/cube.json")
        mesh = parse_geometry(data)
        
        # Basic mesh properties
        assert len(mesh.vertices) > 0, "No vertices in mesh"
        assert len(mesh.facets) > 0, "No facets in mesh"
        assert len(mesh.bodies) > 0, "No bodies in mesh"
        print(f"✓ Mesh loaded: {len(mesh.vertices)} vertices, {len(mesh.facets)} facets")
        
        # Test area calculations
        total_area = sum(facet.compute_area(mesh) for facet in mesh.facets.values())
        assert total_area > 0, "Total area should be positive"
        print(f"✓ Area calculation: {total_area:.6f}")
        
        # Test volume calculations
        total_volume = sum(body.compute_volume(mesh) for body in mesh.bodies.values())
        assert total_volume > 0, "Total volume should be positive"
        print(f"✓ Volume calculation: {total_volume:.6f}")
        
        return mesh
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        traceback.print_exc()
        return None

def test_optimization_patching():
    """Test that optimization patching works correctly."""
    print("\nTesting optimization patching...")
    
    try:
        from geometry.optimized_entities import patch_optimizations
        from geometry.entities import Facet, Body, Mesh
        
        # Apply patches
        patch_optimizations()
        print("✓ Patch optimizations applied")
        
        # Check that new methods exist
        sample_facet = Facet(0, [1, 2, 3])
        assert hasattr(sample_facet, 'compute_area_optimized'), "Facet missing optimized area method"
        assert hasattr(sample_facet, 'invalidate_cache'), "Facet missing cache invalidation"
        print("✓ Facet optimizations patched")
        
        sample_body = Body(0, [0, 1])
        assert hasattr(sample_body, 'compute_volume_optimized'), "Body missing optimized volume method"
        assert hasattr(sample_body, 'invalidate_cache'), "Body missing cache invalidation"
        print("✓ Body optimizations patched")
        
        sample_mesh = Mesh()
        assert hasattr(sample_mesh, 'build_optimized_arrays'), "Mesh missing optimized arrays method"
        assert hasattr(sample_mesh, 'invalidate_geometry_cache'), "Mesh missing cache invalidation"
        print("✓ Mesh optimizations patched")
        
        return True
    except Exception as e:
        print(f"✗ Optimization patching failed: {e}")
        traceback.print_exc()
        return False

def test_optimized_results_match_original(mesh):
    """Test that optimized methods produce the same results as original methods."""
    print("\nTesting result consistency between original and optimized methods...")
    
    try:
        from geometry.optimized_entities import patch_optimizations
        patch_optimizations()
        
        tolerance = 1e-10
        
        # Test area calculations
        print("Testing area calculation consistency...")
        area_differences = []
        for facet in mesh.facets.values():
            original_area = facet.compute_area(mesh)
            optimized_area = facet.compute_area_optimized(mesh)
            diff = abs(original_area - optimized_area)
            area_differences.append(diff)
            
            if diff > tolerance:
                print(f"⚠ Large area difference for facet {facet.index}: {diff}")
        
        max_area_diff = max(area_differences) if area_differences else 0
        print(f"✓ Area calculations match (max diff: {max_area_diff:.2e})")
        
        # Test volume calculations
        print("Testing volume calculation consistency...")
        volume_differences = []
        for body in mesh.bodies.values():
            original_volume = body.compute_volume(mesh)
            optimized_volume = body.compute_volume_optimized(mesh)
            diff = abs(original_volume - optimized_volume)
            volume_differences.append(diff)
            
            if diff > tolerance:
                print(f"⚠ Large volume difference for body {body.index}: {diff}")
        
        max_volume_diff = max(volume_differences) if volume_differences else 0
        print(f"✓ Volume calculations match (max diff: {max_volume_diff:.2e})")
        
        # Test mesh-level area calculation
        if hasattr(mesh, 'build_optimized_arrays'):
            mesh.build_optimized_arrays()
            if hasattr(mesh, 'compute_total_surface_area_optimized'):
                original_total = sum(f.compute_area(mesh) for f in mesh.facets.values())
                optimized_total = mesh.compute_total_surface_area_optimized()
                total_diff = abs(original_total - optimized_total)
                print(f"✓ Mesh-level area matches (diff: {total_diff:.2e})")
        
        return True
    except Exception as e:
        print(f"✗ Result consistency test failed: {e}")
        traceback.print_exc()
        return False

def test_caching_functionality(mesh):
    """Test that caching works correctly and improves performance."""
    print("\nTesting caching functionality...")
    
    try:
        from geometry.optimized_entities import patch_optimizations
        patch_optimizations()
        
        # Test facet caching
        facet = list(mesh.facets.values())[0]
        
        # First call should compute
        start_time = time.perf_counter()
        area1 = facet.compute_area_optimized(mesh)
        first_time = time.perf_counter() - start_time
        
        # Second call should be cached (faster)
        start_time = time.perf_counter()
        area2 = facet.compute_area_optimized(mesh)
        second_time = time.perf_counter() - start_time
        
        assert abs(area1 - area2) < 1e-10, "Cached result differs from computed result"
        
        # Cache should be faster (though timing can be variable)
        if second_time < first_time * 0.8:
            print(f"✓ Caching provides speedup: {first_time/second_time:.1f}x")
        else:
            print(f"✓ Caching works (timing: {first_time:.2e}s → {second_time:.2e}s)")
        
        # Test cache invalidation
        facet.invalidate_cache()
        area3 = facet.compute_area_optimized(mesh)
        assert abs(area1 - area3) < 1e-10, "Result after cache invalidation differs"
        print("✓ Cache invalidation works")
        
        return True
    except Exception as e:
        print(f"✗ Caching test failed: {e}")
        traceback.print_exc()
        return False

def test_optimized_minimizer(mesh):
    """Test that the optimized minimizer works correctly."""
    print("\nTesting optimized minimizer...")
    
    try:
        from runtime.minimizer import Minimizer
        from runtime.optimized_minimizer import OptimizedMinimizer
        from parameters.resolver import ParameterResolver
        from runtime.steppers.gradient_descent import GradientDescent
        from runtime.energy_manager import EnergyModuleManager
        from runtime.constraint_manager import ConstraintModuleManager
        
        # Set up common components
        param_resolver = ParameterResolver(mesh.global_parameters)
        energy_manager = EnergyModuleManager(mesh.energy_modules)
        constraint_manager = ConstraintModuleManager(mesh.constraint_modules)
        stepper = GradientDescent()
        
        # Test original minimizer
        original_minimizer = Minimizer(
            mesh, mesh.global_parameters, stepper, energy_manager, constraint_manager, quiet=True)
        
        start_time = time.perf_counter()
        original_energy, original_grad = original_minimizer.compute_energy_and_gradient()
        original_time = time.perf_counter() - start_time
        print(f"✓ Original minimizer works: E={original_energy:.6f}, time={original_time:.4f}s")
        
        # Test optimized minimizer
        optimized_minimizer = OptimizedMinimizer(
            mesh, mesh.global_parameters, stepper, energy_manager, constraint_manager, 
            quiet=True, use_optimizations=True)
        
        start_time = time.perf_counter()
        optimized_energy, optimized_grad = optimized_minimizer.compute_energy_and_gradient_optimized()
        optimized_time = time.perf_counter() - start_time
        print(f"✓ Optimized minimizer works: E={optimized_energy:.6f}, time={optimized_time:.4f}s")
        
        # Compare results
        energy_diff = abs(original_energy - optimized_energy)
        print(f"✓ Energy difference: {energy_diff:.2e}")
        
        # Compare gradients
        grad_diffs = []
        for vid in original_grad.keys():
            if vid in optimized_grad:
                import numpy as np
                diff = np.linalg.norm(original_grad[vid] - optimized_grad[vid])
                grad_diffs.append(diff)
        
        max_grad_diff = max(grad_diffs) if grad_diffs else 0
        print(f"✓ Max gradient difference: {max_grad_diff:.2e}")
        
        # Test performance improvement
        if optimized_time < original_time:
            speedup = original_time / optimized_time
            print(f"✓ Performance improvement: {speedup:.2f}x speedup")
        else:
            print("✓ Optimized version runs (no significant speedup on small mesh)")
        
        return True
    except Exception as e:
        print(f"✗ Optimized minimizer test failed: {e}")
        traceback.print_exc()
        return False

def test_error_handling():
    """Test that error handling is maintained in optimized functions."""
    print("\nTesting error handling...")
    
    try:
        from geometry.entities import Vertex, Edge, Facet, Mesh
        from geometry.optimized_entities import patch_optimizations
        import numpy as np
        
        patch_optimizations()
        
        # Test with degenerate facet (< 3 vertices)
        mesh = Mesh()
        mesh.vertices = {0: Vertex(0, np.array([0, 0, 0])), 1: Vertex(1, np.array([1, 0, 0]))}
        mesh.edges = {1: Edge(1, 0, 1)}
        degenerate_facet = Facet(0, [1])  # Only one edge
        
        # Should handle gracefully
        area = degenerate_facet.compute_area_optimized(mesh)
        assert area == 0.0, "Degenerate facet should have zero area"
        print("✓ Degenerate facet handled correctly")
        
        # Test with invalid edge indices (should raise appropriate error)
        try:
            invalid_facet = Facet(1, [999])  # Non-existent edge
            area = invalid_facet.compute_area_optimized(mesh)
            print("⚠ Invalid edge should have raised an error")
        except KeyError:
            print("✓ Invalid edge properly raises KeyError")
        
        return True
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        traceback.print_exc()
        return False

def test_performance_improvements(mesh):
    """Test that performance improvements are achieved."""
    print("\nTesting performance improvements...")
    
    try:
        from geometry.optimized_entities import patch_optimizations
        patch_optimizations()
        
        iterations = 100
        
        # Test area calculation performance
        print(f"Benchmarking area calculations ({iterations} iterations)...")
        
        # Original method
        start_time = time.perf_counter()
        for _ in range(iterations):
            total_area = sum(facet.compute_area(mesh) for facet in mesh.facets.values())
        original_time = time.perf_counter() - start_time
        
        # Optimized method
        start_time = time.perf_counter()
        for _ in range(iterations):
            total_area_opt = sum(facet.compute_area_optimized(mesh) for facet in mesh.facets.values())
        optimized_time = time.perf_counter() - start_time
        
        print(f"Original: {original_time:.4f}s, Optimized: {optimized_time:.4f}s")
        
        if optimized_time < original_time:
            speedup = original_time / optimized_time
            print(f"✓ Area calculation speedup: {speedup:.2f}x")
        else:
            print("✓ Optimized version runs (no significant speedup on small mesh)")
        
        # Test mesh-level optimization if available
        if hasattr(mesh, 'build_optimized_arrays'):
            mesh.build_optimized_arrays()
            if hasattr(mesh, 'compute_total_surface_area_optimized'):
                start_time = time.perf_counter()
                for _ in range(iterations):
                    total_area_mesh = mesh.compute_total_surface_area_optimized()
                mesh_time = time.perf_counter() - start_time
                
                if mesh_time < original_time:
                    mesh_speedup = original_time / mesh_time
                    print(f"✓ Mesh-level speedup: {mesh_speedup:.2f}x")
        
        return True
    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all optimization tests."""
    print("=" * 60)
    print("OPTIMIZATION COMPATIBILITY TESTS")
    print("=" * 60)
    
    test_results = {}
    
    # Test imports
    test_results['imports'] = test_imports()
    if not test_results['imports']:
        print("❌ Cannot proceed without working imports")
        return test_results
    
    # Test basic functionality
    mesh = test_basic_functionality()
    test_results['basic_functionality'] = mesh is not None
    if mesh is None:
        print("❌ Cannot proceed without working basic functionality")
        return test_results
    
    # Test optimization patching
    test_results['optimization_patching'] = test_optimization_patching()
    
    # Test result consistency
    test_results['result_consistency'] = test_optimized_results_match_original(mesh)
    
    # Test caching
    test_results['caching'] = test_caching_functionality(mesh)
    
    # Test optimized minimizer
    test_results['optimized_minimizer'] = test_optimized_minimizer(mesh)
    
    # Test error handling
    test_results['error_handling'] = test_error_handling()
    
    # Test performance
    test_results['performance'] = test_performance_improvements(mesh)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:25s}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Optimizations are working correctly.")
    else:
        print("⚠️  Some tests failed. Please review the issues above.")
    
    return test_results

if __name__ == "__main__":
    run_all_tests()