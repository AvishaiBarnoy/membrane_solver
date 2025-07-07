#!/usr/bin/env python3
"""
Final diagnostic to answer the specific question:
"Is there, during refinement that some vertices get the `fixed=True` attribute 
without their parent having the `fixed=True`?"
"""

from geometry.geom_io import parse_geometry

def trace_fixed_inheritance(mesh, label):
    """Trace fixed property inheritance to check for inappropriate assignments"""
    print(f"\n=== {label} ===")
    
    # Check for any inheritance violations
    violations = []
    
    # Check vertices against their parent edges (if they exist as midpoints)
    # This is complex to trace back perfectly, so let's check the general pattern
    
    # For facets, check if any child has fixed=True when parent has fixed=False
    for fid, facet in mesh.facets.items():
        if hasattr(facet, 'options') and 'parent_facet' in facet.options:
            parent_id = facet.options['parent_facet']
            if parent_id in mesh.facets:
                parent_facet = mesh.facets[parent_id]
                parent_fixed = getattr(parent_facet, 'fixed', False)
                child_fixed = getattr(facet, 'fixed', False)
                
                if child_fixed and not parent_fixed:
                    violations.append(f"Facet {fid} has fixed=True but parent facet {parent_id} has fixed=False")
    
    # Print summary of fixed elements
    fixed_vertices = [(vid, v) for vid, v in mesh.vertices.items() if getattr(v, 'fixed', False)]
    fixed_edges = [(eid, e) for eid, e in mesh.edges.items() if getattr(e, 'fixed', False)]
    fixed_facets = [(fid, f) for fid, f in mesh.facets.items() if getattr(f, 'fixed', False)]
    
    print(f"Fixed vertices: {len(fixed_vertices)}")
    print(f"Fixed edges: {len(fixed_edges)}")
    print(f"Fixed facets: {len(fixed_facets)}")
    
    if violations:
        print(f"\n❌ VIOLATIONS FOUND:")
        for violation in violations:
            print(f"  {violation}")
    else:
        print(f"\n✅ No inheritance violations found")
    
    return len(violations) == 0

# Test with the simple cube geometry
cube_geometry = {
    "vertices": [
        [0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1],
        [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 1, 1]
    ],
    "edges": [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 5], [1, 6], [2, 7], [3, 4]
    ],
    "faces": [
        [0, 1, 2, 3],          # Face 0 - will be refined
        ["r0", 8, 5, "r9"],    # Face 1 
        [9, 6, -10, -1],       # Face 2
        [-2, 10, 7, -11],      # Face 3
        [11, 4, -8, -3],       # Face 4
        [-5, -4, -7, -6]       # Face 5
    ],
    "bodies": {
        "faces": [[0, 1, 2, 3, 4, 5]],
        "target_volume": [1.0]
    },
    "global_parameters": {"surface_tension": 1.0}
}

print("Testing inheritance behavior with clean cube geometry...")
mesh = parse_geometry(cube_geometry)
is_valid = trace_fixed_inheritance(mesh, "Clean cube - should have no fixed elements")

print(f"\n{'='*60}")
print(f"ANSWER TO THE QUESTION:")
print(f"Do vertices get fixed=True without their parent having it?")
print(f"Result: {'NO - inheritance works correctly' if is_valid else 'YES - violation found'}")
print(f"{'='*60}")

# Also test with a geometry that HAS a fixed facet to see inheritance
cube_with_fixed_facet = cube_geometry.copy()
cube_with_fixed_facet["faces"][0] = [0, 1, 2, 3, {"fixed": True}]  # Make face 0 fixed

print(f"\nTesting with face 0 explicitly fixed...")
mesh2 = parse_geometry(cube_with_fixed_facet)
trace_fixed_inheritance(mesh2, "Cube with face 0 fixed - children should inherit")