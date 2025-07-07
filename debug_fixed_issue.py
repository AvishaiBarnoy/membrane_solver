#!/usr/bin/env python3
"""
Diagnostic script to trace the fixed property issue
"""

import json
import sys
from geometry.geom_io import parse_geometry

def print_fixed_status(mesh, label):
    """Print the fixed status of all mesh elements"""
    print(f"\n=== {label} ===")
    
    print("Vertices:")
    for vid, vertex in mesh.vertices.items():
        fixed_status = getattr(vertex, 'fixed', False)
        if fixed_status or vertex.options:
            print(f"  V{vid}: fixed={fixed_status}, options={vertex.options}")
    
    print("Edges:")
    for eid, edge in mesh.edges.items():
        fixed_status = getattr(edge, 'fixed', False)
        if fixed_status or edge.options:
            print(f"  E{eid}: fixed={fixed_status}, options={edge.options}")
    
    print("Facets:")
    for fid, facet in mesh.facets.items():
        fixed_status = getattr(facet, 'fixed', False)
        if fixed_status or facet.options:
            print(f"  F{fid}: fixed={fixed_status}, options={facet.options}")

# Create minimal test geometry
minimal_geometry = {
    "vertices": [
        [0, 0, 0], 
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0]
    ],
    "edges": [
        [0, 1],
        [1, 2], 
        [2, 3],
        [3, 0]
    ],
    "faces": [
        [0, 1, 2, 3]  # Single square face - should get refined
    ],
    "bodies": {
        "faces": [[0]],
        "target_volume": [1.0]
    },
    "global_parameters": {"surface_tension": 1.0}
}

print("Testing with minimal geometry (no fixed properties)...")

# Parse the geometry
mesh = parse_geometry(minimal_geometry)

print_fixed_status(mesh, "After parsing and automatic refinement")

print("\nIf you see any 'fixed=True' above, that indicates where the problem is coming from.")