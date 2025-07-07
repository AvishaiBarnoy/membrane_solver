#!/usr/bin/env python3
"""
Diagnostic script to check what happens when parsing cube.json
"""

import json
import sys
from geometry.geom_io import load_data, parse_geometry

def print_fixed_status(mesh, label):
    """Print the fixed status of all mesh elements"""
    print(f"\n=== {label} ===")
    
    print("Vertices:")
    for vid, vertex in mesh.vertices.items():
        fixed_status = getattr(vertex, 'fixed', False)
        print(f"  V{vid}: fixed={fixed_status}, options={vertex.options}")
    
    print("Edges:")  
    for eid, edge in mesh.edges.items():
        fixed_status = getattr(edge, 'fixed', False)
        print(f"  E{eid}: fixed={fixed_status}, options={edge.options}")
    
    print("Facets:")
    for fid, facet in mesh.facets.items():
        fixed_status = getattr(facet, 'fixed', False)
        print(f"  F{fid}: fixed={fixed_status}, options={facet.options}")

print("Loading and parsing cube.json...")

# Load cube.json
try:
    data = load_data("meshes/cube.json")
    print(f"Loaded data keys: {data.keys()}")
    
    # Check raw input data for any fixed properties
    print("\nChecking raw input data:")
    
    for i, vertex_data in enumerate(data["vertices"]):
        if isinstance(vertex_data, list) and len(vertex_data) > 3:
            if isinstance(vertex_data[-1], dict):
                print(f"  Vertex {i} has options: {vertex_data[-1]}")
    
    for i, edge_data in enumerate(data["edges"]):
        if len(edge_data) > 2:
            print(f"  Edge {i} has options: {edge_data[2:]}")
    
    for i, face_data in enumerate(data["faces"]):
        if isinstance(face_data[-1], dict):
            print(f"  Face {i} has options: {face_data[-1]}")
    
    # Parse the geometry
    mesh = parse_geometry(data)
    print_fixed_status(mesh, "After parsing cube.json")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\nThis shows what the original cube.json produces - any fixed=True here indicates the issue.")