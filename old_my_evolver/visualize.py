#!/usr/bin/env python
import argparse
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def load_mesh(filename):
    """
    Load mesh geometry from a JSON file.
    Expected keys:
      "vertices": list of [x, y, z]
      "edges": (optional) list of [i, j] defining connectivity between vertices
      "faces": (optional) list of lists of vertex indices for transparent facets
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    vertices = data.get("vertices", [])
    edges = data.get("edges", None)
    faces = data.get("faces", None)
    return vertices, edges, faces

def visualize_mesh(vertices, edges, faces):
    """
    Plot the mesh:
      - Vertices are red dots.
      - Edges (if provided) are drawn as blue lines.
      - Faces (if provided) are drawn as transparent surfaces.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot vertices.
    if vertices:
        coords = np.array(vertices)
        ax.scatter(coords[:,0], coords[:,1], coords[:,2], color='red', s=50, label='Vertices')
    
    # Plot edges.
    if edges:
        for edge in edges:
            i, j = edge
            p1 = vertices[i]
            p2 = vertices[j]
            xs = [p1[0], p2[0]]
            ys = [p1[1], p2[1]]
            zs = [p1[2], p2[2]]
            ax.plot(xs, ys, zs, color='blue')
    
    # Plot faces.
    if faces:
        face_polys = []
        for face in faces:
            poly = [vertices[i] for i in face]
            face_polys.append(poly)
        poly3d = Poly3DCollection(face_polys, alpha=0.3, facecolor='cyan', edgecolor='k')
        ax.add_collection3d(poly3d)
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.title("Mesh Visualization")
    plt.legend()
    plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="Visualize a geometry JSON file (vertices, edges, and faces) with transparent facets."
    )
    parser.add_argument('-i', '--input', required=False,
                        help="Input JSON file with mesh geometry.")
    args = parser.parse_args()
    if args.input == "":
        args.input = sys.argv[1]
        print("found it")
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist.")
        sys.exit(1)
    
    vertices, edges, faces = load_mesh(args.input)
    visualize_mesh(vertices, edges, faces)

if __name__ == '__main__':
    main()

