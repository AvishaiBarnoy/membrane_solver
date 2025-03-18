#!/usr/bin/env python
import argparse
import json
import os
import sys
import numpy as np

from geometry import Vertex
from save_result import safe_save_mesh
from modules.volume_constraint_module import VolumeConstraintModule
# (Assume any additional modules such as ConstraintModule are imported if needed)

# --- Mesh Class with Surface Tension Energy and Analytical Gradient ---
class Mesh:
    def __init__(self, vertices, faces, edges=None):
        self.vertices = vertices  # list of Vertex objects
        self.faces = faces        # list of tuples (indices)
        self.edges = edges if edges is not None else []
        self.epsilon = 1e-6  # default epsilon (for finite differences if needed)
        if self.edges:
            self.edge_target_lengths = []
            for edge in self.edges:
                i, j = edge
                v0 = self.vertices[i].position
                v1 = self.vertices[j].position
                self.edge_target_lengths.append(np.linalg.norm(v1 - v0))
        else:
            self.edge_target_lengths = []

    def compute_area_energy(self):
        """Total surface area computed by triangulating each face."""
        total_energy = 0.0
        for face in self.faces:
            if len(face) < 3:
                continue
            v0 = self.vertices[face[0]].position
            for i in range(1, len(face) - 1):
                v1 = self.vertices[face[i]].position
                v2 = self.vertices[face[i+1]].position
                area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
                total_energy += area
        return total_energy

    def compute_energy(self):
        return self.compute_area_energy()

    def compute_forces(self):
        """
        Compute forces as the negative analytical gradient of the surface area.
        For each triangle (with vertices v0, v1, v2 obtained by triangulating each face using v0 as common vertex):
           grad(v0) = -0.5 * ((v1 - v2) x n)
           grad(v1) = -0.5 * ((v2 - v0) x n)
           grad(v2) = -0.5 * ((v0 - v1) x n)
        where n is the unit normal of the triangle.
        The forces from all triangles are summed at each vertex.
        """
        # Reset forces.
        for vertex in self.vertices:
            vertex.force = np.zeros(3, dtype=float)
        for face in self.faces:
            if len(face) < 3:
                continue
            v0_idx = face[0]
            v0 = self.vertices[v0_idx].position
            for i in range(1, len(face)-1):
                i1 = face[i]
                i2 = face[i+1]
                v1 = self.vertices[i1].position
                v2 = self.vertices[i2].position
                cross_vec = np.cross(v1 - v0, v2 - v0)
                area = 0.5 * np.linalg.norm(cross_vec)
                if area < 1e-10:
                    continue
                n = cross_vec / np.linalg.norm(cross_vec)
                grad0 = -0.5 * np.cross(v1 - v2, n)
                grad1 = -0.5 * np.cross(v2 - v0, n)
                grad2 = -0.5 * np.cross(v0 - v1, n)
                self.vertices[v0_idx].force += grad0
                self.vertices[i1].force += grad1
                self.vertices[i2].force += grad2

    def refine(self):
        """
        Refine the mesh similar to Surface Evolver's REFINE command:
          - For each face, insert vertices at the midpoints of its edges and a vertex at the face centroid.
          - Subdivide the face (via triangulation) into smaller facets.
          - Rebuild the edge connectivity and recalcualte target edge lengths.
        """
        new_vertices = self.vertices[:]  # Copy current Vertex objects.
        new_faces = []
        edge_midpoints = {}  # Map (min(i,j), max(i,j)) -> new vertex index.
        for face in self.faces:
            n = len(face)
            face_mid_idx = []
            for i in range(n):
                i1 = face[i]
                i2 = face[(i+1) % n]
                key = tuple(sorted((i1, i2)))
                if key not in edge_midpoints:
                    v1 = self.vertices[i1].position
                    v2 = self.vertices[i2].position
                    midpoint = ((v1 + v2) / 2.0).tolist()
                    new_index = len(new_vertices)
                    new_vertices.append(Vertex(midpoint))
                    edge_midpoints[key] = new_index
                face_mid_idx.append(edge_midpoints[key])
            pts = [self.vertices[idx].position for idx in face]
            centroid = np.mean(pts, axis=0).tolist()
            centroid_index = len(new_vertices)
            new_vertices.append(Vertex(centroid))
            for i in range(n):
                tri1 = (face[i], face_mid_idx[i], centroid_index)
                tri2 = (face_mid_idx[i], face[(i+1) % n], centroid_index)
                new_faces.append(tri1)
                new_faces.append(tri2)
        self.vertices = new_vertices
        self.faces = new_faces
        new_edges = set()
        for face in new_faces:
            m = len(face)
            for i in range(m):
                edge = tuple(sorted((face[i], face[(i+1) % m])))
                new_edges.add(edge)
        self.edges = list(new_edges)
        self.edge_target_lengths = []
        for edge in self.edges:
            i, j = edge
            v0 = self.vertices[i].position
            v1 = self.vertices[j].position
            self.edge_target_lengths.append(np.linalg.norm(v1 - v0))
        print("Mesh refined: now has", len(self.vertices), "vertices,",
              len(self.faces), "faces, and", len(self.edges), "edges.")

# --- Evolver Class with Adjustable dt ---
class Evolver:
    def __init__(self, mesh, algorithm="gradient_descent"):
        self.mesh = mesh
        self.modules = []  # Additional modules (e.g., volume constraint)
        self.algorithm = algorithm
        self.dt = 0.0001  # default time step (small to avoid explosion)

    def add_module(self, module):
        self.modules.append(module)

    def step(self, dt=None):
        if dt is None:
            dt = self.dt
        self.mesh.compute_forces()
        for module in self.modules:
            module.modify_forces(self.mesh)
        if self.algorithm == "gradient_descent":
            for vertex in self.mesh.vertices:
                vertex.position -= dt * vertex.force
        elif self.algorithm == "conjugate_gradient":
            for vertex in self.mesh.vertices:
                if not hasattr(vertex, 'initialized_cg') or not vertex.initialized_cg:
                    vertex.search_direction = -vertex.force
                    vertex.initialized_cg = True
                else:
                    prev_norm_sq = np.dot(vertex.prev_force, vertex.prev_force)
                    beta = (np.dot(vertex.force, vertex.force) / prev_norm_sq) if prev_norm_sq != 0 else 0.0
                    vertex.search_direction = -vertex.force + beta * vertex.search_direction
                vertex.position += dt * vertex.search_direction
                vertex.prev_force = vertex.force.copy()
        else:
            raise ValueError("Unknown minimization algorithm: " + self.algorithm)

    def run(self, steps, dt=None):
        if dt is None:
            dt = self.dt
        for i in range(steps):
            self.step(dt)
            energy = self.mesh.compute_energy()
            print(f"Step {i+1}: surface tension energy = {energy:.6f}")

# --- Utility: Estimate Global Volume ---
def volume_estimate(mesh):
    volume = 0.0
    for face in mesh.faces:
        if len(face) < 3:
            continue
        v0 = mesh.vertices[face[0]].position
        for i in range(1, len(face)-1):
            v1 = mesh.vertices[face[i]].position
            v2 = mesh.vertices[face[i+1]].position
            volume += np.dot(v0, np.cross(v1, v2)) / 6.0
    return abs(volume)

# --- Instruction Processing with New Commands ---
def process_instructions(instructions, evolver, mesh, output_filename, body_data=None):
    saved_during_run = False
    for cmd in instructions:
        line = cmd.strip()
        if not line:
            continue
        parts = line.split()
        command = parts[0].lower()
        # Accept various forms for conjugate gradient.
        if (command == "conjugate" and len(parts) >= 2 and parts[1].lower() in ("gradient", "gradients")) or command == "conjugate_gradient":
            evolver.algorithm = "conjugate_gradient"
            for v in mesh.vertices:
                v.initialized_cg = False
            print("Switched to conjugate gradients")
        elif command == "gradient" and len(parts) >= 2 and parts[1].lower() == "descent":
            evolver.algorithm = "gradient_descent"
            print("Switched to gradient descent")
        elif command == "opt":
            if len(parts) >= 2:
                steps = int(parts[1])
                print(f"Optimizing for {steps} steps using {evolver.algorithm}")
                evolver.run(steps=steps)
        elif command == "refine":
            refine_times = 1
            if len(parts) >= 2:
                try:
                    refine_times = int(parts[1])
                except ValueError:
                    refine_times = 1
            for _ in range(refine_times):
                mesh.refine()
            print(f"Refined mesh {refine_times} time(s).")
        elif command == "volume":
            target = None
            if len(parts) >= 2:
                try:
                    target = float(parts[1])
                except ValueError:
                    target = None
            if target is None:
                target = volume_estimate(mesh)
                print(f"No target specified for volume; using current volume {target:.6f} as target.")
            volume_module = VolumeConstraintModule(target_volume=target, k=0.1)
            evolver.add_module(volume_module)
            print(f"Global volume constraint module added with target volume {target}.")
        elif command == "set_dt":
            if len(parts) >= 2:
                try:
                    new_dt = float(parts[1])
                    evolver.dt = new_dt
                    print(f"Time step (dt) set to {new_dt}.")
                except ValueError:
                    print("Invalid dt value.")
        elif command == "set_epsilon":
            if len(parts) >= 2:
                try:
                    new_epsilon = float(parts[1])
                    mesh.epsilon = new_epsilon
                    print(f"Epsilon set to {new_epsilon}.")
                except ValueError:
                    print("Invalid epsilon value.")
        elif command == "set_volume_k":
            if len(parts) >= 2:
                try:
                    new_k = float(parts[1])
                    for mod in evolver.modules:
                        if hasattr(mod, 'k'):
                            mod.k = new_k
                    print(f"Volume constraint coefficient (k) set to {new_k}.")
                except ValueError:
                    print("Invalid volume k value.")
        elif command == "save_geometry":
            body_info = None
            if body_data is not None:
                body_info = {"original_body": body_data,
                             "global_volume": volume_estimate(mesh)}
            safe_save_mesh(mesh.vertices, mesh.faces, output_filename, edges=mesh.edges, body=body_info)
            print("Geometry saved.")
            saved_during_run = True
        else:
            print(f"Unknown command: {line}")
    if output_filename is not None and not saved_during_run:
        body_info = None
        if body_data is not None:
            body_info = {"original_body": body_data,
                         "global_volume": volume_estimate(mesh)}
        safe_save_mesh(mesh.vertices, mesh.faces, output_filename, edges=mesh.edges, body=body_info)
        print("Final geometry saved by default.")
    elif output_filename is None and not saved_during_run:
        print("Final geometry not saved (interactive mode).")

# --- Main Function with CLI Options and Interactive Mode ---
def main():
    parser = argparse.ArgumentParser(
        description="Surface Evolver with Surface Tension Energy, Volume Constraint, Refinement, and Interactive Mode"
    )
    parser.add_argument('-i', '--input', required=True,
                        help="Input JSON file with mesh, bodies, and instructions")
    parser.add_argument('-o', '--output',
                        help="Output file for final mesh geometry (default: final_geometry.json in input file folder)")
    parser.add_argument('-I', '--instr_file',
                        help="External instructions file (if provided, it overrides instructions in the input file)")
    parser.add_argument('--interactive', action='store_true',
                        help="Enable interactive mode (enter commands in the terminal)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input file {args.input} does not exist.")
        sys.exit(1)
    with open(args.input, 'r') as f:
        data = json.load(f)
    
    vertices_data = data.get("vertices", [])
    faces_data = data.get("faces", [])
    edges_data = data.get("edges", None)
    instructions = data.get("instructions", [])
    body_data = data.get("body", None)

    vertices = [Vertex(pos) for pos in vertices_data]
    faces = [tuple(face) for face in faces_data] if faces_data else []
    edges = [tuple(edge) for edge in edges_data] if edges_data else []

    mesh = Mesh(vertices, faces, edges)
    evolver = Evolver(mesh, algorithm="gradient_descent")

    if args.instr_file:
        if instructions:
            print("Instructions found in input file, but external instructions file specified; using external file.")
        with open(args.instr_file, 'r') as f:
            instructions = [line.strip() for line in f if line.strip()]

    # Set default output file if not specified.
    if args.output is None:
        input_dir = os.path.dirname(os.path.abspath(args.input))
        args.output = os.path.join(input_dir, "final_geometry.json")
        print(f"No output file specified. Using default: {args.output}")

    # If body information is provided and no explicit "volume" instruction is given, add a volume constraint.
    volume_instructions = any(instr.strip().lower().startswith("volume") for instr in instructions)
    if body_data is not None and not volume_instructions:
        if isinstance(body_data.get("target_volume"), list):
            target = body_data.get("target_volume")[0]
        else:
            target = body_data.get("target_volume")
        if target is not None:
            volume_module = VolumeConstraintModule(target_volume=target, k=0.1)
            evolver.add_module(volume_module)
            print(f"Volume constraint module added from input file with target volume {target}")
        else:
            print("Body data provided but no target_volume specified.")

    if args.interactive:
        print("Entering interactive mode. Type commands (or 'exit' to quit).")
        while True:
            cmd = input(">> ")
            if cmd.lower() in ("exit", "quit"):
                break
            process_instructions([cmd], evolver, mesh, None, body_data)
            current_energy = evolver.mesh.compute_energy()
            print(f"Current surface tension energy: {current_energy:.6f}")
        print("Exiting interactive mode.")
    else:
        process_instructions(instructions, evolver, mesh, args.output, body_data)

if __name__ == '__main__':
    main()

