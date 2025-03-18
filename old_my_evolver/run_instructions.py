# run_instructions.py
import sys
import os
import numpy as np
from geometry_input import load_geometry
from save_result import safe_save_mesh
from geometry import Vertex

# Import a sample constraint module (if desired)
from modules.constraint_module import ConstraintModule

# Define Mesh and Evolver classes (or import them from your main engine file)
class Mesh:
    def __init__(self, vertices, faces):
        mesh.vertices = vertices  # List of Vertex objects
        self.faces = faces        # List of tuples (indices)

    def compute_energy(self):
        total_energy = 0.0
        for face in self.faces:
            v0 = mesh.vertices[face[0]].position
            v1 = mesh.vertices[face[1]].position
            v2 = mesh.vertices[face[2]].position
            area = np.linalg.norm(np.cross(v1 - v0, v2 - v0)) / 2.0
            total_energy += area
        return total_energy

    def compute_forces(self):
        # Reset forces to zero.
        for vertex in mesh.vertices:
            vertex.force = np.zeros(3, dtype=float)
        for face in self.faces:
            v0 = mesh.vertices[face[0]]
            v1 = mesh.vertices[face[1]]
            v2 = mesh.vertices[face[2]]
            dummy_force = np.array([0.1, 0.1, 0.1])
            v0.force += dummy_force
            v1.force += dummy_force
            v2.force += dummy_force

    def refine(self):
        new_vertices = mesh.vertices[:]  # Copy current vertices
        new_faces = []
        for face in self.faces:
            v0 = mesh.vertices[face[0]].position
            v1 = mesh.vertices[face[1]].position
            v2 = mesh.vertices[face[2]].position
            centroid = (v0 + v1 + v2) / 3.0
            new_vertex = Vertex(centroid)
            new_index = len(new_vertices)
            new_vertices.append(new_vertex)
            # Subdivide the face into three new faces.
            new_faces.append((face[0], face[1], new_index))
            new_faces.append((face[1], face[2], new_index))
            new_faces.append((face[2], face[0], new_index))
        mesh.vertices = new_vertices
        self.faces = new_faces
        print("Mesh refined: now has", len(mesh.vertices), "vertices and", len(self.faces), "faces.")

class Evolver:
    def __init__(self, mesh, algorithm="gradient_descent"):
        self.mesh = mesh
        self.modules = []  # Additional modules (constraints, volume, etc.)
        self.algorithm = algorithm

    def add_module(self, module):
        self.modules.append(module)

    def step(self, dt=0.01):
        self.mesh.compute_forces()
        for module in self.modules:
            module.modify_forces(self.mesh)
        if self.algorithm == "gradient_descent":
            for vertex in self.mesh.vertices:
                vertex.position -= dt * vertex.force
        elif self.algorithm == "conjugate_gradient":
            for vertex in self.mesh.vertices:
                if not vertex.initialized_cg:
                    vertex.search_direction = -vertex.force
                    vertex.initialized_cg = True
                else:
                    prev_norm_sq = np.dot(vertex.prev_force, vertex.prev_force)
                    beta = np.dot(vertex.force, vertex.force) / prev_norm_sq if prev_norm_sq != 0 else 0.0
                    vertex.search_direction = -vertex.force + beta * vertex.search_direction
                vertex.position += dt * vertex.search_direction
                vertex.prev_force = vertex.force.copy()
        else:
            raise ValueError("Unknown minimization algorithm: " + self.algorithm)

    def run(self, steps=100, dt=0.01):
        for i in range(steps):
            self.step(dt)
            energy = self.mesh.compute_energy()
            print(f"Step {i+1}: Energy = {energy:.4f}")

def volume_estimate(mesh):
    volume = 0.0
    for face in mesh.faces:
        v0 = mesh.vertices[face[0]].position
        v1 = mesh.vertices[face[1]].position
        v2 = mesh.vertices[face[2]].position
        volume += np.dot(v0, np.cross(v1, v2)) / 6.0
    return abs(volume)

def run_instructions(instruction_file, evolver, mesh):
    saved_during_run = False
    with open(instruction_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines or lines enclosed in triple quotes/comments.
            if not line or line.startswith('"""') or line.startswith('#'):
                continue
            parts = line.split()
            command = parts[0].lower()
            if command == "gradient":
                if len(parts) >= 2 and parts[1].lower() == "descent":
                    evolver.algorithm = "gradient_descent"
                    print("Switched to gradient descent")
            elif command == "conjugate":
                if len(parts) >= 2 and parts[1].lower() == "gradients":
                    evolver.algorithm = "conjugate_gradient"
                    for v in mesh.vertices:
                        v.initialized_cg = False
                    print("Switched to conjugate gradients")
            elif command == "opt":
                if len(parts) >= 2:
                    steps = int(parts[1])
                    print(f"Optimizing for {steps} steps using {evolver.algorithm}")
                    evolver.run(steps=steps, dt=0.01)
            elif command == "save_geometry":
                safe_save_mesh(mesh.vertices, mesh.faces, "meshes/intermediate_geometry.json")
                print("Intermediate geometry saved.")
                saved_during_run = True
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
                # Add the volume constraint module.
                from modules.volume_constraint_module import VolumeConstraintModule
                target = None
                if len(parts) >= 2:
                    try:
                        target = float(parts[1])
                    except ValueError:
                        target = None
                if target is None:
                    target = volume_estimate(mesh)
                    print(f"No target specified for volume; using current volume {target:.4f} as target.")
                volume_module = VolumeConstraintModule(target_volume=target)
                evolver.add_module(volume_module)
                print(f"Volume constraint module added with target volume {target}.")
            else:
                print(f"Unknown command: {command}")
    if not saved_during_run:
        # By default, save the final geometry.
        safe_save_mesh(mesh.vertices, mesh.faces, "meshes/final_geometry.json")
        print("Final geometry saved by default.")

def main():
    # Load an input geometry (e.g., from a JSON file).
    vertices, faces = load_geometry("meshes/sample_geometry.json")
    mesh = Mesh(vertices, faces)
    evolver = Evolver(mesh, algorithm="gradient_descent")
    # (Optionally, you can add other modules here too, e.g. ConstraintModule.)
    if len(sys.argv) < 2:
        print("Usage: python run_instructions.py <instruction_file>")
        sys.exit(1)
    instruction_file = sys.argv[1]
    run_instructions(instruction_file, evolver, mesh)

if __name__ == '__main__':
    main()

