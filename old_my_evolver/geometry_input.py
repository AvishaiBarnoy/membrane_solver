# geometry_input.py
import json
from geometry import Vertex

def load_geometry(filename):
    """
    Load geometry from a JSON file.
    
    Expected JSON format:
    {
        "vertices": [[x, y, z], ...],
        "faces": [[i, j, k], ...]
    }
    
    Returns:
        vertices (list of Vertex): List of vertex objects.
        faces (list of tuple): List of faces (each a tuple of indices).
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    vertices = [Vertex(pos) for pos in data["vertices"]]
    faces = [tuple(face) for face in data["faces"]]
    return vertices, faces

if __name__ == '__main__':
    # For testing: load a sample geometry and print it.
    vertices, faces = load_geometry("meshes/sample_geometry.json")
    print("Loaded vertices:")
    for v in vertices:
        print(v.position)
    print("Loaded faces:", faces)

