import os
import json

def safe_save_mesh(vertices, faces, output_filename, edges=None, body=None):
    base, ext = os.path.splitext(output_filename)
    counter = 1
    new_filename = output_filename
    while os.path.exists(new_filename):
        new_filename = f"{base}_{counter}{ext}"
        counter += 1
    data = {
        "vertices": [vertex.position.tolist() for vertex in vertices],
        "faces": faces
    }
    if edges is not None:
        data["edges"] = list(edges)
    if body is not None:
        data["body"] = body
    with open(new_filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Mesh saved to {new_filename}")
    return new_filename

if __name__ == '__main__':
    # Test saving with a dummy mesh.
    class DummyVertex:
        def __init__(self, pos):
            self.position = pos
    vertices = [DummyVertex([0, 0, 0]), DummyVertex([1, 0, 0]), DummyVertex([0, 1, 0])]
    faces = [(0, 1, 2)]
    safe_save_mesh(vertices, faces, "meshes/test_save.json")

