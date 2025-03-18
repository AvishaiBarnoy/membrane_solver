import numpy as np

class VolumeConstraintModule:
    def __init__(self, target_volume, k=0.1, body_faces=None):
        """
        target_volume: desired volume.
        k: pressure coefficient.
        body_faces: optional list of face indices to restrict the volume computation.
        """
        self.target_volume = target_volume
        self.k = k
        self.body_faces = body_faces

    def compute_volume(self, mesh):
        volume = 0.0
        if self.body_faces is None:
            faces_to_use = range(len(mesh.faces))
        else:
            faces_to_use = self.body_faces
        for idx in faces_to_use:
            face = mesh.faces[idx]
            if len(face) < 3:
                continue
            v0 = mesh.vertices[face[0]].position
            for i in range(1, len(face)-1):
                v1 = mesh.vertices[face[i]].position
                v2 = mesh.vertices[face[i+1]].position
                volume += np.dot(v0, np.cross(v1, v2)) / 6.0
        return abs(volume)

    def compute_vertex_normals(self, mesh):
        normals = [np.zeros(3, dtype=float) for _ in mesh.vertices]
        if self.body_faces is None:
            faces_to_use = range(len(mesh.faces))
        else:
            faces_to_use = self.body_faces
        for idx in faces_to_use:
            face = mesh.faces[idx]
            if len(face) < 3:
                continue
            v0 = mesh.vertices[face[0]].position
            v1 = mesh.vertices[face[1]].position
            v2 = mesh.vertices[face[2]].position
            face_normal = np.cross(v1 - v0, v2 - v0)
            for vi in face:
                normals[vi] += face_normal
        for i in range(len(normals)):
            norm = np.linalg.norm(normals[i])
            if norm > 1e-8:
                normals[i] /= norm
        return normals

    def modify_forces(self, mesh):
        current_volume = self.compute_volume(mesh)
        delta_vol = self.target_volume - current_volume
        pressure = self.k * delta_vol
        normals = self.compute_vertex_normals(mesh)
        for i, vertex in enumerate(mesh.vertices):
            vertex.force += pressure * normals[i]

