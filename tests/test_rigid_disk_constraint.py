import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.entities import GlobalParameters, Mesh, Vertex  # noqa: E402
from modules.constraints import rigid_disk  # noqa: E402


def _pairwise_distances(pts: np.ndarray) -> np.ndarray:
    n = pts.shape[0]
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(float(np.linalg.norm(pts[i] - pts[j])))
    return np.asarray(dists, dtype=float)


def _fit_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centroid = np.mean(points, axis=0)
    X = points - centroid
    _, _, vh = np.linalg.svd(X, full_matrices=False)
    normal = vh[-1, :]
    normal = normal / max(float(np.linalg.norm(normal)), 1e-12)
    return centroid, normal


def test_rigid_disk_preserves_pairwise_distances_and_rim_radius() -> None:
    mesh = Mesh()
    mesh.global_parameters = GlobalParameters(
        {
            "rigid_disk_group": "diskA",
            "rigid_disk_radius": 1.0,
            "rigid_disk_rim_group": "rim",
        }
    )

    verts = [
        np.array([1.0, 0.0, 0.0], dtype=float),
        np.array([0.0, 1.0, 0.0], dtype=float),
        np.array([-1.0, 0.0, 0.0], dtype=float),
        np.array([0.0, -1.0, 0.0], dtype=float),
    ]
    mesh.vertices = {
        i: Vertex(
            i,
            pos.copy(),
            options={
                "preset": "disk",
                "rigid_disk_group": "diskA",
                "rim_slope_match_group": "rim",
            },
        )
        for i, pos in enumerate(verts)
    }

    # Initialize reference configuration.
    rigid_disk.enforce_constraint(mesh, mesh.global_parameters)

    ref = np.array([v.position for v in mesh.vertices.values()], dtype=float)
    ref_dists = _pairwise_distances(ref)

    # Perturb one vertex out of plane and radially.
    mesh.vertices[2].position += np.array([0.1, -0.05, 0.2], dtype=float)
    rigid_disk.enforce_constraint(mesh, mesh.global_parameters)

    cur = np.array([v.position for v in mesh.vertices.values()], dtype=float)
    cur_dists = _pairwise_distances(cur)
    assert np.allclose(cur_dists, ref_dists, atol=1e-10, rtol=0.0)

    center, normal = _fit_plane(cur)
    rel = cur - center[None, :]
    rel_plane = rel - (rel @ normal)[:, None] * normal[None, :]
    radii = np.linalg.norm(rel_plane, axis=1)
    assert np.allclose(radii, 1.0, atol=1e-10, rtol=0.0)
