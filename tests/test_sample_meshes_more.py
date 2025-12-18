import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.sample_meshes import generate_open_cylinder


def test_generate_open_cylinder_basic_counts_and_fixed_rings():
    mesh = generate_open_cylinder(radius=1.0, height=2.0, n_segments=6)

    assert len(mesh.vertices) == 12
    assert len(mesh.edges) > 0
    assert len(mesh.facets) == 12  # 2 triangles per segment
    assert 1 in mesh.bodies
    assert mesh.bodies[1].target_volume is not None

    for i in range(6):
        assert mesh.vertices[i].fixed is True
        assert mesh.vertices[i + 6].fixed is True

    # Sanity: body volume computed after build, so target_volume is finite.
    assert np.isfinite(mesh.bodies[1].target_volume)
