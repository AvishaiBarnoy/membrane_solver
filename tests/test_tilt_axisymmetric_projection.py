import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from runtime.minimizer import Minimizer


def test_project_tilts_axisymmetric_about_center_keeps_only_radial_component() -> None:
    positions = np.array(
        [
            [1.0, 0.0, 0.0],  # radial = +x, azimuthal = +y
            [0.0, 1.0, 0.0],  # radial = +y, azimuthal = -x
            [0.0, 0.0, 0.0],  # center (degenerate radial)
        ],
        dtype=float,
        order="F",
    )
    normals = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=float), (3, 1))
    center = np.array([0.0, 0.0, 0.0], dtype=float)
    axis = np.array([0.0, 0.0, 1.0], dtype=float)

    tilts = np.array(
        [
            [1.0, 2.0, 0.0],  # radial + azimuthal
            [-4.0, 3.0, 0.0],  # azimuthal + radial
            [5.0, 6.0, 0.0],  # ignored at center (degenerate radial)
        ],
        dtype=float,
        order="F",
    )

    fixed_mask = np.array([False, True, False], dtype=bool)
    proj = Minimizer._project_tilts_axisymmetric_about_center(
        positions=positions,
        tilts=tilts,
        normals=normals,
        center=center,
        axis=axis,
        fixed_mask=fixed_mask,
    )

    # Fixed vertex is unchanged.
    assert np.allclose(proj[1], tilts[1])

    # Vertex at (1,0) keeps only +x radial component (azimuthal removed).
    assert np.allclose(proj[0], np.array([1.0, 0.0, 0.0], dtype=float))

    # Center has no well-defined radial direction; projection returns zero.
    assert np.allclose(proj[2], np.zeros(3, dtype=float))

    # All projected tilts remain tangent to the provided normals.
    dots = np.einsum("ij,ij->i", proj, normals)
    assert float(np.max(np.abs(dots))) < 1e-12
