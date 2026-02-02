import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.curvature import compute_curvature_data  # noqa: E402
from geometry.geom_io import load_data, parse_geometry  # noqa: E402
from runtime.constraint_manager import ConstraintModuleManager  # noqa: E402
from runtime.energy_manager import EnergyModuleManager  # noqa: E402
from runtime.minimizer import Minimizer  # noqa: E402
from runtime.steppers.gradient_descent import GradientDescent  # noqa: E402


@pytest.mark.e2e
def test_kozlov_free_disk_flat_state_has_large_boundary_curvature_baseline() -> None:
    """Diagnostic regression for the flat-reference energy gap vs docs/tex/1_disk_3d.tex.

    In the continuum theory, a perfectly flat membrane patch with zero tilt has
    zero elastic energy. On the current discrete free-disk mesh, the curvature
    operator assigns a nonzero mean-curvature proxy on the *open boundary* even
    when all vertices are coplanar. This baseline then feeds into bending_tilt_*.

    This test pins down the diagnostic facts (boundary dominance) so the follow-up
    physics PR can remove or subtract the correct reference contribution.
    """
    path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "meshes",
        "caveolin",
        "kozlov_1disk_3d_tensionless_single_leaflet_profile_hard_rim_R12_free_disk.yaml",
    )
    mesh = parse_geometry(load_data(path))

    # Force a flat, zero-tilt reference state.
    for vertex in mesh.vertices.values():
        vertex.position[2] = 0.0
    mesh.increment_version()

    n = len(mesh.vertex_ids)
    mesh.set_tilts_in_from_array(np.zeros((n, 3), dtype=float))
    mesh.set_tilts_out_from_array(np.zeros((n, 3), dtype=float))
    mesh.global_parameters.set("tilt_thetaB_value", 0.0)

    minim = Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
    )
    breakdown = minim.compute_energy_breakdown()

    # Baseline: bending_tilt_in is large even when the surface is perfectly flat.
    # (We use a loose threshold; the point is "nontrivial and not numerical noise".)
    assert float(breakdown.get("bending_tilt_in") or 0.0) > 10.0

    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row
    k_vecs, _areas, _weights, _tri_rows = compute_curvature_data(
        mesh, positions, index_map
    )
    k_mag = np.linalg.norm(k_vecs, axis=1)

    boundary_vids = sorted(mesh.boundary_vertex_ids)
    assert boundary_vids, "expected an open boundary loop on this mesh"
    boundary_rows = np.asarray(
        [index_map[int(vid)] for vid in boundary_vids], dtype=int
    )

    total = float(np.sum(k_mag))
    boundary = float(np.sum(k_mag[boundary_rows]))
    share = boundary / total if total > 0.0 else 0.0

    # Diagnostic fact: the boundary dominates the curvature baseline.
    assert share > 0.85

    # Interior curvature should be comparatively small for the coplanar surface.
    interior_mask = np.ones_like(k_mag, dtype=bool)
    interior_mask[boundary_rows] = False
    assert float(np.max(k_mag[interior_mask])) < 1.0
