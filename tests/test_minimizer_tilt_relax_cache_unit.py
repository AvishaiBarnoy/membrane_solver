import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.parameters.global_parameters import GlobalParameters
from geometry.entities import Edge, Facet, Mesh, Vertex
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.gradient_descent import GradientDescent


def _build_single_triangle_mesh_with_presets() -> Mesh:
    mesh = Mesh()
    mesh.vertices = {
        0: Vertex(0, np.array([0.0, 0.0, 0.0]), options={"preset": "disk"}),
        1: Vertex(1, np.array([1.0, 0.0, 0.0]), options={"preset": "rim"}),
        2: Vertex(2, np.array([0.0, 1.0, 0.0]), options={"preset": "rim"}),
    }
    mesh.edges = {1: Edge(1, 0, 1), 2: Edge(2, 1, 2), 3: Edge(3, 2, 0)}
    mesh.facets = {0: Facet(0, edge_indices=[1, 2, 3])}
    mesh.build_facet_vertex_loops()
    mesh.build_position_cache()
    return mesh


def _build_minimizer(mesh: Mesh, gp: GlobalParameters) -> Minimizer:
    # These tests only touch internal caching helpers; no energy modules are needed.
    mesh.energy_modules = []
    mesh.constraint_modules = []
    return Minimizer(
        mesh,
        gp,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
    )


def test_leaflet_out_absence_mask_cache_invalidates_on_gp_change() -> None:
    mesh = _build_single_triangle_mesh_with_presets()
    gp = GlobalParameters({"leaflet_out_absent_presets": ["disk"]})
    minim = _build_minimizer(mesh, gp)

    tri_rows, _ = mesh.triangle_row_cache()
    absent1, keep1 = minim._leaflet_out_absence_masks(tri_rows=tri_rows)
    assert absent1.shape == (len(mesh.vertex_ids),)
    assert bool(absent1[0]) is True
    assert bool(absent1[1]) is False
    assert keep1.shape == (len(tri_rows),)

    # Same inputs: should reuse cached arrays (object identity).
    absent2, keep2 = minim._leaflet_out_absence_masks(tri_rows=tri_rows)
    assert absent2 is absent1
    assert keep2 is keep1

    # Changing global parameters must invalidate the cache.
    gp.set("leaflet_out_absent_presets", [])
    absent3, keep3 = minim._leaflet_out_absence_masks(tri_rows=tri_rows)
    assert absent3 is not absent1
    assert keep3 is not keep1
    assert bool(absent3[0]) is False


def test_tilt_relax_scratch_reuses_and_resizes() -> None:
    mesh = _build_single_triangle_mesh_with_presets()
    gp = GlobalParameters({})
    minim = _build_minimizer(mesh, gp)

    g0, gin0, gout0 = minim._tilt_relax_scratch(n_vertices=3)
    g1, gin1, gout1 = minim._tilt_relax_scratch(n_vertices=3)
    assert g1 is g0
    assert gin1 is gin0
    assert gout1 is gout0

    g2, gin2, gout2 = minim._tilt_relax_scratch(n_vertices=4)
    assert g2 is not g0
    assert gin2 is not gin0
    assert gout2 is not gout0
    assert g2.shape == (4, 3)
    assert gin2.shape == (4, 3)
    assert gout2.shape == (4, 3)
