import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.parameters.global_parameters import GlobalParameters  # noqa: E402
from geometry.entities import Edge, Facet, Mesh, Vertex  # noqa: E402
from runtime.constraint_manager import ConstraintModuleManager  # noqa: E402
from runtime.energy_manager import EnergyModuleManager  # noqa: E402
from runtime.minimizer import Minimizer  # noqa: E402
from runtime.steppers.gradient_descent import GradientDescent  # noqa: E402


def _fan_disk_mesh(*, n_ring: int = 12, radius: float = 1.0) -> Mesh:
    """Return a tiny planar disk mesh with a tagged boundary ring."""
    mesh = Mesh()
    mesh.global_parameters = GlobalParameters()

    mesh.vertices[0] = Vertex(0, np.array([0.0, 0.0, 0.0]), fixed=True)
    for k in range(n_ring):
        theta = 2.0 * np.pi * float(k) / float(n_ring)
        x = float(radius) * float(np.cos(theta))
        y = float(radius) * float(np.sin(theta))
        mesh.vertices[k + 1] = Vertex(
            k + 1,
            np.array([x, y, 0.0]),
            fixed=True,
            options={"rim_slope_match_group": "disk"},
        )

    triangles: list[tuple[int, int, int]] = []
    for k in range(n_ring):
        a = 0
        b = k + 1
        c = ((k + 1) % n_ring) + 1
        triangles.append((a, b, c))

    edge_map: dict[tuple[int, int], int] = {}
    next_eid = 1
    for fidx, tri in enumerate(triangles, start=1):
        e_ids: list[int] = []
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            key = tuple(sorted((a, b)))
            if key not in edge_map:
                edge_map[key] = next_eid
                mesh.edges[next_eid] = Edge(next_eid, a, b)
                next_eid += 1
            eid = edge_map[key]
            edge = mesh.edges[eid]
            e_ids.append(eid if edge.tail_index == a and edge.head_index == b else -eid)
        mesh.facets[fidx] = Facet(fidx, e_ids)

    mesh.build_connectivity_maps()
    mesh.build_facet_vertex_loops()
    return mesh


@pytest.mark.regression
def test_thetaB_optimize_enables_nontrivial_thetaB_under_stiff_boundary_penalty():
    """Regression: thetaB optimization samples reduced energy, not local closed form.

    Without thetaB optimization, the contact module's local update drives thetaB
    to a tiny value O(gamma/k). With optimization enabled, thetaB is chosen by
    (approximately) minimizing the total energy after tilt relaxation, allowing
    a nontrivial thetaB even when the boundary penalty k is large.
    """
    base_params = {
        # Tilt relaxation setup (fast for this tiny mesh).
        "tilt_solve_mode": "coupled",
        "tilt_solver": "gd",
        "tilt_step_size": 0.05,
        "tilt_inner_steps": 120,
        "tilt_tol": 1e-12,
        "tilt_thetaB_optimize_every": 1,
        "tilt_thetaB_optimize_delta": 0.05,
        "tilt_thetaB_optimize_inner_steps": 120,
        # θ_B module configuration.
        "rim_slope_match_disk_group": "disk",
        "tilt_thetaB_center": [0.0, 0.0, 0.0],
        "tilt_thetaB_strength_in": 1.0e3,
        "tilt_thetaB_contact_strength_in": 1.0,
        "tilt_thetaB_value": 0.0,
        # Ensure the reduced energy has a meaningful quadratic cost in tilts.
        "tilt_modulus_in": 50.0,
    }

    def run(thetaB_optimize: bool) -> float:
        mesh = _fan_disk_mesh(n_ring=16, radius=1.0)
        gp = GlobalParameters(dict(base_params))
        gp.set("tilt_thetaB_optimize", bool(thetaB_optimize))
        mesh.global_parameters = gp
        mesh.energy_modules = ["tilt_in", "tilt_thetaB_contact_in"]
        mesh.constraint_modules = []

        minim = Minimizer(
            mesh,
            gp,
            GradientDescent(),
            EnergyModuleManager(mesh.energy_modules),
            ConstraintModuleManager(mesh.constraint_modules),
            quiet=True,
            tol=1e-12,
        )
        minim.minimize(n_steps=1)
        return float(gp.get("tilt_thetaB_value") or 0.0)

    thetaB_local = run(thetaB_optimize=False)
    thetaB_opt = run(thetaB_optimize=True)

    assert abs(thetaB_local) < 5.0e-3
    assert abs(thetaB_opt) > 1.0e-2
