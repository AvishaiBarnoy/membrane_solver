import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.entities import Mesh, Vertex, Edge, Facet, Body
from modules.constraints.body_area import enforce_constraint as enforce_body_area


def _build_square_body(scale: float = 1.0) -> Mesh:
    """Construct a simple square patch as a single body.

    The square lies in the z=0 plane with side length ``scale``. It is
    represented as one four‑edge facet and a single body containing that
    facet, so the body surface area is the facet area.
    """
    v0 = Vertex(0, np.array([0.0, 0.0, 0.0]))
    v1 = Vertex(1, np.array([scale, 0.0, 0.0]))
    v2 = Vertex(2, np.array([scale, scale, 0.0]))
    v3 = Vertex(3, np.array([0.0, scale, 0.0]))

    e0 = Edge(1, v0.index, v1.index)
    e1 = Edge(2, v1.index, v2.index)
    e2 = Edge(3, v2.index, v3.index)
    e3 = Edge(4, v3.index, v0.index)

    facet = Facet(0, [e0.index, e1.index, e2.index, e3.index])
    body = Body(0, [facet.index])

    mesh = Mesh()
    mesh.vertices = {v.index: v for v in (v0, v1, v2, v3)}
    mesh.edges = {e.index: e for e in (e0, e1, e2, e3)}
    mesh.facets = {facet.index: facet}
    mesh.bodies = {body.index: body}

    return mesh


def test_body_area_constraint_noop_at_target():
    """If the body already has the target area, the constraint does nothing."""
    mesh = _build_square_body(scale=1.0)
    body = next(iter(mesh.bodies.values()))

    # Exact target equal to current area
    area0 = body.compute_surface_area(mesh)
    body.options["target_area"] = area0

    # Snapshot vertex positions
    positions_before = {vidx: v.position.copy() for vidx, v in mesh.vertices.items()}

    enforce_body_area(mesh)

    # Area is unchanged within numerical tolerance
    area_after = body.compute_surface_area(mesh)
    assert np.isclose(area_after, area0, rtol=1e-12, atol=1e-12)

    # Vertex positions are unchanged (no spurious motion)
    for vidx, v in mesh.vertices.items():
        assert np.allclose(v.position, positions_before[vidx])


def test_body_area_constraint_converges_to_target():
    """Body area constraint should drive the area back to the target."""
    # Start from a larger square so the area deviates from the target.
    mesh = _build_square_body(scale=1.1)
    body = next(iter(mesh.bodies.values()))

    target_area = 1.0  # target area of unit square
    body.options["target_area"] = target_area

    # Initial area should differ noticeably from the target.
    area_before = body.compute_surface_area(mesh)
    assert not np.isclose(area_before, target_area, rtol=1e-6)

    # Apply the constraint a few times to allow the iterative scheme to
    # converge even from a moderately perturbed configuration.
    for _ in range(5):
        enforce_body_area(mesh)

    area_after = body.compute_surface_area(mesh)
    # The constraint uses a first‑order Lagrange update, so in general we
    # expect it to drive the area very close to the target, but not to
    # machine precision in a few iterations. A relative error of 1e‑3 is
    # more than sufficient for this unit test.
    assert abs(area_after - target_area) < 1e-3
