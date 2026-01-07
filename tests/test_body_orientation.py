import numpy as np
import pytest

from core.exceptions import BodyOrientationError
from geometry.entities import Body, Edge, Facet, Mesh, Vertex


def _two_triangle_body_mesh(*, inconsistent: bool) -> Mesh:
    """Return a minimal open body made of two triangles sharing one edge."""
    mesh = Mesh()

    mesh.vertices[0] = Vertex(0, np.array([0.0, 0.0, 0.0]))
    mesh.vertices[1] = Vertex(1, np.array([1.0, 0.0, 0.0]))
    mesh.vertices[2] = Vertex(2, np.array([0.0, 1.0, 0.0]))
    mesh.vertices[3] = Vertex(3, np.array([1.0, 1.0, 0.0]))

    # Edge indices are 1-based in memory.
    mesh.edges[1] = Edge(1, 0, 1)  # shared edge
    mesh.edges[2] = Edge(2, 1, 2)
    mesh.edges[3] = Edge(3, 2, 0)
    mesh.edges[4] = Edge(4, 1, 3)
    mesh.edges[5] = Edge(5, 3, 0)

    # Triangle A: 0 -> 1 -> 2 -> 0
    mesh.facets[0] = Facet(0, [1, 2, 3])

    # Triangle B shares edge (0, 1). When oriented consistently, it traverses
    # the shared edge in the opposite direction (signed edge index -1).
    if inconsistent:
        # 0 -> 1 -> 3 -> 0, shares edge +1 with Triangle A (inconsistent)
        mesh.facets[1] = Facet(1, [1, 4, 5])
    else:
        # 1 -> 0 -> 3 -> 1, shares edge -1 with Triangle A (consistent)
        mesh.facets[1] = Facet(1, [-1, -5, -4])

    mesh.bodies[0] = Body(0, [0, 1])
    return mesh


def _tetra_body_mesh(*, invert: bool) -> Mesh:
    """Return a closed tetrahedron body, optionally inverted."""
    mesh = Mesh()

    mesh.vertices[0] = Vertex(0, np.array([0.0, 0.0, 0.0]))
    mesh.vertices[1] = Vertex(1, np.array([1.0, 0.0, 0.0]))
    mesh.vertices[2] = Vertex(2, np.array([0.0, 1.0, 0.0]))
    mesh.vertices[3] = Vertex(3, np.array([0.0, 0.0, 1.0]))

    mesh.edges[1] = Edge(1, 0, 1)
    mesh.edges[2] = Edge(2, 1, 2)
    mesh.edges[3] = Edge(3, 2, 0)
    mesh.edges[4] = Edge(4, 0, 3)
    mesh.edges[5] = Edge(5, 1, 3)
    mesh.edges[6] = Edge(6, 2, 3)

    mesh.facets[0] = Facet(0, [1, 2, 3])  # 0->1->2->0
    mesh.facets[1] = Facet(1, [1, 5, -4])  # 0->1->3->0
    mesh.facets[2] = Facet(2, [-3, 6, -4])  # 0->2->3->0
    mesh.facets[3] = Facet(3, [2, 6, -5])  # 1->2->3->1

    mesh.bodies[0] = Body(0, [0, 1, 2, 3])

    vol = mesh.bodies[0].compute_volume(mesh)
    should_flip = (vol > 0 and invert) or (vol < 0 and not invert)
    if should_flip:
        for facet in mesh.facets.values():
            facet.edge_indices = [-int(ei) for ei in reversed(facet.edge_indices)]
        mesh.increment_version()
    return mesh


def test_validate_body_orientation_accepts_consistent_patch():
    mesh = _two_triangle_body_mesh(inconsistent=False)
    assert mesh.validate_body_orientation() is True


def test_validate_body_orientation_rejects_inconsistent_patch():
    mesh = _two_triangle_body_mesh(inconsistent=True)
    with pytest.raises(BodyOrientationError):
        mesh.validate_body_orientation()


def test_orient_body_facets_repairs_inconsistent_patch():
    mesh = _two_triangle_body_mesh(inconsistent=True)
    with pytest.raises(BodyOrientationError):
        mesh.validate_body_orientation()

    flipped = mesh.orient_body_facets(0)
    assert flipped > 0
    assert mesh.validate_body_orientation() is True


def test_validate_body_outwardness_skips_open_body():
    mesh = _two_triangle_body_mesh(inconsistent=False)
    assert mesh.validate_body_outwardness() is True


def test_validate_body_outwardness_rejects_inverted_closed_body():
    mesh = _tetra_body_mesh(invert=True)
    with pytest.raises(BodyOrientationError):
        mesh.validate_body_outwardness()


def test_orient_body_outward_repairs_inverted_closed_body():
    mesh = _tetra_body_mesh(invert=True)
    flipped = mesh.orient_body_outward(0)
    assert flipped == len(mesh.facets)
    assert mesh.validate_body_outwardness() is True
