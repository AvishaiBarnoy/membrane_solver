import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.parameters.global_parameters import GlobalParameters  # noqa: E402
from geometry.entities import Edge, Facet, Mesh, Vertex  # noqa: E402
from runtime.interface_validation import validate_disk_interface_topology  # noqa: E402


def _fan_mesh(*, straddles: bool) -> Mesh:
    """Return a tiny mesh with a tagged 'disk boundary' ring.

    If straddles is False, the ring only has disk-side triangles.
    If straddles is True, the ring has both disk-side and rim-side triangles.
    """
    mesh = Mesh()
    mesh.global_parameters = GlobalParameters(
        {
            "rim_slope_match_disk_group": "disk",
            "disk_interface_validate": True,
        }
    )

    # Disk center and a 3-vertex boundary ring.
    mesh.vertices[0] = Vertex(0, np.array([0.0, 0.0, 0.0]), options={"preset": "disk"})
    ring = [
        (1, np.array([1.0, 0.0, 0.0])),
        (2, np.array([0.0, 1.0, 0.0])),
        (3, np.array([-1.0, 0.0, 0.0])),
    ]
    for vid, p in ring:
        mesh.vertices[vid] = Vertex(
            vid,
            p,
            options={"preset": "disk", "rim_slope_match_group": "disk"},
        )

    # Optional rim-side apex sharing the ring.
    if straddles:
        mesh.vertices[10] = Vertex(
            10, np.array([0.0, 0.0, 0.0]), options={"preset": "rim"}
        )

    tris: list[tuple[int, int, int]] = [
        (0, 1, 2),
        (0, 2, 3),
        (0, 3, 1),
    ]
    if straddles:
        tris.extend([(10, 1, 2), (10, 2, 3), (10, 3, 1)])

    # Build edges/facets in Evolver-style representation.
    edge_map: dict[tuple[int, int], int] = {}
    next_eid = 1
    for fid, tri in enumerate(tris, start=1):
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
        mesh.facets[fid] = Facet(fid, e_ids)

    mesh.build_connectivity_maps()
    mesh.build_facet_vertex_loops()
    return mesh


@pytest.mark.regression
def test_disk_interface_validator_fails_for_internal_disk_ring() -> None:
    mesh = _fan_mesh(straddles=False)
    with pytest.raises(ValueError, match="Disk interface topology invalid"):
        validate_disk_interface_topology(mesh, mesh.global_parameters)


@pytest.mark.regression
def test_disk_interface_validator_passes_for_straddling_ring() -> None:
    mesh = _fan_mesh(straddles=True)
    validate_disk_interface_topology(mesh, mesh.global_parameters)


@pytest.mark.regression
def test_disk_interface_validator_fails_when_rim_group_equals_disk_group() -> None:
    mesh = _fan_mesh(straddles=True)
    mesh.global_parameters.set("rim_slope_match_group", "disk")
    with pytest.raises(ValueError, match="rim_slope_match_group matches"):
        validate_disk_interface_topology(mesh, mesh.global_parameters)


@pytest.mark.regression
def test_disk_interface_validator_fails_on_rim_outer_count_mismatch() -> None:
    mesh = _fan_mesh(straddles=True)
    # Tag only two ring vertices as rim, and none as outer.
    mesh.global_parameters.set("rim_slope_match_group", "rim")
    mesh.global_parameters.set("rim_slope_match_outer_group", "outer")
    mesh.vertices[1].options["rim_slope_match_group"] = "rim"
    mesh.vertices[2].options["rim_slope_match_group"] = "rim"
    mesh.vertices[3].options["rim_slope_match_group"] = "outer"
    with pytest.raises(ValueError, match="rim_slope_match_group.*vertex counts"):
        validate_disk_interface_topology(mesh, mesh.global_parameters)
