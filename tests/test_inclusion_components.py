import numpy as np

from geometry.geom_io import load_data, parse_geometry
from modules.constraints.inclusion_components import collect_group_components


def test_one_disk_interface_resolves_to_one_component() -> None:
    mesh = parse_geometry(
        load_data(
            "tests/fixtures/"
            "kozlov_1disk_3d_free_disk_theory_parity_physical_edge_default.yaml"
        )
    )

    components = collect_group_components(mesh, group="disk")

    assert len(components) == 1
    assert components[0].rows.size == 12
    assert np.allclose(components[0].center, [0.0, 0.0, 0.0], atol=1.0e-12)


def test_two_hole_rim_resolves_to_two_local_components() -> None:
    mesh = parse_geometry(load_data("meshes/kozlov_two_holes.yaml"))

    components = collect_group_components(mesh, group="rim")

    assert len(components) == 2
    assert [component.vertex_ids.tolist() for component in components] == [
        [7, 8, 9, 10],
        [11, 12, 13, 14],
    ]
    assert np.allclose(components[0].center, [3.0, 3.0, 0.0])
    assert np.allclose(components[1].center, [9.0, 3.0, 0.0])


def test_component_cache_invalidates_on_topology_change() -> None:
    mesh = parse_geometry(load_data("meshes/kozlov_two_holes.yaml"))

    first = collect_group_components(mesh, group="rim")
    cached = collect_group_components(mesh, group="rim")
    assert cached is first

    mesh.increment_topology_version()
    refreshed = collect_group_components(mesh, group="rim")
    assert refreshed is not first
