import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import load_data, parse_geometry


def test_benchmark_dented_cube_mesh_parses_and_tags_present():
    mesh = parse_geometry(load_data("meshes/bench_dented_cube.json"))

    tagged_facets = [
        f
        for f in mesh.facets.values()
        if f.options.get("dent_region") and f.options.get("no_refine")
    ]
    assert tagged_facets, (
        "Expected at least one dent_region/no_refine facet after parsing"
    )


def test_fixed_constraint_is_treated_as_flag():
    data = {
        "vertices": [
            [0.0, 0.0, 0.0, {"constraints": ["fixed"]}],
            [1.0, 0.0, 0.0],
        ],
        "edges": [
            [0, 1],
        ],
        "global_parameters": {},
    }
    mesh = parse_geometry(data)
    assert mesh.vertices[0].fixed is True
    assert "fixed" not in mesh.constraint_modules
    assert "constraints" not in mesh.vertices[0].options


def test_fixed_edge_freezes_endpoints():
    data = {
        "vertices": [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        "edges": [
            [0, 1, {"fixed": True}],
        ],
    }
    mesh = parse_geometry(data)
    assert mesh.edges[1].fixed is True
    assert mesh.vertices[0].fixed is True
    assert mesh.vertices[1].fixed is True


def test_benchmark_two_disks_sphere_mesh_parses_and_tags_present():
    mesh = parse_geometry(load_data("meshes/bench_two_disks_sphere.json"))

    disk_facets = [
        f
        for f in mesh.facets.values()
        if f.options.get("no_refine")
        and f.options.get("disk_patch") in {"top", "bottom"}
    ]
    assert disk_facets, "Expected disk_patch facets tagged as top/bottom with no_refine"

    circle_edges = [
        e
        for e in mesh.edges.values()
        if "pin_to_circle" in (e.options.get("constraints") or [])
    ]
    assert circle_edges, "Expected at least one edge tagged with pin_to_circle"

    plane_vertices = [
        v
        for v in mesh.vertices.values()
        if "pin_to_plane" in (v.options.get("constraints") or [])
    ]
    assert plane_vertices, "Expected at least one vertex tagged with pin_to_plane"


def test_refinement_preserves_closed_manifold_for_two_disks_sphere():
    from runtime.refinement import refine_triangle_mesh

    mesh = parse_geometry(load_data("meshes/bench_two_disks_sphere.json"))
    refined = refine_triangle_mesh(mesh)
    refined.build_connectivity_maps()

    edge_facet_counts = [len(fs) for fs in refined.edge_to_facets.values()]
    assert min(edge_facet_counts) == 2
    assert max(edge_facet_counts) == 2
