import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.cache_writes import (
    store_barycentric_vertex_areas_cache,
    store_p1_triangle_grad_cache,
    store_triangle_area_normals_cache,
    store_vertex_normals_cache,
)


def _mesh_stub():
    return SimpleNamespace(
        _version=7,
        _facet_loops_version=11,
        _cached_tri_areas=None,
        _cached_tri_normals=None,
        _cached_tri_areas_version=-1,
        _cached_barycentric_vertex_areas=None,
        _cached_barycentric_vertex_areas_version=-1,
        _cached_barycentric_vertex_areas_rows_version=-1,
        _cached_vertex_normals=None,
        _cached_vertex_normals_version=-1,
        _cached_vertex_normals_loops_version=-1,
        _cached_p1_tri_areas=None,
        _cached_p1_tri_g0=None,
        _cached_p1_tri_g1=None,
        _cached_p1_tri_g2=None,
        _cached_p1_tri_grads_version=-1,
        _cached_p1_tri_grads_rows_version=-1,
    )


def test_store_triangle_area_normals_cache_writes_expected_fields():
    mesh = _mesh_stub()
    areas = np.array([1.0, 2.0], dtype=float)
    normals = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
    store_triangle_area_normals_cache(mesh, areas=areas, normals=normals)
    assert mesh._cached_tri_areas is areas
    assert mesh._cached_tri_normals is normals
    assert mesh._cached_tri_areas_version == mesh._version


def test_store_barycentric_and_vertex_normal_cache_writes_versions():
    mesh = _mesh_stub()
    bary = np.array([0.1, 0.2, 0.3], dtype=float)
    vnorm = np.zeros((3, 3), dtype=float)
    store_barycentric_vertex_areas_cache(mesh, vertex_areas=bary)
    store_vertex_normals_cache(mesh, normals=vnorm)
    assert mesh._cached_barycentric_vertex_areas is bary
    assert mesh._cached_barycentric_vertex_areas_version == mesh._version
    assert (
        mesh._cached_barycentric_vertex_areas_rows_version == mesh._facet_loops_version
    )
    assert mesh._cached_vertex_normals is vnorm
    assert mesh._cached_vertex_normals_version == mesh._version
    assert mesh._cached_vertex_normals_loops_version == mesh._facet_loops_version


def test_store_p1_triangle_grad_cache_writes_all_arrays_and_versions():
    mesh = _mesh_stub()
    area = np.array([1.0], dtype=float)
    g = np.array([[1.0, 2.0, 3.0]], dtype=float)
    store_p1_triangle_grad_cache(mesh, area=area, g0=g, g1=g, g2=g)
    assert mesh._cached_p1_tri_areas is area
    assert mesh._cached_p1_tri_g0 is g
    assert mesh._cached_p1_tri_g1 is g
    assert mesh._cached_p1_tri_g2 is g
    assert mesh._cached_p1_tri_grads_version == mesh._version
    assert mesh._cached_p1_tri_grads_rows_version == mesh._facet_loops_version
