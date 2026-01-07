import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.curvature import compute_angle_defects, compute_curvature_fields
from geometry.geom_io import load_data, parse_geometry
from runtime.refinement import refine_triangle_mesh


def _project_vertices_to_sphere(mesh, *, radius: float = 1.0) -> None:
    for vertex in mesh.vertices.values():
        pos = np.asarray(vertex.position, dtype=float)
        nrm = float(np.linalg.norm(pos))
        if nrm <= 0.0:
            continue
        vertex.position[:] = (float(radius) / nrm) * pos
    mesh.increment_version()


def _area_weighted_stats(
    values: np.ndarray, weights: np.ndarray
) -> tuple[float, float]:
    weights = np.asarray(weights, dtype=float)
    total = float(np.sum(weights))
    if total <= 0.0:
        return float("nan"), float("nan")
    w = weights / total
    mean = float(np.sum(values * w))
    return mean, float(total)


def _area_weighted_rmse(
    values: np.ndarray, weights: np.ndarray, *, target: float
) -> float:
    weights = np.asarray(weights, dtype=float)
    total = float(np.sum(weights))
    if total <= 0.0:
        return float("nan")
    w = weights / total
    return float(np.sqrt(np.sum(((values - float(target)) ** 2) * w)))


def test_sphere_curvature_converges_under_refinement():
    """Validate discrete curvature on an analytic sphere under refinement."""
    mesh = parse_geometry(
        load_data("benchmarks/inputs/bench_helfrich_sphere_match.json")
    )

    levels = []
    for _ in range(3):
        _project_vertices_to_sphere(mesh, radius=1.0)

        mesh.build_facet_vertex_loops()
        positions = mesh.positions_view()
        idx_map = mesh.vertex_index_to_row

        fields = compute_curvature_fields(mesh, positions, idx_map)
        area = float(mesh.compute_total_surface_area())
        defect_sum = float(np.sum(compute_angle_defects(mesh, positions, idx_map)))

        h_mean, _ = _area_weighted_stats(fields.mean_curvature, fields.mixed_area)
        k_mean, _ = _area_weighted_stats(fields.gaussian_curvature, fields.mixed_area)

        h_rmse = _area_weighted_rmse(
            fields.mean_curvature, fields.mixed_area, target=1.0
        )
        k_rmse = _area_weighted_rmse(
            fields.gaussian_curvature, fields.mixed_area, target=1.0
        )

        levels.append(
            {
                "n_vertices": int(len(mesh.vertices)),
                "area": area,
                "defect_sum": defect_sum,
                "H_mean": h_mean,
                "H_rmse": h_rmse,
                "K_mean": k_mean,
                "K_rmse": k_rmse,
            }
        )
        mesh = refine_triangle_mesh(mesh)

    target = 4.0 * math.pi
    for lev in levels:
        assert math.isclose(lev["defect_sum"], target, rel_tol=0.0, abs_tol=1e-9)
        assert abs(lev["H_mean"] - 1.0) < 5e-3

    # Geometry convergence (inscribed polyhedra approach 4π area from below).
    assert levels[0]["area"] < levels[1]["area"] < levels[2]["area"]
    assert (target - levels[2]["area"]) < 0.05 * target

    # Curvature consistency improves with refinement when projecting back to the sphere.
    assert levels[2]["K_rmse"] < levels[1]["K_rmse"] < levels[0]["K_rmse"]
    assert abs(levels[2]["K_mean"] - 1.0) < 0.03
    assert levels[2]["H_rmse"] < 1e-3
