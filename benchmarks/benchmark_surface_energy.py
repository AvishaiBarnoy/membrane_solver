#!/usr/bin/env python3
"""Compare timing between baseline and optimized surface energy calculations."""

import time
from collections import defaultdict
from typing import Dict, Tuple

import numpy as np

from geometry.geom_io import load_data, parse_geometry
from modules.energy import surface
from parameters.global_parameters import GlobalParameters
from parameters.resolver import ParameterResolver


def compute_energy_and_gradient_baseline(
    mesh, global_params: GlobalParameters, param_resolver: ParameterResolver, *, compute_gradient: bool = True
) -> Tuple[float, Dict[int, np.ndarray]]:
    """Original loop-based implementation for benchmarking."""
    E = 0.0
    grad = defaultdict(lambda: np.zeros(3)) if compute_gradient else None

    for facet in mesh.facets.values():
        surface_tension = param_resolver.get(facet, "surface_tension")
        if surface_tension is None:
            surface_tension = global_params.get("surface_tension")

        area = facet.compute_area(mesh)
        E += surface_tension * area

        if compute_gradient:
            area_gradient = facet.compute_area_gradient(mesh)
            for vertex_index, gradient_vector in area_gradient.items():
                grad[vertex_index] += surface_tension * gradient_vector

    if compute_gradient:
        return E, dict(grad)
    return E, {}


def benchmark(func, mesh, global_params, param_resolver, *, iterations: int = 1000) -> float:
    """Return the time (in seconds) required to run ``func`` ``iterations`` times."""
    start = time.perf_counter()
    for _ in range(iterations):
        func(mesh, global_params, param_resolver)
    return time.perf_counter() - start


if __name__ == "__main__":
    mesh = parse_geometry(load_data("meshes/cube.json"))
    global_params = GlobalParameters()
    resolver = ParameterResolver(global_params)

    iterations = 1000

    baseline_time = benchmark(
        compute_energy_and_gradient_baseline, mesh, global_params, resolver, iterations=iterations
    )
    optimized_time = benchmark(
        surface.compute_energy_and_gradient, mesh, global_params, resolver, iterations=iterations
    )

    print(f"Baseline time:  {baseline_time:.4f}s")
    print(f"Optimized time: {optimized_time:.4f}s")
    if optimized_time:
        print(f"Speedup:       {baseline_time/optimized_time:.2f}x")
