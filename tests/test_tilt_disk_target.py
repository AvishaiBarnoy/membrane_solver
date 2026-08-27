import importlib

import numpy as np
import pytest

from core.parameters.resolver import ParameterResolver
from geometry.geom_io import parse_geometry

LEAFLETS = ("in", "out")


def _disk_fan_mesh(*, leaflet: str, n: int = 8, radius: float = 1.0) -> dict:
    """Return a simple disk triangulation with a center and ring."""
    group_key = f"tilt_disk_target_group_{leaflet}"
    vertices = [[0.0, 0.0, 0.0, {group_key: "disk"}]]
    for index in range(n):
        theta = 2.0 * np.pi * index / n
        vertices.append(
            [
                float(radius * np.cos(theta)),
                float(radius * np.sin(theta)),
                0.0,
                {group_key: "disk"},
            ]
        )

    edges: list[list[int]] = []
    for index in range(n):
        edges.append([1 + index, 1 + (index + 1) % n])
    for index in range(n):
        edges.append([0, 1 + index])

    edge_index_by_pair = {
        (int(tail), int(head)): int(index)
        for index, (tail, head, *_rest) in enumerate(edges)
    }

    def edge_ref(tail: int, head: int) -> int | str:
        forward = edge_index_by_pair.get((int(tail), int(head)))
        if forward is not None:
            return forward
        reverse = edge_index_by_pair.get((int(head), int(tail)))
        if reverse is not None:
            return f"r{reverse}"
        raise KeyError(f"Missing edge for face: {tail}->{head}")

    faces = []
    for index in range(n):
        center = 0
        first = 1 + index
        second = 1 + (index + 1) % n
        faces.append(
            [
                edge_ref(center, first),
                edge_ref(first, second),
                edge_ref(second, center),
            ]
        )

    profile_suffix = "" if leaflet == "in" else "_out"
    return {
        "global_parameters": {
            group_key: "disk",
            f"tilt_disk_target_strength_{leaflet}": 10.0,
            f"tilt_disk_target_theta_B{profile_suffix}": 1.0,
            f"tilt_disk_target_lambda{profile_suffix}": 1.0,
            f"tilt_disk_target_center{profile_suffix}": [0.0, 0.0, 0.0],
            f"tilt_disk_target_normal{profile_suffix}": [0.0, 0.0, 1.0],
        },
        "energy_modules": [],
        "vertices": vertices,
        "edges": edges,
        "faces": faces,
        "instructions": [],
    }


def _bessel_i1_series(x: np.ndarray, n_terms: int = 30) -> np.ndarray:
    half_x = 0.5 * x
    half_x_squared = half_x * half_x
    term = half_x.copy()
    result = term.copy()
    for index in range(1, n_terms):
        term *= half_x_squared / (index * (index + 1))
        result += term
    return result


def _target_profile(
    radii: np.ndarray, *, theta_b: float, lam: float, r_max: float
) -> np.ndarray:
    if lam <= 1e-12 or r_max <= 0.0:
        return theta_b * radii / max(r_max, 1e-12)
    numerator = _bessel_i1_series(lam * radii)
    denominator = _bessel_i1_series(np.array([lam * r_max], dtype=float))[0]
    return theta_b * numerator / denominator


def _module(leaflet: str):
    return importlib.import_module(f"modules.energy.tilt_disk_target_{leaflet}")


def _tilts(mesh, leaflet: str) -> np.ndarray:
    return getattr(mesh, f"tilts_{leaflet}_view")()


def _set_tilts(mesh, leaflet: str, values: np.ndarray) -> None:
    getattr(mesh, f"set_tilts_{leaflet}_from_array")(values)


def _gradient_kwargs(mesh, leaflet: str, tilt_grad: np.ndarray) -> dict:
    return {
        f"tilts_{leaflet}": _tilts(mesh, leaflet),
        f"tilt_{leaflet}_grad_arr": tilt_grad,
    }


@pytest.mark.parametrize("leaflet", LEAFLETS)
def test_tilt_disk_target_zero_when_matching(leaflet: str) -> None:
    module = _module(leaflet)
    mesh = parse_geometry(_disk_fan_mesh(leaflet=leaflet))
    resolver = ParameterResolver(mesh.global_parameters)
    positions = mesh.positions_view()
    grad_arr = np.zeros_like(positions)
    tilt_grad = np.zeros_like(positions)

    radii = np.linalg.norm(positions[:, :2], axis=1)
    radial = np.zeros_like(positions)
    good = radii > 1e-12
    radial[good, 0] = positions[good, 0] / radii[good]
    radial[good, 1] = positions[good, 1] / radii[good]
    theta = _target_profile(radii, theta_b=1.0, lam=1.0, r_max=float(np.max(radii)))
    _set_tilts(mesh, leaflet, theta[:, None] * radial)

    energy = module.compute_energy_and_gradient_array(
        mesh,
        mesh.global_parameters,
        resolver,
        positions=positions,
        index_map=mesh.vertex_index_to_row,
        grad_arr=grad_arr,
        **_gradient_kwargs(mesh, leaflet, tilt_grad),
    )
    assert abs(float(energy)) < 1e-8
    assert float(np.linalg.norm(tilt_grad)) < 1e-6


@pytest.mark.parametrize("leaflet", LEAFLETS)
def test_tilt_disk_target_penalizes_mismatch(leaflet: str) -> None:
    module = _module(leaflet)
    mesh = parse_geometry(_disk_fan_mesh(leaflet=leaflet))
    resolver = ParameterResolver(mesh.global_parameters)
    positions = mesh.positions_view()
    grad_arr = np.zeros_like(positions)
    tilt_grad = np.zeros_like(positions)

    _set_tilts(mesh, leaflet, np.zeros_like(positions))
    energy = module.compute_energy_and_gradient_array(
        mesh,
        mesh.global_parameters,
        resolver,
        positions=positions,
        index_map=mesh.vertex_index_to_row,
        grad_arr=grad_arr,
        **_gradient_kwargs(mesh, leaflet, tilt_grad),
    )
    assert float(energy) > 1e-4
    assert float(np.linalg.norm(tilt_grad)) > 1e-4


@pytest.mark.parametrize("leaflet", LEAFLETS)
def test_tilt_disk_target_energy_array_matches_gradient_path(leaflet: str) -> None:
    module = _module(leaflet)
    mesh = parse_geometry(_disk_fan_mesh(leaflet=leaflet))
    resolver = ParameterResolver(mesh.global_parameters)
    positions = mesh.positions_view()
    tilts = _tilts(mesh, leaflet)
    tilt_grad = np.zeros_like(positions)

    energy_with_gradient = module.compute_energy_and_gradient_array(
        mesh,
        mesh.global_parameters,
        resolver,
        positions=positions,
        index_map=mesh.vertex_index_to_row,
        grad_arr=np.zeros_like(positions),
        **_gradient_kwargs(mesh, leaflet, tilt_grad),
    )
    energy_only = module.compute_energy_array(
        mesh,
        mesh.global_parameters,
        resolver,
        positions=positions,
        index_map=mesh.vertex_index_to_row,
        **{f"tilts_{leaflet}": tilts},
    )
    assert float(energy_only) == pytest.approx(
        float(energy_with_gradient), rel=1e-12, abs=1e-12
    )
