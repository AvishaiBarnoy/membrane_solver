import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.parameters.resolver import ParameterResolver
from geometry.geom_io import parse_geometry


def _disk_fan_mesh(*, n: int = 8, radius: float = 1.0) -> dict:
    """Return a simple disk triangulation with a center + ring."""
    vertices: list[list] = []
    vertices.append([0.0, 0.0, 0.0, {"tilt_disk_target_group_out": "disk"}])
    for i in range(n):
        theta = 2.0 * np.pi * i / n
        vertices.append(
            [
                float(radius * np.cos(theta)),
                float(radius * np.sin(theta)),
                0.0,
                {"tilt_disk_target_group_out": "disk"},
            ]
        )

    def vid(k: int) -> int:
        return int(k)

    edges: list[list[int]] = []
    # Ring edges
    for i in range(n):
        edges.append([vid(1 + i), vid(1 + (i + 1) % n)])
    # Spokes
    for i in range(n):
        edges.append([vid(0), vid(1 + i)])

    edge_index_by_pair: dict[tuple[int, int], int] = {}
    for idx, (tail, head, *_rest) in enumerate(edges):
        edge_index_by_pair[(int(tail), int(head))] = int(idx)

    def edge_ref(tail: int, head: int) -> int | str:
        forward = edge_index_by_pair.get((int(tail), int(head)))
        if forward is not None:
            return forward
        reverse = edge_index_by_pair.get((int(head), int(tail)))
        if reverse is not None:
            return f"r{reverse}"
        raise KeyError(f"Missing edge for face: {tail}->{head}")

    faces: list[list] = []
    for i in range(n):
        v0 = vid(0)
        v1 = vid(1 + i)
        v2 = vid(1 + (i + 1) % n)
        faces.append([edge_ref(v0, v1), edge_ref(v1, v2), edge_ref(v2, v0)])

    return {
        "global_parameters": {
            "tilt_disk_target_group_out": "disk",
            "tilt_disk_target_strength_out": 10.0,
            "tilt_disk_target_theta_B_out": 1.0,
            "tilt_disk_target_lambda_out": 1.0,
            "tilt_disk_target_center_out": [0.0, 0.0, 0.0],
            "tilt_disk_target_normal_out": [0.0, 0.0, 1.0],
        },
        "energy_modules": [],
        "vertices": vertices,
        "edges": edges,
        "faces": faces,
        "instructions": [],
    }


def _bessel_i1_series(x: np.ndarray, n_terms: int = 30) -> np.ndarray:
    t = 0.5 * x
    t2 = t * t
    term = t.copy()
    out = term.copy()
    for k in range(1, n_terms):
        term *= t2 / (k * (k + 1))
        out += term
    return out


def _target_profile(
    r: np.ndarray, *, theta_b: float, lam: float, r_max: float
) -> np.ndarray:
    if lam <= 1e-12 or r_max <= 0.0:
        return theta_b * r / max(r_max, 1e-12)
    num = _bessel_i1_series(lam * r)
    den = _bessel_i1_series(np.array([lam * r_max], dtype=float))[0]
    return theta_b * num / den


def test_tilt_disk_target_out_zero_when_matching() -> None:
    from modules.energy import tilt_disk_target_out

    mesh = parse_geometry(_disk_fan_mesh())
    resolver = ParameterResolver(mesh.global_parameters)
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row
    grad_arr = np.zeros_like(positions)
    tilt_grad = np.zeros_like(positions)

    r = np.linalg.norm(positions[:, :2], axis=1)
    r_hat = np.zeros_like(positions)
    good = r > 1e-12
    r_hat[good, 0] = positions[good, 0] / r[good]
    r_hat[good, 1] = positions[good, 1] / r[good]
    r_max = float(np.max(r))
    theta = _target_profile(r, theta_b=1.0, lam=1.0, r_max=r_max)
    tilts_out = theta[:, None] * r_hat
    mesh.set_tilts_out_from_array(tilts_out)

    energy = tilt_disk_target_out.compute_energy_and_gradient_array(
        mesh,
        mesh.global_parameters,
        resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=grad_arr,
        tilts_out=mesh.tilts_out_view(),
        tilt_out_grad_arr=tilt_grad,
    )
    assert abs(float(energy)) < 1e-8
    assert float(np.linalg.norm(tilt_grad)) < 1e-6


def test_tilt_disk_target_out_penalizes_mismatch() -> None:
    from modules.energy import tilt_disk_target_out

    mesh = parse_geometry(_disk_fan_mesh())
    resolver = ParameterResolver(mesh.global_parameters)
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row
    grad_arr = np.zeros_like(positions)
    tilt_grad = np.zeros_like(positions)

    mesh.set_tilts_out_from_array(np.zeros_like(positions))

    energy = tilt_disk_target_out.compute_energy_and_gradient_array(
        mesh,
        mesh.global_parameters,
        resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=grad_arr,
        tilts_out=mesh.tilts_out_view(),
        tilt_out_grad_arr=tilt_grad,
    )
    assert float(energy) > 1e-4
    assert float(np.linalg.norm(tilt_grad)) > 1e-4
