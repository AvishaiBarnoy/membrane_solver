import numpy as np
import pytest

from core.parameters.global_parameters import GlobalParameters
from core.parameters.resolver import ParameterResolver
from geometry.entities import Mesh, Vertex
from modules.energy import tilt_disk_contact_in


def _make_ring_mesh(*, n: int, radius: float, group: str) -> Mesh:
    mesh = Mesh()
    mesh.global_parameters = GlobalParameters(
        {
            "tilt_disk_contact_group_in": group,
            "tilt_disk_contact_strength_in": 1.0,
            "tilt_disk_contact_center": [0.0, 0.0, 0.0],
            "tilt_disk_contact_normal": [0.0, 0.0, 1.0],
        }
    )
    for i in range(n):
        ang = 2.0 * np.pi * float(i) / float(n)
        x = radius * float(np.cos(ang))
        y = radius * float(np.sin(ang))
        mesh.vertices[i] = Vertex(
            index=i,
            position=np.array([x, y, 0.0], dtype=float),
            options={"tilt_disk_contact_group": group},
        )
    mesh.increment_version()
    return mesh


def test_tilt_disk_contact_energy_array_matches_gradient_path() -> None:
    mesh = _make_ring_mesh(n=12, radius=1.0, group="disk")
    resolver = ParameterResolver(mesh.global_parameters)
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row

    r = np.linalg.norm(positions[:, :2], axis=1)
    r_hat = np.zeros_like(positions)
    good = r > 1e-12
    r_hat[good, 0] = positions[good, 0] / r[good]
    r_hat[good, 1] = positions[good, 1] / r[good]
    mesh.set_tilts_in_from_array(r_hat)

    grad_arr = np.zeros_like(positions)
    tilt_grad = np.zeros_like(positions)

    e_grad = tilt_disk_contact_in.compute_energy_and_gradient_array(
        mesh,
        mesh.global_parameters,
        resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=grad_arr,
        tilts_in=mesh.tilts_in_view(),
        tilt_in_grad_arr=tilt_grad,
    )
    e_only = tilt_disk_contact_in.compute_energy_array(
        mesh,
        mesh.global_parameters,
        resolver,
        positions=positions,
        index_map=index_map,
        tilts_in=mesh.tilts_in_view(),
    )
    assert float(e_only) == pytest.approx(float(e_grad), rel=1e-12, abs=1e-12)
