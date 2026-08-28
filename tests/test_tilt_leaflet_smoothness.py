import importlib

import numpy as np
import pytest

from core.parameters.global_parameters import GlobalParameters
from core.parameters.resolver import ParameterResolver
from tests.sample_meshes import single_triangle_mesh as _build_single_triangle_mesh

LEAFLET_CASES = {
    "in": {
        "module": "modules.energy.tilt_smoothness_in",
        "field": "tilt_in",
        "touch": "touch_tilts_in",
    },
    "out": {
        "module": "modules.energy.tilt_smoothness_out",
        "field": "tilt_out",
        "touch": "touch_tilts_out",
    },
}


@pytest.mark.parametrize("leaflet", ["in", "out"])
def test_tilt_leaflet_smoothness_constant_field_has_zero_energy(
    leaflet: str,
) -> None:
    case = LEAFLET_CASES[leaflet]
    module = importlib.import_module(case["module"])
    mesh = _build_single_triangle_mesh()

    for v in mesh.vertices.values():
        setattr(v, case["field"], np.array([1.0, -0.5, 0.0], dtype=float))
    getattr(mesh, case["touch"])()

    gp = GlobalParameters({"bending_modulus": 1.0})
    resolver = ParameterResolver(gp)

    energy, shape_grad, tilt_grad = module.compute_energy_and_gradient(
        mesh, gp, resolver
    )

    assert float(energy) == pytest.approx(0.0, rel=0.0, abs=1e-12)
    assert all(np.allclose(g, 0.0, atol=1e-12) for g in shape_grad.values())
    assert all(np.allclose(g, 0.0, atol=1e-12) for g in tilt_grad.values())
