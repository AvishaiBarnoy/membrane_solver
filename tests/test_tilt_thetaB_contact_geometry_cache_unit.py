import numpy as np

from core.parameters.resolver import ParameterResolver
from geometry.geom_io import load_data, parse_geometry
from modules.energy import tilt_thetaB_contact_in


def _fixture_path(name: str) -> str:
    import os

    here = os.path.dirname(__file__)
    return os.path.join(here, "fixtures", name)


def test_thetaB_contact_geometry_cache_reuses_and_invalidates(monkeypatch) -> None:
    mesh = parse_geometry(
        load_data(_fixture_path("kozlov_free_disk_coarse_refinable.yaml"))
    )
    gp = mesh.global_parameters
    resolver = ParameterResolver(gp)
    gp.set("tilt_thetaB_value", 0.04)
    gp.set("tilt_thetaB_contact_penalty_mode", "off")

    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row
    tilts_in = np.zeros_like(positions)

    real_order = tilt_thetaB_contact_in._order_by_angle
    calls = {"n": 0}

    def _counted_order(*args, **kwargs):
        calls["n"] += 1
        return real_order(*args, **kwargs)

    monkeypatch.setattr(tilt_thetaB_contact_in, "_order_by_angle", _counted_order)

    e1 = tilt_thetaB_contact_in.compute_energy_array(
        mesh,
        gp,
        resolver,
        positions=positions,
        index_map=index_map,
        tilts_in=tilts_in,
    )
    e2 = tilt_thetaB_contact_in.compute_energy_array(
        mesh,
        gp,
        resolver,
        positions=positions,
        index_map=index_map,
        tilts_in=tilts_in,
    )
    assert calls["n"] == 1
    assert float(e1) == float(e2)

    mesh.increment_version()
    e3 = tilt_thetaB_contact_in.compute_energy_array(
        mesh,
        gp,
        resolver,
        positions=positions,
        index_map=index_map,
        tilts_in=tilts_in,
    )
    assert calls["n"] == 2
    assert float(e2) == float(e3)
