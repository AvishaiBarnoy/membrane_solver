import numpy as np

from commands.executor import execute_command_line
from tests.minimizer_test_utils import build_command_context as _build_context
from tests.sample_meshes import annulus_source_mesh_data


def _annulus_source_mesh(*, n: int = 10) -> dict:
    return annulus_source_mesh_data(
        n=n,
        global_parameters={
            "tilt_rim_source_group_in": "inner",
            "tilt_rim_source_strength_in": 1.0,
            "tilt_rim_source_center": [0.0, 0.0, 0.0],
        },
        energy_modules=["tilt_rim_source_in"],
    )


def _set_radial_tilt_in(mesh) -> None:
    positions = mesh.positions_view()
    r = positions.copy()
    r[:, 2] = 0.0
    rn = np.linalg.norm(r, axis=1)
    tilts = np.zeros_like(positions)
    good = rn > 1e-12
    tilts[good] = r[good] / rn[good][:, None]
    mesh.set_tilts_in_from_array(tilts)


def test_energy_command_reports_sources_and_reference(capsys):
    ctx = _build_context(_annulus_source_mesh())
    _set_radial_tilt_in(ctx.mesh)

    execute_command_line(ctx, "energy")
    out = capsys.readouterr().out
    assert "external work (sources)" in out
    assert "tilt_rim_source_in" in out

    execute_command_line(ctx, "energy ref")
    _ = capsys.readouterr().out

    ctx.mesh.set_tilts_in_from_array(np.zeros_like(ctx.mesh.positions_view()))
    execute_command_line(ctx, "energy")
    out = capsys.readouterr().out
    assert "Δtotal vs ref" in out
