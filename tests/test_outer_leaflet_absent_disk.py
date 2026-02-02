import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import load_data, parse_geometry
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.gradient_descent import GradientDescent


def _mesh_path() -> str:
    return os.path.join(
        os.path.dirname(__file__),
        "..",
        "meshes",
        "caveolin",
        "kozlov_1disk_3d_tensionless_single_leaflet_profile_hard_rim_R12_free_disk.yaml",
    )


def _build_minimizer(mesh) -> Minimizer:
    return Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
    )


def _find_row_by_preset(mesh, preset: str, *, require_free: bool = False) -> int:
    mesh.build_position_cache()
    for row, vid in enumerate(mesh.vertex_ids):
        v = mesh.vertices[int(vid)]
        opts = getattr(v, "options", None) or {}
        if opts.get("preset") != preset:
            continue
        if require_free:
            if getattr(v, "fixed", False):
                continue
            if opts.get("tilt_fixed_out", False):
                continue
        return int(row)
    raise AssertionError(f"No vertex found with preset={preset!r}")


def test_outer_leaflet_absent_on_disk_masks_out_energies() -> None:
    mesh = parse_geometry(load_data(_mesh_path()))
    minim = _build_minimizer(mesh)
    gp = mesh.global_parameters

    # Avoid any tilt relaxation during this test; we want deterministic
    # sensitivity checks to explicit tilt_out edits.
    gp.set("tilt_solve_mode", "fixed")

    # The Kozlov free-disk YAML may enable outer-leaflet absence by default for
    # theory parity. For this regression test we first explicitly disable it so
    # we can demonstrate that disk tilt_out affects energies when the leaflet is
    # present, and then re-enable it to confirm masking works.
    gp.set("leaflet_out_absent_presets", [])

    disk_row = _find_row_by_preset(mesh, "disk", require_free=True)
    rim_row = _find_row_by_preset(mesh, "rim", require_free=False)

    # Baseline energies.
    baseline = minim.compute_energy_breakdown()

    # Perturb disk tilt_out and confirm it affects outer-leaflet energies by default.
    tout = mesh.tilts_out_view().copy(order="F")
    tout[disk_row] = np.array([0.25, 0.0, 0.0], dtype=float)
    mesh.set_tilts_out_from_array(tout)
    perturbed = minim.compute_energy_breakdown()

    assert perturbed["tilt_out"] != baseline["tilt_out"]
    assert perturbed["bending_tilt_out"] != baseline["bending_tilt_out"]

    # Enable absence of the outer leaflet on the disk. The same perturbation
    # should no longer affect outer-leaflet energies.
    gp.set("leaflet_out_absent_presets", ["disk"])
    masked_baseline = minim.compute_energy_breakdown()

    tout2 = mesh.tilts_out_view().copy(order="F")
    tout2[disk_row] = np.array([0.5, 0.0, 0.0], dtype=float)
    mesh.set_tilts_out_from_array(tout2)
    masked_perturbed = minim.compute_energy_breakdown()

    assert masked_perturbed["tilt_out"] == pytest.approx(masked_baseline["tilt_out"])
    assert masked_perturbed["bending_tilt_out"] == pytest.approx(
        masked_baseline["bending_tilt_out"]
    )

    # Sanity: rim remains a lipid region, so rim tilt_out should still matter.
    tout3 = mesh.tilts_out_view().copy(order="F")
    tout3[rim_row] = np.array([0.2, 0.0, 0.0], dtype=float)
    mesh.set_tilts_out_from_array(tout3)
    masked_rim = minim.compute_energy_breakdown()
    assert masked_rim["tilt_out"] != masked_baseline["tilt_out"]
