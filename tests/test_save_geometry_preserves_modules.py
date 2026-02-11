import math

from geometry.geom_io import load_data, parse_geometry, save_geometry
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.gradient_descent import GradientDescent


def test_save_geometry_writes_energy_and_constraint_modules(tmp_path):
    base = load_data("tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity.yaml")
    mesh = parse_geometry(base)
    out_path = tmp_path / "mesh.json"
    save_geometry(mesh, str(out_path), compact=True)

    out = load_data(str(out_path))
    assert out["energy_modules"] == mesh.energy_modules
    assert sorted(out["constraint_modules"]) == sorted(mesh.constraint_modules)


def test_save_load_preserves_energy_breakdown(tmp_path):
    base = load_data("tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity.yaml")
    mesh = parse_geometry(base)
    minim = Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
        tol=1e-6,
    )
    minim.minimize(2)
    expected = minim.compute_energy_breakdown()

    out_path = tmp_path / "mesh.json"
    save_geometry(mesh, str(out_path), compact=True)

    out = load_data(str(out_path))
    mesh2 = parse_geometry(out)
    minim2 = Minimizer(
        mesh2,
        mesh2.global_parameters,
        GradientDescent(),
        EnergyModuleManager(mesh2.energy_modules),
        ConstraintModuleManager(mesh2.constraint_modules),
        quiet=True,
        tol=1e-6,
    )
    got = minim2.compute_energy_breakdown()

    assert set(got) == set(expected)
    for key, expected_val in expected.items():
        got_val = got[key]
        assert math.isclose(got_val, expected_val, rel_tol=1e-5, abs_tol=2e-7)
