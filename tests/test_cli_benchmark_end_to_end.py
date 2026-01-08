import math
import os
import subprocess
import sys
from pathlib import Path

sys.path.append(os.getcwd())

from geometry.geom_io import load_data, parse_geometry
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.gradient_descent import GradientDescent


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run_main(*args: str, env: dict[str, str]) -> subprocess.CompletedProcess:
    root = _repo_root()
    cmd = [sys.executable, str(root / "main.py"), *args]
    return subprocess.run(
        cmd,
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _energy_from_json(path: str) -> float:
    mesh = parse_geometry(load_data(path))
    minim = Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
    )
    return float(minim.compute_energy())


def _base_env(tmp_path: Path) -> dict[str, str]:
    config_dir = tmp_path / "mplconfig"
    config_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["MPLCONFIGDIR"] = str(config_dir)
    env.setdefault("MPLBACKEND", "Agg")
    return env


def test_main_benchmark_helfrich_sphere_match_smoke(tmp_path: Path) -> None:
    """End-to-end: run a small bending benchmark without crashing."""
    input_path = "benchmarks/inputs/bench_helfrich_sphere_match.json"
    output_path = tmp_path / "out.json"
    instructions_path = tmp_path / "instructions.txt"
    instructions_path.write_text("g1\n", encoding="utf-8")

    env = _base_env(tmp_path)
    proc = _run_main(
        "-i",
        input_path,
        "--instructions",
        str(instructions_path),
        "--non-interactive",
        "--output",
        str(output_path),
        "--quiet",
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    assert output_path.exists()

    energy = _energy_from_json(str(output_path))
    assert math.isfinite(energy)
    # For this benchmark the initial geometry is already close to the target.
    assert abs(energy) <= 1e-8

    mesh = parse_geometry(load_data(str(output_path)))
    assert mesh.validate_edge_indices() is True
    assert mesh.validate_body_orientation() is True


def test_main_benchmark_dented_cube_energy_decreases(tmp_path: Path) -> None:
    """End-to-end: a short minimization run decreases energy and keeps topology sane."""
    input_path = "meshes/bench_dented_cube.json"
    output_path = tmp_path / "out.json"
    instructions_path = tmp_path / "instructions.txt"
    instructions_path.write_text("g3\n", encoding="utf-8")

    env = _base_env(tmp_path)

    e0 = _energy_from_json(input_path)
    proc = _run_main(
        "-i",
        input_path,
        "--instructions",
        str(instructions_path),
        "--non-interactive",
        "--output",
        str(output_path),
        "--quiet",
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    assert output_path.exists()

    e1 = _energy_from_json(str(output_path))
    assert e1 <= e0 + 1e-12

    mesh = parse_geometry(load_data(str(output_path)))
    assert mesh.validate_edge_indices() is True
    assert mesh.validate_body_orientation() is True
    assert mesh.validate_body_outwardness() is True
