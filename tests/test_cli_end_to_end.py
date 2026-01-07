import json
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


def _run_main(
    *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess:
    root = _repo_root()
    cmd = [sys.executable, str(root / "main.py"), *args]
    proc = subprocess.run(
        cmd,
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc


def _energy_from_json(path: str) -> float:
    mesh = parse_geometry(load_data(path))
    gp = mesh.global_parameters
    minim = Minimizer(
        mesh,
        gp,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
    )
    return float(minim.compute_energy())


def _square_patch_input() -> dict:
    verts = [
        [0.0, 0.0, 0.0, {"fixed": True}],
        [1.0, 0.0, 0.0, {"fixed": True}],
        [1.0, 1.0, 0.0, {"fixed": True}],
        [0.0, 1.0, 0.0, {"fixed": True}],
        [0.5, 0.5, 0.2],
    ]
    edges = [[0, 1], [1, 2], [2, 3], [3, 0], [0, 4], [1, 4], [2, 4], [3, 4]]
    faces = [[0, 5, "r4"], [1, 6, "r5"], [2, 7, "r6"], [3, 4, "r7"]]
    return {
        "vertices": verts,
        "edges": edges,
        "faces": faces,
        "global_parameters": {
            "surface_tension": 1.0,
            "step_size": 1e-2,
            "step_size_mode": "fixed",
        },
        "instructions": [],
    }


def test_main_properties_smoke(tmp_path: Path) -> None:
    config_dir = tmp_path / "mplconfig"
    config_dir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["MPLCONFIGDIR"] = str(config_dir)
    env.setdefault("MPLBACKEND", "Agg")

    proc = _run_main("-i", "meshes/sample_geometry.json", "--properties", env=env)
    assert proc.returncode == 0, proc.stderr
    assert "=== Physical Properties ===" in proc.stdout


def test_main_runs_instructions_and_writes_output(tmp_path: Path) -> None:
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "out.json"
    instructions_path = tmp_path / "instructions.txt"

    input_path.write_text(json.dumps(_square_patch_input()), encoding="utf-8")
    instructions_path.write_text("g5\n", encoding="utf-8")

    config_dir = tmp_path / "mplconfig"
    config_dir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["MPLCONFIGDIR"] = str(config_dir)
    env.setdefault("MPLBACKEND", "Agg")

    e0 = _energy_from_json(str(input_path))
    proc = _run_main(
        "-i",
        str(input_path),
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
