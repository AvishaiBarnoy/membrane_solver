import json
import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_tilt_benchmark_runner_smoke(tmp_path: Path) -> None:
    root = _repo_root()
    mesh_dir = root / "meshes" / "tilt_benchmarks"
    mesh_paths = sorted(mesh_dir.glob("*.yaml"))
    assert mesh_paths, "Expected tilt benchmark meshes to exist"

    output_json = tmp_path / "summary.json"
    cmd = [
        sys.executable,
        str(root / "tools" / "tilt_benchmark_runner.py"),
        "--glob",
        str(mesh_dir / "*.yaml"),
        "--output-json",
        str(output_json),
    ]
    proc = subprocess.run(
        cmd,
        cwd=root,
        env=dict(os.environ),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert output_json.exists()

    rows = json.loads(output_json.read_text(encoding="utf-8"))
    assert len(rows) == len(mesh_paths)

    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    assert lines, "Expected benchmark runner to print a metrics table"
    header = lines[0]
    for label in ("mesh", "energy", "|t|_mean", "|t|_max", "div_mean", "div_max"):
        assert label in header

    for path in mesh_paths:
        assert any(line.startswith(path.name) for line in lines[1:]), (
            f"Expected metrics for {path.name}"
        )
