import json
import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_profile_macro_hotspots_smoke(tmp_path: Path) -> None:
    """Smoke-test macro hotspot profiler outputs on a small mesh."""
    root = _repo_root()
    mesh = root / "meshes" / "toy_evolver.yaml"
    outdir = tmp_path / "profiles"
    label = "smoke"
    cmd = [
        sys.executable,
        str(root / "tools" / "profile_macro_hotspots.py"),
        "--mesh",
        str(mesh),
        "--macro",
        "gogo",
        "--max-steps",
        "1",
        "--profile-command",
        "g1",
        "--outdir",
        str(outdir),
        "--label",
        label,
        "--top",
        "10",
        "--quiet",
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

    steps_path = outdir / f"{label}_steps.json"
    profile_json = outdir / f"{label}_g1.json"
    profile_pstats = outdir / f"{label}_g1.pstats"
    profile_txt = outdir / f"{label}_g1.txt"

    assert steps_path.exists()
    assert profile_json.exists()
    assert profile_pstats.exists()
    assert profile_txt.exists()

    step_data = json.loads(steps_path.read_text(encoding="utf-8"))
    assert step_data["macro"] == "gogo"
    assert step_data["num_steps"] == 1
    assert step_data["steps"][0]["command"] == "r"

    profile_data = json.loads(profile_json.read_text(encoding="utf-8"))
    assert profile_data["command"] == "g1"
    assert isinstance(profile_data["elapsed_seconds"], float)
    assert profile_data["elapsed_seconds"] >= 0.0
