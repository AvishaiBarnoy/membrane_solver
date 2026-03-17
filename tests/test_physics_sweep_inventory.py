import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.diagnostics.physics_sweep import (
    ModuleInventory,
    discover_inventory,
    evaluate_inventory_contract,
)

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "diagnostics" / "physics_sweep.py"
MATRIX = ROOT / "tests" / "fixtures" / "physics_sweep_matrix.yaml"


def test_evaluate_inventory_contract_reports_missing_and_unexpected() -> None:
    discovered = ModuleInventory(
        array_api_modules=["tilt_in", "tilt_out"],
        leaflet_api_modules=[],
        helper_modules=["scatter"],
    )
    matrix = {
        "required_array_api_modules": ["tilt_in", "bending"],
        "required_leaflet_api_modules": ["bending_tilt_leaflet"],
    }
    out = evaluate_inventory_contract(matrix=matrix, discovered=discovered)
    assert out["missing_required_array_api_modules"] == ["bending"]
    assert out["missing_required_leaflet_api_modules"] == ["bending_tilt_leaflet"]
    assert out["unexpected_array_api_modules"] == ["tilt_out"]
    assert out["required_missing_count"] == 2
    assert out["unexpected_count"] == 1
    assert out["inventory_matches_fixture"] is False


def test_discover_inventory_reports_known_array_and_leaflet_modules() -> None:
    inventory = discover_inventory()
    assert "tilt_in" in inventory.array_api_modules
    assert "bending_tilt_leaflet" in inventory.leaflet_api_modules


@pytest.mark.acceptance
def test_physics_sweep_cli_writes_yaml_and_passes_inventory_gate(tmp_path) -> None:
    out_yaml = tmp_path / "physics_sweep_report.yaml"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--matrix",
            str(MATRIX),
            "--out",
            str(out_yaml),
            "--fail-on-inventory-mismatch",
        ],
        check=True,
        cwd=str(ROOT),
    )
    report = yaml.safe_load(out_yaml.read_text(encoding="utf-8"))
    assert report["meta"]["mode"] == "physics_sweep"
    assert report["meta"]["format"] == "yaml"
    assert report["summary"]["required_missing_count"] == 0
    assert report["summary"]["unexpected_count"] == 0
    assert report["summary"]["all_pass"] is True
    assert report["gradient_checks"]["status"] == "deferred_pr2"
