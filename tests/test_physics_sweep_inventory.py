import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.diagnostics.physics_sweep import (
    ModuleInventory,
    discover_inventory,
    evaluate_inventory_contract,
)


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
