#!/usr/bin/env python3
"""Inventory helpers for physics sweep diagnostics."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
ENERGY_DIR = ROOT / "modules" / "energy"


@dataclass(frozen=True)
class ModuleInventory:
    """Discovered energy-module API coverage."""

    array_api_modules: list[str]
    leaflet_api_modules: list[str]
    helper_modules: list[str]


def _parse_module_functions(module_path: Path) -> set[str]:
    """Return top-level function names defined in an energy module."""

    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    return {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}


def _sorted_unique(values: list[str]) -> list[str]:
    """Return sorted unique strings while preserving value semantics."""

    return sorted(dict.fromkeys(values))


def discover_inventory() -> ModuleInventory:
    """Discover energy modules that implement the expected public APIs."""

    array_api_modules: list[str] = []
    leaflet_api_modules: list[str] = []
    helper_modules: list[str] = []

    for module_path in sorted(ENERGY_DIR.glob("*.py")):
        stem = module_path.stem
        if stem.startswith("__") or stem.startswith("dummy_"):
            continue
        functions = _parse_module_functions(module_path)
        if "compute_energy_and_gradient_array" in functions:
            array_api_modules.append(stem)
        if (
            "compute_energy_leaflet" in functions
            or "compute_energy_and_gradient_leaflet" in functions
            or "compute_energy_and_gradient_array_leaflet" in functions
        ):
            leaflet_api_modules.append(stem)
        if (
            "prepare_contact_rows" in functions
            or "active_leaflet_vertices" in functions
        ):
            helper_modules.append(stem)

    return ModuleInventory(
        array_api_modules=_sorted_unique(array_api_modules),
        leaflet_api_modules=_sorted_unique(leaflet_api_modules),
        helper_modules=_sorted_unique(helper_modules),
    )


def evaluate_inventory_contract(
    matrix: dict[str, Any], discovered: ModuleInventory
) -> dict[str, Any]:
    """Evaluate required inventory and report missing or unexpected modules."""

    required_array = _sorted_unique(list(matrix.get("required_array_api_modules", [])))
    required_leaflet = _sorted_unique(
        list(matrix.get("required_leaflet_api_modules", []))
    )

    discovered_array = _sorted_unique(discovered.array_api_modules)
    discovered_leaflet = _sorted_unique(discovered.leaflet_api_modules)

    missing_array = sorted(set(required_array) - set(discovered_array))
    missing_leaflet = sorted(set(required_leaflet) - set(discovered_leaflet))
    unexpected_array = sorted(set(discovered_array) - set(required_array))

    required_missing_count = len(missing_array) + len(missing_leaflet)
    unexpected_count = len(unexpected_array)

    return {
        "required_array_api_modules": required_array,
        "required_leaflet_api_modules": required_leaflet,
        "discovered_array_api_modules": discovered_array,
        "discovered_leaflet_api_modules": discovered_leaflet,
        "discovered_helper_modules": _sorted_unique(discovered.helper_modules),
        "missing_required_array_api_modules": missing_array,
        "missing_required_leaflet_api_modules": missing_leaflet,
        "unexpected_array_api_modules": unexpected_array,
        "required_missing_count": required_missing_count,
        "unexpected_count": unexpected_count,
        "inventory_matches_fixture": required_missing_count == 0
        and unexpected_count == 0,
    }


def _deferred_status(pr_label: str | None) -> str:
    """Return the deferred status token for a planned PR stage."""

    if not pr_label:
        return "deferred"
    return f"deferred_{pr_label.strip().lower()}"


def _build_report(matrix: dict[str, Any], inventory: dict[str, Any]) -> dict[str, Any]:
    """Build a lightweight physics-sweep report without executing deferred stages."""

    gradient_groups = list(matrix.get("gradient_check_groups", []))
    gradient_status = _deferred_status(
        gradient_groups[0].get("pr") if gradient_groups else None
    )
    return {
        "meta": {
            "mode": "physics_sweep",
            "format": "yaml",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "root": str(ROOT),
        },
        "summary": {
            "required_missing_count": inventory["required_missing_count"],
            "unexpected_count": inventory["unexpected_count"],
            "inventory_matches_fixture": inventory["inventory_matches_fixture"],
            "all_pass": inventory["inventory_matches_fixture"],
        },
        "routine_lock": matrix.get("routine_lock", {}),
        "inventory": inventory,
        "gradient_checks": {
            "status": gradient_status,
            "groups": gradient_groups,
            "executed": False,
        },
    }


def _write_report(report: dict[str, Any], out_path: Path) -> None:
    """Write the physics-sweep report as YAML."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the lightweight physics sweep."""

    parser = argparse.ArgumentParser(description="Physics sweep inventory reporter")
    parser.add_argument(
        "--matrix",
        type=Path,
        default=ROOT / "tests" / "fixtures" / "physics_sweep_matrix.yaml",
        help="Path to physics sweep matrix YAML",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/tmp/physics_sweep_report.yaml"),
        help="Output YAML report path",
    )
    parser.add_argument(
        "--fail-on-inventory-mismatch",
        action="store_true",
        help="Exit non-zero when the inventory does not match the matrix fixture",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the physics-sweep inventory CLI."""

    args = parse_args(argv)
    matrix = yaml.safe_load(args.matrix.read_text(encoding="utf-8"))
    discovered = discover_inventory()
    inventory = evaluate_inventory_contract(matrix=matrix, discovered=discovered)
    report = _build_report(matrix=matrix, inventory=inventory)
    _write_report(report, args.out)
    if args.fail_on_inventory_mismatch and not inventory["inventory_matches_fixture"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
