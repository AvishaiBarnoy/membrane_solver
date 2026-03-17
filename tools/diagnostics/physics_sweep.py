#!/usr/bin/env python3
"""Inventory helpers for physics sweep diagnostics."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
