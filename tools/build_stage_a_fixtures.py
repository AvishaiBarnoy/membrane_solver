from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = ROOT / "tests" / "fixtures"
SOURCE_FIXTURE = FIXTURE_DIR / "kozlov_1disk_3d_free_disk_theory_parity.yaml"
BASE_FIXTURE = FIXTURE_DIR / "kozlov_1disk_3d_stage_a_base.yaml"
SEEDED_FIXTURE = FIXTURE_DIR / "kozlov_1disk_3d_stage_a_seeded.yaml"

STAGE_A_RIM_SOURCE_STRENGTH = 0.5
STAGE_A_SEED_Z = 1.0e-4

REMOVED_GLOBAL_KEYS = (
    "tilt_thetaB_group_in",
    "tilt_thetaB_contact_penalty_mode",
    "tilt_thetaB_strength_in",
    "tilt_thetaB_contact_strength_in",
    "tilt_thetaB_value",
    "tilt_thetaB_center",
    "tilt_thetaB_normal",
    "tilt_thetaB_optimize",
    "tilt_thetaB_optimize_every",
    "tilt_thetaB_optimize_delta",
    "tilt_thetaB_optimize_inner_steps",
    "rim_slope_match_thetaB_param",
)


def _ordered_without(values: list[str], removed: set[str]) -> list[str]:
    return [value for value in values if value not in removed]


def _append_unique(values: list[str], value: str) -> list[str]:
    if value not in values:
        values.append(value)
    return values


def _load_source_doc() -> dict[str, Any]:
    return yaml.safe_load(SOURCE_FIXTURE.read_text(encoding="utf-8"))


def build_stage_a_fixture_docs(
    source_doc: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return `(base_doc, seeded_doc)` for the Stage A lane."""
    doc = copy.deepcopy(source_doc if source_doc is not None else _load_source_doc())
    gp = dict(doc.get("global_parameters") or {})

    for key in REMOVED_GLOBAL_KEYS:
        gp.pop(key, None)

    gp["tilt_rim_source_group"] = "disk"
    gp["tilt_rim_source_strength"] = float(STAGE_A_RIM_SOURCE_STRENGTH)
    gp["tilt_rim_source_edge_mode"] = "all"
    gp["theory_parity_lane"] = "stage_a_emergent"
    doc["global_parameters"] = gp

    constraint_modules = list(doc.get("constraint_modules") or [])
    constraint_modules = _ordered_without(
        constraint_modules, {"tilt_thetaB_boundary_in"}
    )
    _append_unique(constraint_modules, "rim_slope_match_out")
    doc["constraint_modules"] = constraint_modules

    energy_modules = list(doc.get("energy_modules") or [])
    energy_modules = _ordered_without(energy_modules, {"tilt_thetaB_contact_in"})
    _append_unique(energy_modules, "tilt_rim_source_bilayer")
    doc["energy_modules"] = energy_modules

    seeded = copy.deepcopy(doc)
    for vertex in seeded.get("vertices") or []:
        if len(vertex) < 4 or not isinstance(vertex[3], dict):
            continue
        opts = vertex[3]
        if str(opts.get("preset") or "") == "rim":
            vertex[2] = float(STAGE_A_SEED_Z)

    return doc, seeded


def write_stage_a_fixtures() -> tuple[Path, Path]:
    """Write the tracked Stage A fixtures to `tests/fixtures/`."""
    base_doc, seeded_doc = build_stage_a_fixture_docs()
    BASE_FIXTURE.write_text(yaml.safe_dump(base_doc, sort_keys=False), encoding="utf-8")
    SEEDED_FIXTURE.write_text(
        yaml.safe_dump(seeded_doc, sort_keys=False), encoding="utf-8"
    )
    return BASE_FIXTURE, SEEDED_FIXTURE


if __name__ == "__main__":
    write_stage_a_fixtures()
