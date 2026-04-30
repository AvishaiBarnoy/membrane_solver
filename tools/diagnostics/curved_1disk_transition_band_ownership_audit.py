#!/usr/bin/env python3
"""Audit transition-band energy ownership versus shape-gradient ownership.

This diagnostic does not change runtime physics.  It compares the scalar energy
attributed to the shared-rim support transition band with the projected shape
gradient that drives accepted curved 1-disk updates.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

import numpy as np

from tools.diagnostics.curved_1disk_energy_control_volume_audit import (
    _outer_membrane_tilt_shell_energy,
)
from tools.diagnostics.curved_1disk_first_two_shell_ingredient_audit import (
    _aggregate_row_records,
    _leaflet_runtime_payload,
)
from tools.diagnostics.curved_1disk_shape_direction_audit import (
    _prepare_minimizer,
    _radius_labels,
)
from tools.diagnostics.curved_1disk_shared_rim_phi_target_audit import THEORY_THETA_B
from tools.diagnostics.curved_1disk_theory_benchmark import _run_curved_theta_candidate
from tools.diagnostics.curved_1disk_trumpet_descent_audit import _row_region

THETA_CANDIDATES = (0.06, 0.12, THEORY_THETA_B)
ALLOWED_CLASSIFICATIONS = {
    "support_gradient_matches_energy_ownership",
    "support_gradient_exceeds_energy_ownership",
    "support_gradient_is_constraint_metric_artifact",
    "theta_ordering_depends_on_support_energy",
    "inconclusive",
}


def _transition_rows(mesh) -> np.ndarray:
    """Return the one-ring transition rows incident to the outer support ring."""
    support = {
        int(row)
        for row in range(len(mesh.vertex_ids))
        if _row_region(mesh, int(row)) == "outer_support"
    }
    if not support:
        return np.asarray([], dtype=int)
    tri_rows, _ = mesh.triangle_row_cache()
    rows = set(support)
    if tri_rows is not None:
        for tri in tri_rows:
            tri_set = {int(row) for row in tri}
            if support.intersection(tri_set):
                rows.update(tri_set)
    return np.asarray(sorted(rows), dtype=int)


def _row_masks(mesh) -> dict[str, np.ndarray]:
    n = len(mesh.vertex_ids)
    transition = np.zeros(n, dtype=bool)
    transition[_transition_rows(mesh)] = True
    support = np.asarray(
        [_row_region(mesh, row) == "outer_support" for row in range(n)], dtype=bool
    )
    labels = _radius_labels(mesh)
    free = np.asarray([_row_region(mesh, row) == "outer_free" for row in range(n)])
    return {
        "transition_band": transition,
        "outer_support": support,
        "outer_free": free,
        "outside_transition": ~transition,
        "all": np.ones(n, dtype=bool),
        "radius_labels": labels,
    }


def _project_shape_gradient(minim, grad: np.ndarray) -> np.ndarray:
    out = np.asarray(grad, dtype=float).copy()
    minim.project_constraints_array(out)
    minim._project_curved_free_disk_shape_dofs(out)
    return out


def _module_projected_gradients(minim) -> tuple[dict[str, dict[str, object]], float]:
    """Return projected shape-gradient rows by runtime module."""
    positions, index_map, grad_dummy = minim._soa_views()
    module_rows: dict[str, dict[str, object]] = {}
    for name, module in zip(minim.energy_module_names, minim.energy_modules):
        grad = np.zeros_like(positions)
        energy = minim._call_module_array(
            module,
            positions=positions,
            index_map=index_map,
            grad_arr=grad,
        )
        projected = _project_shape_gradient(minim, grad)
        module_rows[str(name)] = {
            "energy": float(energy),
            "projected_gradient": projected,
            "projected_gradient_norm": float(np.linalg.norm(projected)),
        }

    grad_dummy.fill(0.0)
    _energy, full_grad = minim.compute_energy_and_gradient_array()
    full_projected = _project_shape_gradient(minim, full_grad)
    sum_projected = np.sum(
        [row["projected_gradient"] for row in module_rows.values()], axis=0
    )
    residual = float(np.linalg.norm(sum_projected - full_projected))
    return module_rows, residual


def _row_energy_by_module(mesh) -> dict[str, np.ndarray]:
    """Return approximate local row energy attribution for outer-membrane modules."""
    n = len(mesh.vertex_ids)
    out = {
        "bending_tilt_in": np.zeros(n, dtype=float),
        "bending_tilt_out": np.zeros(n, dtype=float),
        "tilt_in": np.zeros(n, dtype=float),
        "tilt_out": np.zeros(n, dtype=float),
    }
    payload_in = _leaflet_runtime_payload(mesh, leaflet="in")
    payload_out = _leaflet_runtime_payload(mesh, leaflet="out")
    for row, rec in _aggregate_row_records(mesh, payload_in).items():
        out["bending_tilt_in"][int(row)] += float(rec["local_contribution_sum"])
    for row, rec in _aggregate_row_records(mesh, payload_out).items():
        out["bending_tilt_out"][int(row)] += float(rec["local_contribution_sum"])
    for row, value in _outer_membrane_tilt_shell_energy(mesh, payload_in).items():
        out["tilt_in"][int(row)] += float(value)
    for row, value in _outer_membrane_tilt_shell_energy(mesh, payload_out).items():
        out["tilt_out"][int(row)] += float(value)
    return out


def _region_gradient_summary(
    *,
    mesh,
    module_gradients: dict[str, dict[str, object]],
    row_energy: dict[str, np.ndarray],
) -> dict[str, object]:
    masks = _row_masks(mesh)
    transition = masks["transition_band"]
    area = _row_control_area(mesh)
    module_rows: list[dict[str, object]] = []
    total_transition_grad_sq = 0.0
    total_grad_sq = 0.0
    total_transition_energy = 0.0
    total_energy = 0.0
    for name, payload in module_gradients.items():
        grad = np.asarray(payload["projected_gradient"], dtype=float)
        grad_norm_by_row = np.linalg.norm(grad, axis=1)
        grad_total = float(np.linalg.norm(grad))
        grad_transition = float(np.linalg.norm(grad[transition]))
        energy_rows = row_energy.get(str(name), np.zeros(len(mesh.vertex_ids)))
        energy_total = float(np.sum(energy_rows))
        energy_transition = float(np.sum(energy_rows[transition]))
        area_transition = float(np.sum(area[transition]))
        total_transition_grad_sq += grad_transition**2
        total_grad_sq += grad_total**2
        total_transition_energy += energy_transition
        total_energy += energy_total
        module_rows.append(
            {
                "module": str(name),
                "energy_total": energy_total,
                "energy_transition_band": energy_transition,
                "energy_transition_fraction": _safe_ratio(
                    abs(energy_transition), abs(energy_total)
                ),
                "projected_gradient_norm_total": grad_total,
                "projected_gradient_norm_transition_band": grad_transition,
                "gradient_transition_fraction": _safe_ratio(
                    grad_transition, grad_total
                ),
                "gradient_per_abs_energy_transition": _safe_ratio(
                    grad_transition, abs(energy_transition)
                ),
                "gradient_per_area_transition": _safe_ratio(
                    grad_transition, area_transition
                ),
                "top_transition_rows": _top_rows(mesh, grad_norm_by_row, transition),
            }
        )
    return {
        "modules": sorted(
            module_rows,
            key=lambda row: float(row["projected_gradient_norm_transition_band"]),
            reverse=True,
        ),
        "totals": {
            "energy_total_attributed": float(total_energy),
            "energy_transition_band_attributed": float(total_transition_energy),
            "energy_transition_fraction": _safe_ratio(
                abs(total_transition_energy), abs(total_energy)
            ),
            "projected_gradient_norm_total_rss": float(np.sqrt(total_grad_sq)),
            "projected_gradient_norm_transition_band_rss": float(
                np.sqrt(total_transition_grad_sq)
            ),
            "gradient_transition_fraction": _safe_ratio(
                float(np.sqrt(total_transition_grad_sq)), float(np.sqrt(total_grad_sq))
            ),
        },
    }


def _row_control_area(mesh) -> np.ndarray:
    positions = mesh.positions_view()
    tri_rows, _ = mesh.triangle_row_cache()
    area = np.zeros(len(mesh.vertex_ids), dtype=float)
    if tri_rows is None or len(tri_rows) == 0:
        return np.ones(len(mesh.vertex_ids), dtype=float)
    tri_pos = positions[tri_rows]
    tri_area = 0.5 * np.linalg.norm(
        np.cross(
            tri_pos[:, 1, :] - tri_pos[:, 0, :], tri_pos[:, 2, :] - tri_pos[:, 0, :]
        ),
        axis=1,
    )
    np.add.at(area, tri_rows.ravel(), np.repeat(tri_area / 3.0, 3))
    return area


def _top_rows(mesh, values: np.ndarray, mask: np.ndarray) -> list[dict[str, object]]:
    labels = _radius_labels(mesh)
    rows = []
    for row in np.flatnonzero(mask):
        rows.append(
            {
                "row": int(row),
                "region": _row_region(mesh, int(row)),
                "radius": float(labels[int(row)]),
                "value": float(values[int(row)]),
            }
        )
    return sorted(rows, key=lambda row: abs(float(row["value"])), reverse=True)[:8]


def _safe_ratio(numer: float, denom: float) -> float:
    if abs(float(denom)) <= 1.0e-14:
        return 0.0
    return float(numer) / float(denom)


def _theta_candidate_rows(theta_values: Sequence[float]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for theta in theta_values:
        result = _run_curved_theta_candidate(float(theta))
        mesh = result["mesh"]
        row_energy = _row_energy_by_module(mesh)
        transition = _row_masks(mesh)["transition_band"]
        support_energy = {
            name: float(np.sum(values[transition]))
            for name, values in row_energy.items()
        }
        total_support_energy = float(sum(support_energy.values()))
        total_energy = float(result["near_rim"]["total_energy"])
        rows.append(
            {
                "theta_B": float(theta),
                "total_energy": total_energy,
                "transition_band_energy_by_module": support_energy,
                "transition_band_energy_total": total_support_energy,
                "energy_without_transition_band_attributed": float(
                    total_energy - total_support_energy
                ),
            }
        )
    rows_sorted = sorted(rows, key=lambda row: float(row["total_energy"]))
    rows_wo = sorted(
        rows, key=lambda row: float(row["energy_without_transition_band_attributed"])
    )
    for row in rows:
        row["selected_by_total_energy"] = bool(row is rows_sorted[0])
        row["selected_without_transition_band_attributed"] = bool(row is rows_wo[0])
    return rows


def _classify(
    region_summary: dict[str, object],
    theta_rows: list[dict[str, object]],
    gradient_residual: float,
) -> str:
    totals = region_summary["totals"]
    grad_fraction = float(totals["gradient_transition_fraction"])
    energy_fraction = float(totals["energy_transition_fraction"])
    selected_total = next(
        row["theta_B"] for row in theta_rows if row["selected_by_total_energy"]
    )
    selected_wo = next(
        row["theta_B"]
        for row in theta_rows
        if row["selected_without_transition_band_attributed"]
    )
    if gradient_residual > 1.0e-8:
        return "support_gradient_is_constraint_metric_artifact"
    if selected_total != selected_wo:
        return "theta_ordering_depends_on_support_energy"
    if grad_fraction > max(0.65, 2.0 * energy_fraction):
        return "support_gradient_exceeds_energy_ownership"
    if abs(grad_fraction - energy_fraction) <= 0.20:
        return "support_gradient_matches_energy_ownership"
    return "inconclusive"


def run_curved_1disk_transition_band_ownership_audit(
    *,
    theta_b: float = THEORY_THETA_B,
    theta_values: Sequence[float] = THETA_CANDIDATES,
) -> dict[str, object]:
    minim = _prepare_minimizer(float(theta_b))
    mesh = minim.mesh
    module_gradients, gradient_residual = _module_projected_gradients(minim)
    row_energy = _row_energy_by_module(mesh)
    region_summary = _region_gradient_summary(
        mesh=mesh,
        module_gradients=module_gradients,
        row_energy=row_energy,
    )
    theta_rows = _theta_candidate_rows(theta_values)
    classification = _classify(region_summary, theta_rows, gradient_residual)
    return {
        "theta_B": float(theta_b),
        "transition_band": {
            "row_count": int(_transition_rows(mesh).size),
            "row_regions": sorted(
                {_row_region(mesh, int(row)) for row in _transition_rows(mesh).tolist()}
            ),
        },
        "module_gradient_reconciliation": {
            "sum_projected_minus_full_norm": float(gradient_residual),
        },
        "region_ownership": region_summary,
        "theta_candidate_ordering": theta_rows,
        "diagnosis": {
            "classification": classification,
            "allowed_classifications": sorted(ALLOWED_CLASSIFICATIONS),
            "recommendation": _recommendation(classification),
            "no_energy_rescaling": True,
        },
    }


def _recommendation(classification: str) -> str:
    if classification == "support_gradient_exceeds_energy_ownership":
        return (
            "Next Feature Contract should target transition-band control-volume "
            "or shape-gradient ownership using geometry-derived invariants."
        )
    if classification == "support_gradient_matches_energy_ownership":
        return (
            "Next Feature Contract should target a geometry-derived solver metric "
            "or preconditioner with benchmark validation."
        )
    if classification == "theta_ordering_depends_on_support_energy":
        return (
            "Separate support-band scalar energy ownership from shape-step metric "
            "before changing runtime behavior. Do not freeze the support band."
        )
    return (
        "Collect a narrower transition-band module audit before runtime changes. "
        "Keep the next stream diagnostic-first."
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theta", type=float, default=THEORY_THETA_B)
    args = parser.parse_args(argv)
    report = run_curved_1disk_transition_band_ownership_audit(theta_b=float(args.theta))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
