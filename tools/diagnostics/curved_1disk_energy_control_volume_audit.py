#!/usr/bin/env python3
"""Energy/control-volume audit for the curved 1-disk free-membrane miss.

This is a diagnostic-only report.  It reuses the current runtime state and
post-processes local shell contributions; it does not change constraints,
energy modules, solver settings, or benchmark acceptance thresholds.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable

import numpy as np

from tools.diagnostics.curved_1disk_first_two_shell_ingredient_audit import (
    OUTER_RADIUS,
    THEORY_THETA_B,
    _aggregate_row_records,
    _leaflet_runtime_payload,
)
from tools.diagnostics.curved_1disk_theory_benchmark import (
    _compute_numeric_energy_split,
    _run_curved_theta_candidate,
)
from tools.diagnostics.curved_disk_theory import (
    compute_curved_disk_theory,
    tex_reference_params,
)
from tools.diagnostics.free_disk_profile_protocol import (
    _shared_rim_group_rows,
    _shared_rim_inner_control_volume_audit,
    _triangle_region_masks,
    shared_rim_continuum_annulus_audit,
    shared_rim_shell_area_audit,
)

SELECTED_THETA_B_AFTER_SHARED_RIM_FIX = 0.12
DEFAULT_THETA_VALUES = (SELECTED_THETA_B_AFTER_SHARED_RIM_FIX, THEORY_THETA_B)


def _safe_ratio(numer: float, denom: float) -> float:
    """Return ``numer / denom`` with a finite small-denominator fallback."""
    if abs(float(denom)) <= 1.0e-12:
        return float("inf") if float(numer) else 0.0
    return float(numer) / float(denom)


def _expected_tex_energy(theta_b: float) -> dict[str, float]:
    """Return TeX quadratic/linear split evaluated at ``theta_b``."""
    theory = compute_curved_disk_theory(tex_reference_params())
    theta_opt = float(theory.theta_star)
    theta_sq_ratio = (float(theta_b) / max(abs(theta_opt), 1.0e-12)) ** 2
    theta_ratio = float(theta_b) / max(abs(theta_opt), 1.0e-12)
    inner = float(theory.elastic_inner) * theta_sq_ratio
    outer = float(theory.elastic_outer) * theta_sq_ratio
    contact = float(theory.contact) * theta_ratio
    return {
        "theta_B": float(theta_b),
        "theta_B_opt": theta_opt,
        "inner_elastic": inner,
        "outer_elastic": outer,
        "contact": contact,
        "total": inner + outer + contact,
    }


def _row_regions(mesh) -> list[str]:
    """Return a compact region label for every vertex row."""
    labels: list[str] = []
    for vid in mesh.vertex_ids:
        opts = getattr(mesh.vertices[int(vid)], "options", None) or {}
        preset = str(opts.get("preset") or "")
        group = str(opts.get("rim_slope_match_group") or "")
        if preset == "disk":
            labels.append("disk")
        elif group == "rim":
            labels.append("shared_rim")
        elif group == "outer":
            labels.append("outer_support")
        else:
            labels.append("outer_free")
    return labels


def _shell_index_map(radii: np.ndarray) -> dict[float, int]:
    """Return shell index keyed by rounded radius."""
    shells = sorted({round(float(v), 6) for v in radii if float(v) > 1.0e-12})
    return {float(radius): idx for idx, radius in enumerate(shells)}


def _outer_membrane_tilt_shell_energy(
    mesh, payload: dict[str, object]
) -> dict[int, float]:
    """Return module-shaped tilt-magnitude energy by row on outer-membrane triangles."""
    leaflet = str(payload["leaflet"])
    gp = mesh.global_parameters
    tri_rows = np.asarray(payload["tri_rows"], dtype=np.int32)
    outer_mask = np.asarray(payload["outer_mask"], dtype=bool)
    tri_area = np.asarray(payload["tri_area"], dtype=float)
    tilts = np.asarray(payload["tilt_vectors"], dtype=float).copy()
    if tri_rows.size == 0 or not np.any(outer_mask):
        return {}

    if leaflet == "out":
        k_tilt = float(gp.get("tilt_modulus_out") or 0.0)
        mode = str(gp.get("tilt_mass_mode_out") or gp.get("tilt_mass_mode") or "lumped")
        shell_mode = None
        if bool(
            gp.get("tilt_out_exclude_shared_rim_outer_rows")
            or gp.get("tilt_out_exclude_shared_rim_rows")
        ):
            for row in _shared_rim_group_rows(mesh, "outer"):
                tilts[int(row)] = 0.0
    else:
        k_tilt = float(gp.get("tilt_modulus_in") or 0.0)
        mode = str(gp.get("tilt_mass_mode_in") or gp.get("tilt_mass_mode") or "lumped")
        raw_shell_mode = gp.get("tilt_in_shared_rim_outer_shell_mass_mode")
        shell_mode = None if raw_shell_mode is None else str(raw_shell_mode)
        outer_weight = gp.get("tilt_in_shared_rim_outer_row_energy_weight")
        if outer_weight is not None:
            scale = float(np.sqrt(float(outer_weight)))
            for row in _shared_rim_group_rows(mesh, "outer"):
                tilts[int(row)] *= scale
        if bool(
            gp.get("tilt_in_exclude_shared_rim_rows")
            or gp.get("tilt_exclude_shared_rim_rows_in")
        ):
            for row in _shared_rim_group_rows(mesh, "rim"):
                tilts[int(row)] = 0.0
        if bool(
            gp.get("tilt_in_exclude_shared_rim_outer_rows")
            or gp.get("tilt_exclude_shared_rim_outer_rows_in")
        ):
            for row in _shared_rim_group_rows(mesh, "outer"):
                tilts[int(row)] = 0.0

    mode = mode.strip().lower()
    shell_mode = None if shell_mode is None else shell_mode.strip().lower()
    if mode not in {"lumped", "consistent"}:
        raise ValueError("Leaflet tilt mass mode must be 'lumped' or 'consistent'.")
    if shell_mode not in {None, "lumped", "consistent"}:
        raise ValueError(
            "tilt_in_shared_rim_outer_shell_mass_mode must be 'lumped' or 'consistent'."
        )

    rows_eff = tri_rows[outer_mask]
    area_eff = tri_area[outer_mask]
    region_masks_full = _triangle_region_masks(mesh, tri_rows)
    outer_support_mask = np.asarray(
        region_masks_full["outer_support_band"], dtype=bool
    )[outer_mask]
    use_consistent = np.full(len(rows_eff), mode == "consistent", dtype=bool)
    if leaflet == "in" and shell_mode is not None:
        use_consistent[outer_support_mask] = shell_mode == "consistent"

    energy_by_row = np.zeros(len(mesh.vertex_ids), dtype=float)
    t0 = tilts[rows_eff[:, 0]]
    t1 = tilts[rows_eff[:, 1]]
    t2 = tilts[rows_eff[:, 2]]
    corner_sq = np.stack(
        [
            np.einsum("ij,ij->i", t0, t0),
            np.einsum("ij,ij->i", t1, t1),
            np.einsum("ij,ij->i", t2, t2),
        ],
        axis=1,
    )

    lumped = ~use_consistent
    if np.any(lumped):
        corner_e = 0.5 * k_tilt * corner_sq[lumped] * (area_eff[lumped, None] / 3.0)
        np.add.at(energy_by_row, rows_eff[lumped], corner_e)

    if np.any(use_consistent):
        dot01 = np.einsum("ij,ij->i", t0[use_consistent], t1[use_consistent])
        dot12 = np.einsum("ij,ij->i", t1[use_consistent], t2[use_consistent])
        dot20 = np.einsum("ij,ij->i", t2[use_consistent], t0[use_consistent])
        c_sq = corner_sq[use_consistent]
        c0 = c_sq[:, 0] + 0.5 * (dot01 + dot20)
        c1 = c_sq[:, 1] + 0.5 * (dot01 + dot12)
        c2 = c_sq[:, 2] + 0.5 * (dot12 + dot20)
        corner_e = (
            (k_tilt / 12.0)
            * area_eff[use_consistent, None]
            * np.stack([c0, c1, c2], axis=1)
        )
        np.add.at(energy_by_row, rows_eff[use_consistent], corner_e)

    return {
        int(row): float(value)
        for row, value in enumerate(energy_by_row)
        if abs(float(value)) > 1.0e-15
    }


def _shell_energy_rows(mesh) -> list[dict[str, object]]:
    """Return shellwise outer-membrane energy and control-area rows."""
    positions = mesh.positions_view()
    radii = np.linalg.norm(positions[:, :2], axis=1)
    radius_labels = np.asarray([round(float(v), 6) for v in radii], dtype=float)
    shell_indices = _shell_index_map(radii)
    row_regions = _row_regions(mesh)

    payload_in = _leaflet_runtime_payload(mesh, leaflet="in")
    payload_out = _leaflet_runtime_payload(mesh, leaflet="out")
    bend_in = _aggregate_row_records(mesh, payload_in)
    bend_out = _aggregate_row_records(mesh, payload_out)
    tilt_in = _outer_membrane_tilt_shell_energy(mesh, payload_in)
    tilt_out = _outer_membrane_tilt_shell_energy(mesh, payload_out)

    rows_by_shell: dict[float, dict[str, object]] = {}
    for row, shell_radius in enumerate(radius_labels):
        if float(shell_radius) <= OUTER_RADIUS + 1.0e-6:
            continue
        entry = rows_by_shell.setdefault(
            float(shell_radius),
            {
                "shell_index": int(shell_indices[float(shell_radius)]),
                "radius": float(shell_radius),
                "row_count": 0,
                "row_regions": set(),
                "tilt_in_outer_membrane": 0.0,
                "tilt_out_outer_membrane": 0.0,
                "bending_tilt_in_outer_membrane": 0.0,
                "bending_tilt_out_outer_membrane": 0.0,
                "effective_area_in": 0.0,
                "effective_area_out": 0.0,
                "voronoi_area_in": 0.0,
                "voronoi_area_out": 0.0,
            },
        )
        entry["row_count"] = int(entry["row_count"]) + 1
        entry["row_regions"].add(row_regions[row])
        entry["tilt_in_outer_membrane"] += float(tilt_in.get(row, 0.0))
        entry["tilt_out_outer_membrane"] += float(tilt_out.get(row, 0.0))
        if row in bend_in:
            rec = bend_in[row]
            entry["bending_tilt_in_outer_membrane"] += float(
                rec["local_contribution_sum"]
            )
            entry["effective_area_in"] += float(rec["effective_area_sum"])
            entry["voronoi_area_in"] += float(rec["vertex_area_vor"])
        if row in bend_out:
            rec = bend_out[row]
            entry["bending_tilt_out_outer_membrane"] += float(
                rec["local_contribution_sum"]
            )
            entry["effective_area_out"] += float(rec["effective_area_sum"])
            entry["voronoi_area_out"] += float(rec["vertex_area_vor"])

    out: list[dict[str, object]] = []
    for shell_radius, entry in sorted(rows_by_shell.items()):
        total = (
            float(entry["tilt_in_outer_membrane"])
            + float(entry["tilt_out_outer_membrane"])
            + float(entry["bending_tilt_in_outer_membrane"])
            + float(entry["bending_tilt_out_outer_membrane"])
        )
        regions = sorted(str(v) for v in entry.pop("row_regions"))
        out.append(
            {
                **entry,
                "radius": float(shell_radius),
                "row_regions": regions,
                "outer_membrane_elastic_total": float(total),
            }
        )
    return out


def _support_concentration(shell_rows: list[dict[str, object]]) -> dict[str, float]:
    """Return concentration of outer elastic energy in shared-rim support shells."""
    total = sum(float(row["outer_membrane_elastic_total"]) for row in shell_rows)
    support = sum(
        float(row["outer_membrane_elastic_total"])
        for row in shell_rows
        if "shared_rim" in row["row_regions"] or "outer_support" in row["row_regions"]
    )
    first_two = sum(
        float(row["outer_membrane_elastic_total"])
        for row in sorted(shell_rows, key=lambda item: float(item["radius"]))[:2]
    )
    return {
        "outer_membrane_elastic_total_from_shell_rows": float(total),
        "shared_rim_support_shell_elastic": float(support),
        "first_two_outer_shell_elastic": float(first_two),
        "support_fraction_of_outer_shell_elastic": _safe_ratio(support, total),
        "first_two_fraction_of_outer_shell_elastic": _safe_ratio(first_two, total),
    }


def _shell_attribution_coverage(
    *,
    shell_concentration: dict[str, float],
    energy_split: dict[str, float],
) -> dict[str, float]:
    """Return how much numeric outer elastic is explained by shell rows."""
    shell_total = float(
        shell_concentration["outer_membrane_elastic_total_from_shell_rows"]
    )
    numeric_outer = float(energy_split["outer_elastic_numeric"])
    residual = numeric_outer - shell_total
    return {
        "numeric_outer_elastic": numeric_outer,
        "shell_attributed_outer_elastic": shell_total,
        "unattributed_outer_elastic": float(residual),
        "shell_attributed_fraction": _safe_ratio(shell_total, numeric_outer),
        "unattributed_fraction": _safe_ratio(residual, numeric_outer),
    }


def _runtime_module_totals(runtime_breakdown: dict[str, float]) -> dict[str, float]:
    """Return runtime module totals grouped for diagnostic energy ownership."""
    tilt_in = float(runtime_breakdown.get("tilt_in", 0.0))
    tilt_out = float(runtime_breakdown.get("tilt_out", 0.0))
    bending_tilt_in = float(runtime_breakdown.get("bending_tilt_in", 0.0))
    bending_tilt_out = float(runtime_breakdown.get("bending_tilt_out", 0.0))
    contact = float(runtime_breakdown.get("tilt_thetaB_contact_in", 0.0))
    elastic = tilt_in + tilt_out + bending_tilt_in + bending_tilt_out
    total = float(sum(float(v) for v in runtime_breakdown.values()))
    return {
        "tilt_in": tilt_in,
        "tilt_out": tilt_out,
        "bending_tilt_in": bending_tilt_in,
        "bending_tilt_out": bending_tilt_out,
        "elastic_total": float(elastic),
        "contact": contact,
        "total": total,
    }


def _reconciled_runtime_energy_split(
    *,
    legacy_energy_split: dict[str, float],
    runtime_breakdown: dict[str, float],
    shell_concentration: dict[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    """Return a disk/outer split reconciled to runtime module totals.

    The older split helper estimates disk/outer energies independently from
    local formulas.  Gate A uses the shell-attributed outer energy as the outer
    split and assigns the remaining runtime elastic module energy to the disk
    side, so the diagnostic split cannot invent elastic energy.
    """
    modules = _runtime_module_totals(runtime_breakdown)
    outer_elastic = float(
        shell_concentration["outer_membrane_elastic_total_from_shell_rows"]
    )
    inner_elastic = float(modules["elastic_total"] - outer_elastic)
    split = {
        "total_numeric": float(modules["total"]),
        "inner_elastic_numeric": inner_elastic,
        "outer_elastic_numeric": outer_elastic,
        "contact_numeric": float(modules["contact"]),
    }
    elastic_residual = float(
        modules["elastic_total"]
        - split["inner_elastic_numeric"]
        - split["outer_elastic_numeric"]
    )
    total_residual = float(
        modules["total"]
        - split["inner_elastic_numeric"]
        - split["outer_elastic_numeric"]
        - split["contact_numeric"]
    )
    reconciliation = {
        **modules,
        "elastic_residual": elastic_residual,
        "total_residual": total_residual,
        "legacy_total_minus_runtime_total": float(
            legacy_energy_split["total_numeric"] - modules["total"]
        ),
        "legacy_inner_minus_reconciled_inner": float(
            legacy_energy_split["inner_elastic_numeric"] - inner_elastic
        ),
        "legacy_outer_minus_reconciled_outer": float(
            legacy_energy_split["outer_elastic_numeric"] - outer_elastic
        ),
    }
    return split, reconciliation


def _control_volume_evidence(mesh) -> dict[str, object]:
    """Return shared-rim support-area evidence."""
    control = _shared_rim_inner_control_volume_audit(mesh)
    annulus = shared_rim_continuum_annulus_audit(mesh)
    shells = shared_rim_shell_area_audit(mesh)
    outer_annulus_ratio = _safe_ratio(
        float(control["outer_control_area"]), float(annulus["outer_annulus_area"])
    )
    rim_annulus_ratio = _safe_ratio(
        float(control["rim_control_area"]), float(annulus["rim_annulus_area"])
    )
    outer_shell_ratio = _safe_ratio(
        float(control["outer_control_area"]), float(shells["outer_shell_area"])
    )
    rim_shell_ratio = _safe_ratio(
        float(control["rim_control_area"]), float(shells["rim_shell_area"])
    )
    return {
        "inner_leaflet_barycentric_control_area": control,
        "continuum_gap_annulus": annulus,
        "adjacent_shell_area": shells,
        "ratios": {
            "outer_control_over_gap_annulus": outer_annulus_ratio,
            "rim_control_over_gap_annulus": rim_annulus_ratio,
            "outer_control_over_adjacent_shell": outer_shell_ratio,
            "rim_control_over_adjacent_shell": rim_shell_ratio,
        },
        "call": (
            "shared-rim support control volume is oversized versus narrow gap annulus"
            if outer_annulus_ratio > 4.0 or rim_annulus_ratio > 2.0
            else "shared-rim support control volume is not oversized by gap-annulus test"
        ),
    }


def _diagnose_case(
    *,
    energy_split: dict[str, float],
    expected: dict[str, float],
    shell_concentration: dict[str, float],
    shell_coverage: dict[str, float],
    control_volume: dict[str, object],
) -> dict[str, object]:
    """Return compact root-cause calls for one theta case."""
    outer_ratio = _safe_ratio(
        float(energy_split["outer_elastic_numeric"]), float(expected["outer_elastic"])
    )
    inner_ratio = _safe_ratio(
        float(energy_split["inner_elastic_numeric"]), float(expected["inner_elastic"])
    )
    support_fraction = float(
        shell_concentration["support_fraction_of_outer_shell_elastic"]
    )
    gap_ratio = float(
        control_volume["ratios"]["outer_control_over_gap_annulus"]  # type: ignore[index]
    )
    calls: list[str] = []
    if outer_ratio > 5.0:
        calls.append("outer elastic remains far above TeX quadratic energy")
    if inner_ratio < 0.25:
        calls.append("inner elastic remains far below TeX quadratic energy")
    if support_fraction > 0.5:
        calls.append("outer elastic is concentrated in shared-rim support shells")
    if float(shell_coverage["unattributed_fraction"]) > 0.5:
        calls.append("shell-local attribution does not explain most outer elastic")
    if gap_ratio > 4.0:
        calls.append("shared-rim support control volume exceeds narrow gap annulus")
    return {
        "outer_numeric_over_tex": outer_ratio,
        "inner_numeric_over_tex": inner_ratio,
        "contact_numeric_over_tex": _safe_ratio(
            float(energy_split["contact_numeric"]), float(expected["contact"])
        ),
        "dominant_calls": calls,
    }


def _run_case(theta_b: float) -> dict[str, object]:
    """Run one forced theta case and return the energy/control-volume audit."""
    result = _run_curved_theta_candidate(float(theta_b))
    mesh = result["mesh"]
    legacy_energy_split = _compute_numeric_energy_split(mesh)
    expected = _expected_tex_energy(float(theta_b))
    shell_rows = _shell_energy_rows(mesh)
    shell_concentration = _support_concentration(shell_rows)
    energy_split, runtime_reconciliation = _reconciled_runtime_energy_split(
        legacy_energy_split=legacy_energy_split,
        runtime_breakdown=result["breakdown"],
        shell_concentration=shell_concentration,
    )
    shell_coverage = _shell_attribution_coverage(
        shell_concentration=shell_concentration,
        energy_split=energy_split,
    )
    control_volume = _control_volume_evidence(mesh)
    diagnosis = _diagnose_case(
        energy_split=energy_split,
        expected=expected,
        shell_concentration=shell_concentration,
        shell_coverage=shell_coverage,
        control_volume=control_volume,
    )
    return {
        "theta_B": float(theta_b),
        "total_energy": float(result["near_rim"]["total_energy"]),
        "near_rim": {
            key: float(result["near_rim"][key])
            for key in (
                "theta_b",
                "theta_outer_in",
                "theta_outer_out",
                "phi",
                "closure",
                "z_span",
            )
        },
        "tex_at_theta": expected,
        "legacy_numeric_energy_split": legacy_energy_split,
        "numeric_energy_split": energy_split,
        "runtime_module_totals": _runtime_module_totals(result["breakdown"]),
        "runtime_energy_reconciliation": runtime_reconciliation,
        "energy_ratios": {
            "outer_numeric_over_tex": diagnosis["outer_numeric_over_tex"],
            "inner_numeric_over_tex": diagnosis["inner_numeric_over_tex"],
            "contact_numeric_over_tex": diagnosis["contact_numeric_over_tex"],
        },
        "shell_energy_rows": shell_rows,
        "shell_concentration": shell_concentration,
        "shell_attribution_coverage": shell_coverage,
        "control_volume": control_volume,
        "diagnosis": diagnosis,
    }


def _rank_root_causes(cases: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return ranked remaining root causes across selected/theory theta cases."""
    outer_ratios = [
        float(case["energy_ratios"]["outer_numeric_over_tex"]) for case in cases
    ]
    inner_ratios = [
        float(case["energy_ratios"]["inner_numeric_over_tex"]) for case in cases
    ]
    support_fractions = [
        float(case["shell_concentration"]["support_fraction_of_outer_shell_elastic"])
        for case in cases
    ]
    unattributed_fractions = [
        float(case["shell_attribution_coverage"]["unattributed_fraction"])
        for case in cases
    ]
    gap_ratios = [
        float(
            case["control_volume"]["ratios"]["outer_control_over_gap_annulus"]  # type: ignore[index]
        )
        for case in cases
    ]
    return sorted(
        [
            {
                "cause": "excess shared-rim/local-shell elastic cost",
                "rank_score": int(
                    min(95.0, 20.0 + 5.0 * max(outer_ratios))
                    + (20.0 if max(support_fractions) > 0.5 else 0.0)
                ),
                "evidence": {
                    "max_outer_numeric_over_tex": float(max(outer_ratios)),
                    "max_support_fraction": float(max(support_fractions)),
                },
                "recommended_stream": (
                    "write a Feature Contract for shared-rim support energy ownership"
                ),
            },
            {
                "cause": "outer energy attribution mismatch",
                "rank_score": int(80.0 if max(unattributed_fractions) > 0.5 else 35.0),
                "evidence": {
                    "max_unattributed_outer_fraction": float(
                        max(unattributed_fractions)
                    ),
                },
                "recommended_stream": (
                    "audit the disk/outer split helper against runtime module ownership before changing physics"
                ),
            },
            {
                "cause": "excessive shared-rim support control volume",
                "rank_score": int(min(90.0, 20.0 + 10.0 * max(gap_ratios))),
                "evidence": {
                    "max_outer_control_over_gap_annulus": float(max(gap_ratios)),
                },
                "recommended_stream": (
                    "decide whether support-shell or narrow-gap area is the theory-facing control volume"
                ),
            },
            {
                "cause": "inner/outer leaflet elastic imbalance",
                "rank_score": int(85.0 if min(inner_ratios) < 0.25 else 30.0),
                "evidence": {
                    "min_inner_numeric_over_tex": float(min(inner_ratios)),
                    "max_outer_numeric_over_tex": float(max(outer_ratios)),
                },
                "recommended_stream": (
                    "audit leaflet ownership/sign conventions before any energy rescaling"
                ),
            },
            {
                "cause": "residual shape propagation weakness",
                "rank_score": 45,
                "evidence": {
                    "basis": (
                        "this audit explains energy localization; profile/log/K1 "
                        "shape propagation still needs the aggregate benchmark evidence"
                    )
                },
                "recommended_stream": (
                    "keep shape propagation as a separate fix stream unless energy ownership is ruled out"
                ),
            },
        ],
        key=lambda item: int(item["rank_score"]),
        reverse=True,
    )


def run_curved_1disk_energy_control_volume_audit(
    theta_values: Iterable[float] = DEFAULT_THETA_VALUES,
) -> dict[str, object]:
    """Run selected/theory theta energy-control-volume diagnostics."""
    cases = [_run_case(float(theta)) for theta in theta_values]
    return {
        "title": "Curved 1-disk energy/control-volume audit",
        "scope": {
            "diagnosis_only": True,
            "runtime_physics_changed": False,
            "reference": "docs/1_disk_3d.tex",
        },
        "theta_values": [float(case["theta_B"]) for case in cases],
        "cases": cases,
        "root_causes_ranked": _rank_root_causes(cases),
        "recommended_next_pr": {
            "feature_contract_required": True,
            "reason": (
                "Any correction to support-shell energy ownership, control volume, "
                "or leaflet balance changes theory-facing numerical behavior."
            ),
        },
    }


def main() -> None:
    """Run the audit and print JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--theta",
        action="append",
        type=float,
        help="Forced theta_B value to audit. May be supplied more than once.",
    )
    args = parser.parse_args()
    report = run_curved_1disk_energy_control_volume_audit(
        DEFAULT_THETA_VALUES if not args.theta else tuple(args.theta)
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
