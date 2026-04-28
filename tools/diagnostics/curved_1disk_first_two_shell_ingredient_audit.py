#!/usr/bin/env python3
# ruff: noqa: E402
"""Audit first-two-shell discrete bending-tilt ingredients on the curved free-disk lane."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geometry.tilt_operators import (
    _resolve_transport_model,
    compute_divergence_from_basis,
)
from modules.energy import bending_tilt_leaflet as _btl
from tools.diagnostics.curved_1disk_theory_benchmark import _run_curved_theta_candidate
from tools.diagnostics.free_disk_profile_protocol import (
    _triangle_region_masks,
    measure_free_disk_curved_bilayer_near_rim,
)

THEORY_THETA_B = 0.1845693593
OUTER_RADIUS = 7.0 / 15.0


def _positions_radii(mesh) -> np.ndarray:
    """Return xy radii for all rows."""
    return np.linalg.norm(mesh.positions_view()[:, :2], axis=1)


def _row_shell_radius_map(mesh) -> np.ndarray:
    """Return rounded shell radius label for every row."""
    radii = _positions_radii(mesh)
    return np.asarray([round(float(v), 6) for v in radii], dtype=float)


def _active_group_labels(mesh, row: int) -> list[str]:
    """Return active group labels for one row."""
    vid = int(mesh.vertex_ids[int(row)])
    opts = getattr(mesh.vertices[vid], "options", None) or {}
    labels: list[str] = []
    for key in (
        "rim_slope_match_group",
        "rim_slope_match_outer_group",
        "rim_slope_match_disk_group",
        "tilt_thetaB_group",
        "tilt_thetaB_group_in",
        "tilt_thetaB_group_out",
        "pin_to_circle_group",
        "pin_to_plane_group",
        "group",
    ):
        val = opts.get(key)
        if val:
            labels.append(f"{key}:{val}")
    return sorted(set(labels))


def _rowwise_radial_tilt(mesh, tilts: np.ndarray) -> np.ndarray:
    """Return radial tilt scalar per row."""
    positions = mesh.positions_view()
    radii = _positions_radii(mesh)
    r_hat = np.zeros_like(positions)
    good = radii > 1.0e-12
    r_hat[good, 0] = positions[good, 0] / radii[good]
    r_hat[good, 1] = positions[good, 1] / radii[good]
    return np.einsum("ij,ij->i", np.asarray(tilts, dtype=float), r_hat)


def _base_term_vertex(
    mesh,
    *,
    cache_tag: str,
    kappa_key: str,
    positions: np.ndarray,
    index_map,
    k_vecs: np.ndarray,
    vertex_areas_vor: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return per-row base-term ingredients used by runtime evaluation."""
    gp = mesh.global_parameters
    model = _btl._energy_model(gp)
    if model != "helfrich":
        model = "helfrich"
    _kappa_arr, c0_arr = _btl._per_vertex_params_leaflet(
        mesh,
        gp,
        model=model,
        kappa_key=kappa_key,
        cache_tag=cache_tag,
    )
    safe_areas_vor = np.maximum(np.asarray(vertex_areas_vor, dtype=float), 1.0e-12)
    k_mag = np.linalg.norm(np.asarray(k_vecs, dtype=float), axis=1)
    h_vor = k_mag / (2.0 * safe_areas_vor)
    is_interior = _btl._interior_mask_leaflet(
        mesh,
        gp,
        cache_tag=cache_tag,
        index_map=index_map,
    )
    base_term = (2.0 * h_vor) - c0_arr
    base_term[~is_interior] = 0.0

    presets = _btl._assume_J0_presets(gp, cache_tag=cache_tag)
    assume_rows = np.zeros(0, dtype=int)
    if presets:
        assume_rows = _btl._collect_preset_rows(
            mesh,
            presets=presets,
            cache_tag=cache_tag,
            index_map=index_map,
            radius_max=_btl._assume_J0_radius_max(gp, cache_tag=cache_tag),
            center_xy=_btl._assume_J0_center_xy(gp),
        )
        if assume_rows.size:
            base_term[assume_rows] = 0.0

    region_rows = _btl._base_term_region_zero_rows(
        mesh,
        gp,
        cache_tag=cache_tag,
        index_map=index_map,
    )
    if region_rows.size:
        base_term[region_rows] = 0.0

    boundary_group = _btl._base_term_boundary_group(gp, cache_tag=cache_tag)
    boundary_rows = (
        _btl._collect_group_rows(mesh, group=boundary_group, index_map=index_map)
        if boundary_group
        else np.zeros(0, dtype=int)
    )
    boundary_mask = np.zeros(len(mesh.vertex_ids), dtype=bool)
    if boundary_rows.size:
        boundary_mask[boundary_rows] = True
    assume_mask = np.zeros(len(mesh.vertex_ids), dtype=bool)
    if assume_rows.size:
        assume_mask[assume_rows] = True
    region_mask = np.zeros(len(mesh.vertex_ids), dtype=bool)
    if region_rows.size:
        region_mask[region_rows] = True
    return {
        "base_term_vertex": np.asarray(base_term, dtype=float),
        "h_vor": np.asarray(h_vor, dtype=float),
        "c0_arr": np.asarray(c0_arr, dtype=float),
        "safe_areas_vor": safe_areas_vor,
        "is_interior": np.asarray(is_interior, dtype=bool),
        "boundary_rows_mask": boundary_mask,
        "assume_rows_mask": assume_mask,
        "region_rows_mask": region_mask,
    }


def _leaflet_runtime_payload(mesh, *, leaflet: str) -> dict[str, object]:
    """Return exact runtime ingredients for one leaflet on the current mesh."""
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row
    gp = mesh.global_parameters
    cache_tag = "out" if str(leaflet) == "out" else "in"
    kappa_key = "bending_modulus_out" if cache_tag == "out" else "bending_modulus_in"
    div_sign_fn = getattr(_btl, "_resolved_div_sign", None)
    if div_sign_fn is None:
        div_sign = 1.0 if cache_tag == "out" else -1.0
    else:
        div_sign = div_sign_fn(
            gp,
            cache_tag=cache_tag,
            default=(1.0 if cache_tag == "out" else -1.0),
        )
    tilts = mesh.tilts_out_view() if cache_tag == "out" else mesh.tilts_in_view()
    payload = _btl._leaflet_triangle_payload(
        mesh,
        gp,
        positions=positions,
        index_map=index_map,
        cache_tag=cache_tag,
    )
    tri_rows = np.asarray(payload["tri_rows"], dtype=np.int32)
    if tri_rows.size == 0:
        raise RuntimeError(f"No triangles available for leaflet {cache_tag}.")
    weights = np.asarray(payload["weights"], dtype=float)
    k_vecs = np.asarray(payload["k_vecs"], dtype=float)
    vertex_areas_vor = np.asarray(payload["vertex_areas_vor"], dtype=float)
    tri_area = np.asarray(payload["tri_area"], dtype=float)
    g0 = np.asarray(payload["g0"], dtype=float)
    g1 = np.asarray(payload["g1"], dtype=float)
    g2 = np.asarray(payload["g2"], dtype=float)
    transport_model = _resolve_transport_model(
        gp.get("tilt_transport_model", "ambient_v1") if gp is not None else "ambient_v1"
    )
    div_raw = compute_divergence_from_basis(
        mesh=mesh,
        tilts=np.asarray(tilts, dtype=float),
        tri_rows=tri_rows,
        g0=g0,
        g1=g1,
        g2=g2,
        positions=positions,
        transport_model=transport_model,
    )
    div_signed = float(div_sign) * div_raw
    div_term = _btl._apply_inner_divergence_update_mode(
        mesh,
        gp,
        positions=positions,
        tri_rows=tri_rows,
        cache_tag=cache_tag,
        div_term=div_signed.copy(),
    )
    if _btl._use_inner_recovered_divergence(gp, cache_tag=cache_tag):
        div_eval, _, _ = _btl._inner_recovered_divergence(
            global_params=gp,
            cache_tag=cache_tag,
            tri_rows=tri_rows,
            tri_area=tri_area,
            div_tri=div_term,
            n_vertices=len(mesh.vertex_ids),
            scratch_tag=f"btl_{cache_tag}",
        )
    else:
        div_eval = np.asarray(div_term, dtype=float)

    vertex_areas_eff, va0_eff, va1_eff, va2_eff = _btl._compute_effective_areas(
        mesh,
        positions,
        tri_rows,
        weights,
        index_map,
        cache_token=f"diag_first_two_shell_{cache_tag}",
    )
    static_payload = _btl._leaflet_static_tilt_payload(
        mesh,
        gp,
        positions=positions,
        index_map=index_map,
        k_vecs=k_vecs,
        vertex_areas_vor=vertex_areas_vor,
        tri_rows=tri_rows,
        kappa_key=kappa_key,
        cache_tag=cache_tag,
    )
    base_tri = np.asarray(static_payload["base_tri"], dtype=float)
    kappa_tri = np.asarray(static_payload["kappa_tri"], dtype=float)
    va_eff = np.stack(
        [
            np.asarray(va0_eff, dtype=float),
            np.asarray(va1_eff, dtype=float),
            np.asarray(va2_eff, dtype=float),
        ],
        axis=1,
    )
    term_tri = base_tri + np.asarray(div_eval, dtype=float)[:, None]
    energy_vertex = 0.5 * kappa_tri * term_tri**2 * va_eff
    outer_mask = np.asarray(
        _triangle_region_masks(mesh, tri_rows)["outer_membrane"], dtype=bool
    )
    row_meta = _base_term_vertex(
        mesh,
        cache_tag=cache_tag,
        kappa_key=kappa_key,
        positions=positions,
        index_map=index_map,
        k_vecs=k_vecs,
        vertex_areas_vor=vertex_areas_vor,
    )
    return {
        "leaflet": cache_tag,
        "tri_rows": tri_rows,
        "weights": weights,
        "tri_area": tri_area,
        "div_raw": np.asarray(div_raw, dtype=float),
        "div_signed": np.asarray(div_signed, dtype=float),
        "div_term": np.asarray(div_term, dtype=float),
        "div_eval": np.asarray(div_eval, dtype=float),
        "base_tri": base_tri,
        "kappa_tri": kappa_tri,
        "va_eff": va_eff,
        "energy_vertex": energy_vertex,
        "outer_mask": outer_mask,
        "row_meta": row_meta,
        "tilt_vectors": np.asarray(tilts, dtype=float),
        "radial_tilt": _rowwise_radial_tilt(mesh, np.asarray(tilts, dtype=float)),
        "vertex_areas_eff": np.asarray(vertex_areas_eff, dtype=float),
        "vertex_areas_vor": np.asarray(vertex_areas_vor, dtype=float),
        "row_shell_radius": _row_shell_radius_map(mesh),
        "row_radii": _positions_radii(mesh),
    }


def _aggregate_row_records(
    mesh, leaflet_payload: dict[str, object]
) -> dict[int, dict[str, object]]:
    """Aggregate exact local contributions per row over outer-membrane triangles."""
    tri_rows = np.asarray(leaflet_payload["tri_rows"], dtype=np.int32)
    outer_mask = np.asarray(leaflet_payload["outer_mask"], dtype=bool)
    weights = np.asarray(leaflet_payload["weights"], dtype=float)
    tri_area = np.asarray(leaflet_payload["tri_area"], dtype=float)
    div_raw = np.asarray(leaflet_payload["div_raw"], dtype=float)
    div_signed = np.asarray(leaflet_payload["div_signed"], dtype=float)
    div_term = np.asarray(leaflet_payload["div_term"], dtype=float)
    div_eval = np.asarray(leaflet_payload["div_eval"], dtype=float)
    base_tri = np.asarray(leaflet_payload["base_tri"], dtype=float)
    kappa_tri = np.asarray(leaflet_payload["kappa_tri"], dtype=float)
    va_eff = np.asarray(leaflet_payload["va_eff"], dtype=float)
    energy_vertex = np.asarray(leaflet_payload["energy_vertex"], dtype=float)
    row_shell_radius = np.asarray(leaflet_payload["row_shell_radius"], dtype=float)
    row_radii = np.asarray(leaflet_payload["row_radii"], dtype=float)
    radial_tilt = np.asarray(leaflet_payload["radial_tilt"], dtype=float)
    tilt_vectors = np.asarray(leaflet_payload["tilt_vectors"], dtype=float)
    vertex_areas_eff = np.asarray(leaflet_payload["vertex_areas_eff"], dtype=float)
    vertex_areas_vor = np.asarray(leaflet_payload["vertex_areas_vor"], dtype=float)
    row_meta = leaflet_payload["row_meta"]

    rows_out: dict[int, dict[str, object]] = {}
    for tri_idx in np.flatnonzero(outer_mask):
        rows = tri_rows[tri_idx]
        for corner in range(3):
            row = int(rows[corner])
            rec = rows_out.setdefault(
                row,
                {
                    "row": row,
                    "row_radius": float(row_radii[row]),
                    "shell_radius": float(row_shell_radius[row]),
                    "tilt_vector": [float(v) for v in tilt_vectors[row]],
                    "radial_tilt": float(radial_tilt[row]),
                    "vertex_area_vor": float(vertex_areas_vor[row]),
                    "vertex_area_eff_total": float(vertex_areas_eff[row]),
                    "base_term_vertex": float(row_meta["base_term_vertex"][row]),
                    "h_vor": float(row_meta["h_vor"][row]),
                    "c0": float(row_meta["c0_arr"][row]),
                    "is_interior": bool(row_meta["is_interior"][row]),
                    "base_term_boundary_zeroed": bool(
                        row_meta["boundary_rows_mask"][row]
                    ),
                    "assume_J0_zeroed": bool(row_meta["assume_rows_mask"][row]),
                    "region_mode_zeroed": bool(row_meta["region_rows_mask"][row]),
                    "group_labels": _active_group_labels(mesh, row),
                    "incident_triangles": [],
                    "neighbor_rows": set(),
                    "neighbor_shell_radii": set(),
                    "local_contribution_sum": 0.0,
                    "effective_area_sum": 0.0,
                    "div_raw_values": [],
                    "div_signed_values": [],
                    "div_term_values": [],
                    "div_eval_values": [],
                    "base_corner_values": [],
                    "term_values": [],
                    "kappa_values": [],
                    "triangle_weight_vectors": [],
                    "triangle_areas": [],
                },
            )
            rec["incident_triangles"].append(int(tri_idx))
            rec["local_contribution_sum"] += float(energy_vertex[tri_idx, corner])
            rec["effective_area_sum"] += float(va_eff[tri_idx, corner])
            rec["div_raw_values"].append(float(div_raw[tri_idx]))
            rec["div_signed_values"].append(float(div_signed[tri_idx]))
            rec["div_term_values"].append(float(div_term[tri_idx]))
            rec["div_eval_values"].append(float(div_eval[tri_idx]))
            rec["base_corner_values"].append(float(base_tri[tri_idx, corner]))
            rec["term_values"].append(
                float(base_tri[tri_idx, corner] + div_eval[tri_idx])
            )
            rec["kappa_values"].append(float(kappa_tri[tri_idx, corner]))
            rec["triangle_weight_vectors"].append([float(v) for v in weights[tri_idx]])
            rec["triangle_areas"].append(float(tri_area[tri_idx]))
            others = [int(v) for j, v in enumerate(rows) if j != corner]
            rec["neighbor_rows"].update(others)
            rec["neighbor_shell_radii"].update(
                float(row_shell_radius[v]) for v in others
            )

    for rec in rows_out.values():
        rec["incident_triangle_count"] = int(len(rec["incident_triangles"]))
        rec["neighbor_rows"] = sorted(int(v) for v in rec["neighbor_rows"])
        rec["neighbor_shell_radii"] = sorted(
            float(v) for v in rec["neighbor_shell_radii"]
        )
        rec["effective_over_vor_ratio"] = float(
            rec["effective_area_sum"] / max(abs(rec["vertex_area_vor"]), 1.0e-12)
        )
        rec["div_raw_median"] = float(
            np.median(np.asarray(rec["div_raw_values"], dtype=float))
        )
        rec["div_signed_median"] = float(
            np.median(np.asarray(rec["div_signed_values"], dtype=float))
        )
        rec["div_term_median"] = float(
            np.median(np.asarray(rec["div_term_values"], dtype=float))
        )
        rec["div_eval_median"] = float(
            np.median(np.asarray(rec["div_eval_values"], dtype=float))
        )
        rec["base_corner_median"] = float(
            np.median(np.asarray(rec["base_corner_values"], dtype=float))
        )
        rec["term_median"] = float(
            np.median(np.asarray(rec["term_values"], dtype=float))
        )
        rec["kappa_median"] = float(
            np.median(np.asarray(rec["kappa_values"], dtype=float))
        )
        rec["triangle_area_median"] = float(
            np.median(np.asarray(rec["triangle_areas"], dtype=float))
        )
    return rows_out


def _select_target_shells(row_records_in: dict[int, dict[str, object]]) -> list[float]:
    """Return the first two outer shells with nonzero inner-leaflet contribution."""
    shell_energy: dict[float, float] = {}
    for rec in row_records_in.values():
        rr = float(rec["shell_radius"])
        if rr <= OUTER_RADIUS + 1.0e-6:
            continue
        shell_energy[rr] = shell_energy.get(rr, 0.0) + float(
            rec["local_contribution_sum"]
        )
    target = [rr for rr in sorted(shell_energy) if abs(shell_energy[rr]) > 1.0e-12][:2]
    if len(target) != 2:
        raise AssertionError("Failed to identify exactly two target outer shells.")
    return target


def _triangle_records_for_shells(
    leaflet_payload: dict[str, object], target_shells: list[float]
) -> dict[float, list[dict[str, object]]]:
    """Return trianglewise local ingredients for triangles touching target-shell rows."""
    tri_rows = np.asarray(leaflet_payload["tri_rows"], dtype=np.int32)
    outer_mask = np.asarray(leaflet_payload["outer_mask"], dtype=bool)
    row_shell_radius = np.asarray(leaflet_payload["row_shell_radius"], dtype=float)
    weights = np.asarray(leaflet_payload["weights"], dtype=float)
    tri_area = np.asarray(leaflet_payload["tri_area"], dtype=float)
    div_raw = np.asarray(leaflet_payload["div_raw"], dtype=float)
    div_signed = np.asarray(leaflet_payload["div_signed"], dtype=float)
    div_term = np.asarray(leaflet_payload["div_term"], dtype=float)
    div_eval = np.asarray(leaflet_payload["div_eval"], dtype=float)
    base_tri = np.asarray(leaflet_payload["base_tri"], dtype=float)
    kappa_tri = np.asarray(leaflet_payload["kappa_tri"], dtype=float)
    va_eff = np.asarray(leaflet_payload["va_eff"], dtype=float)
    energy_vertex = np.asarray(leaflet_payload["energy_vertex"], dtype=float)
    radial_tilt = np.asarray(leaflet_payload["radial_tilt"], dtype=float)

    out = {float(shell): [] for shell in target_shells}
    for tri_idx in np.flatnonzero(outer_mask):
        rows = tri_rows[tri_idx]
        tri_shells = sorted(
            float(v)
            for v in {float(row_shell_radius[int(row)]) for row in rows}
            if float(v) in out
        )
        if not tri_shells:
            continue
        rec = {
            "triangle_index": int(tri_idx),
            "rows": [int(v) for v in rows],
            "row_shell_radii": [float(row_shell_radius[int(v)]) for v in rows],
            "weights": [float(v) for v in weights[tri_idx]],
            "triangle_area": float(tri_area[tri_idx]),
            "div_raw": float(div_raw[tri_idx]),
            "div_signed": float(div_signed[tri_idx]),
            "div_term": float(div_term[tri_idx]),
            "div_eval": float(div_eval[tri_idx]),
            "corners": [
                {
                    "row": int(rows[c]),
                    "shell_radius": float(row_shell_radius[int(rows[c])]),
                    "radial_tilt": float(radial_tilt[int(rows[c])]),
                    "base_term_corner": float(base_tri[tri_idx, c]),
                    "kappa_corner": float(kappa_tri[tri_idx, c]),
                    "effective_area_corner": float(va_eff[tri_idx, c]),
                    "term_corner": float(base_tri[tri_idx, c] + div_eval[tri_idx]),
                    "local_contribution": float(energy_vertex[tri_idx, c]),
                }
                for c in range(3)
            ],
        }
        for shell in tri_shells:
            out[float(shell)].append(rec)
    return out


def _shellwise_summary(
    shell: float,
    *,
    in_rows: list[dict[str, object]],
    out_rows: list[dict[str, object]],
    near_rim: dict[str, float],
) -> dict[str, object]:
    """Return side-by-side shell summary for one target shell."""

    def _agg(rows: list[dict[str, object]]) -> dict[str, float]:
        radial = np.asarray([float(r["radial_tilt"]) for r in rows], dtype=float)
        base = np.asarray([float(r["base_term_vertex"]) for r in rows], dtype=float)
        div_eval = np.asarray([float(r["div_eval_median"]) for r in rows], dtype=float)
        eff_ratio = np.asarray(
            [float(r["effective_over_vor_ratio"]) for r in rows], dtype=float
        )
        energy = np.asarray(
            [float(r["local_contribution_sum"]) for r in rows], dtype=float
        )
        return {
            "row_count": int(len(rows)),
            "theta_median": float(np.median(radial)),
            "base_term_median": float(np.median(base)),
            "div_eval_median": float(np.median(div_eval)),
            "effective_over_vor_ratio_median": float(np.median(eff_ratio)),
            "local_contribution_total": float(np.sum(energy)),
        }

    inner = _agg(in_rows)
    outer = _agg(out_rows)
    return {
        "shell_radius": float(shell),
        "rim_reference": {
            "theta_outer_in": float(near_rim["theta_outer_in"]),
            "theta_outer_out": float(near_rim["theta_outer_out"]),
            "phi": float(near_rim["phi"]),
            "theta_B_half": 0.5 * float(near_rim["theta_b"]),
        },
        "in": inner,
        "out": outer,
        "deltas": {
            "theta_in_minus_rim": float(
                inner["theta_median"] - float(near_rim["theta_outer_in"])
            ),
            "theta_out_minus_rim": float(
                outer["theta_median"] - float(near_rim["theta_outer_out"])
            ),
            "theta_in_minus_out": float(inner["theta_median"] - outer["theta_median"]),
            "base_term_in_minus_out": float(
                inner["base_term_median"] - outer["base_term_median"]
            ),
            "div_eval_in_minus_out": float(
                inner["div_eval_median"] - outer["div_eval_median"]
            ),
            "eff_ratio_in_over_out": float(
                inner["effective_over_vor_ratio_median"]
                / max(abs(outer["effective_over_vor_ratio_median"]), 1.0e-12)
            ),
        },
    }


def _detect_first_departure(shellwise: list[dict[str, object]]) -> dict[str, object]:
    """Return earliest detected mismatch level across the two target shells."""
    departure_level = "combined local expression departure"
    departure_reason = "No earlier isolated level exceeded the comparison heuristics."
    departure_shell = None
    for row in shellwise:
        shell = float(row["shell_radius"])
        rim_in = float(row["rim_reference"]["theta_outer_in"])
        in_theta = float(row["in"]["theta_median"])
        if rim_in != 0.0 and (
            np.sign(in_theta) != np.sign(rim_in) or abs(in_theta) > 1.5 * abs(rim_in)
        ):
            departure_level = "tilt field departure"
            departure_reason = (
                "Inner-shell radial tilt stops smoothly continuing the rim reference."
            )
            departure_shell = shell
            break
        if (
            np.sign(float(row["in"]["div_eval_median"]))
            != np.sign(float(row["out"]["div_eval_median"]))
            or abs(float(row["deltas"]["div_eval_in_minus_out"])) > 0.05
        ):
            departure_level = "divergence/shape-term departure"
            departure_reason = "Base/divergence-side medians separate before normalization can explain the shell energy split."
            departure_shell = shell
            break
        if abs(float(row["deltas"]["eff_ratio_in_over_out"]) - 1.0) > 1.0:
            departure_level = "normalization/area-weight departure"
            departure_reason = "Effective-area normalization differs materially between leaflets on the same shell."
            departure_shell = shell
            break
    return {
        "departure_level": departure_level,
        "departure_shell_radius": None
        if departure_shell is None
        else float(departure_shell),
        "reason": departure_reason,
    }


def run_curved_1disk_first_two_shell_ingredient_audit() -> dict[str, object]:
    """Run the first-two-shell ingredient audit and return a report."""
    result = _run_curved_theta_candidate(THEORY_THETA_B)
    mesh = result["mesh"]
    near_rim = measure_free_disk_curved_bilayer_near_rim(mesh, theta_b=THEORY_THETA_B)
    payload_in = _leaflet_runtime_payload(mesh, leaflet="in")
    payload_out = _leaflet_runtime_payload(mesh, leaflet="out")

    row_records_in = _aggregate_row_records(mesh, payload_in)
    row_records_out = _aggregate_row_records(mesh, payload_out)
    target_shells = _select_target_shells(row_records_in)

    rowwise = {float(shell): {"in": [], "out": []} for shell in target_shells}
    for rec in row_records_in.values():
        shell = float(rec["shell_radius"])
        if shell in rowwise:
            rowwise[shell]["in"].append(rec)
    for rec in row_records_out.values():
        shell = float(rec["shell_radius"])
        if shell in rowwise:
            rowwise[shell]["out"].append(rec)

    shellwise = [
        _shellwise_summary(
            shell,
            in_rows=rowwise[float(shell)]["in"],
            out_rows=rowwise[float(shell)]["out"],
            near_rim=near_rim,
        )
        for shell in target_shells
    ]
    trianglewise = {
        "in": _triangle_records_for_shells(payload_in, target_shells),
        "out": _triangle_records_for_shells(payload_out, target_shells),
    }
    stencil_membership = {
        str(float(shell)): {
            "in": [
                {
                    "row": int(rec["row"]),
                    "incident_triangle_count": int(rec["incident_triangle_count"]),
                    "neighbor_rows": rec["neighbor_rows"],
                    "neighbor_shell_radii": rec["neighbor_shell_radii"],
                    "group_labels": rec["group_labels"],
                }
                for rec in sorted(
                    rowwise[float(shell)]["in"], key=lambda item: int(item["row"])
                )
            ],
            "out": [
                {
                    "row": int(rec["row"]),
                    "incident_triangle_count": int(rec["incident_triangle_count"]),
                    "neighbor_rows": rec["neighbor_rows"],
                    "neighbor_shell_radii": rec["neighbor_shell_radii"],
                    "group_labels": rec["group_labels"],
                }
                for rec in sorted(
                    rowwise[float(shell)]["out"], key=lambda item: int(item["row"])
                )
            ],
        }
        for shell in target_shells
    }
    normalization = {
        str(float(shell)): {
            "in": [
                {
                    "row": int(rec["row"]),
                    "vertex_area_vor": float(rec["vertex_area_vor"]),
                    "vertex_area_eff_total": float(rec["vertex_area_eff_total"]),
                    "effective_area_sum_on_shell_triangles": float(
                        rec["effective_area_sum"]
                    ),
                    "effective_over_vor_ratio": float(rec["effective_over_vor_ratio"]),
                }
                for rec in sorted(
                    rowwise[float(shell)]["in"], key=lambda item: int(item["row"])
                )
            ],
            "out": [
                {
                    "row": int(rec["row"]),
                    "vertex_area_vor": float(rec["vertex_area_vor"]),
                    "vertex_area_eff_total": float(rec["vertex_area_eff_total"]),
                    "effective_area_sum_on_shell_triangles": float(
                        rec["effective_area_sum"]
                    ),
                    "effective_over_vor_ratio": float(rec["effective_over_vor_ratio"]),
                }
                for rec in sorted(
                    rowwise[float(shell)]["out"], key=lambda item: int(item["row"])
                )
            ],
        }
        for shell in target_shells
    }
    first_departure = _detect_first_departure(shellwise)
    return {
        "case": {
            "theta_B": float(THEORY_THETA_B),
            "total_energy": float(sum(float(v) for v in result["breakdown"].values())),
        },
        "shell_selection": {
            "outer_radius": float(OUTER_RADIUS),
            "target_shell_radii": [float(v) for v in target_shells],
            "selection_rule": "first two outer shells with nonzero inner-leaflet outer-membrane local contribution",
        },
        "rim_continuation_reference": {
            key: float(near_rim[key])
            for key in (
                "theta_b",
                "theta_outer_in",
                "theta_outer_out",
                "phi",
                "closure",
                "ring_r",
            )
        },
        "shellwise_comparison": shellwise,
        "rowwise_ingredient_audit": {
            str(float(shell)): {
                "in": sorted(
                    rowwise[float(shell)]["in"], key=lambda item: int(item["row"])
                ),
                "out": sorted(
                    rowwise[float(shell)]["out"], key=lambda item: int(item["row"])
                ),
            }
            for shell in target_shells
        },
        "trianglewise_ingredient_audit": {
            str(float(shell)): {
                "in": trianglewise["in"][float(shell)],
                "out": trianglewise["out"][float(shell)],
            }
            for shell in target_shells
        },
        "stencil_membership_audit": stencil_membership,
        "normalization_audit": normalization,
        "first_departure": first_departure,
        "diagnosis": {
            "call": str(first_departure["departure_level"]),
            "recommended_next_stream": "Next stream should isolate the exact first-two-shell inner-leaflet divergence/base-term assembly that drives the shell-localized mismatch before any broader operator changes.",
        },
    }


def main() -> None:
    """Run the audit and print JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.parse_args()
    report = run_curved_1disk_first_two_shell_ingredient_audit()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
