"""Shared measurement helpers for the flat-disk benchmark and diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np


def _collect_disk_boundary_rows(mesh, *, group: str = "disk") -> np.ndarray:
    rows: list[int] = []
    for vid in mesh.vertex_ids:
        opts = getattr(mesh.vertices[int(vid)], "options", None) or {}
        if (
            opts.get("rim_slope_match_group") == group
            or opts.get("tilt_thetaB_group") == group
            or opts.get("tilt_thetaB_group_in") == group
        ):
            row = mesh.vertex_index_to_row.get(int(vid))
            if row is not None:
                rows.append(int(row))
    out = np.asarray(rows, dtype=int)
    if out.size == 0:
        raise AssertionError(f"Missing or empty disk boundary group: {group!r}")
    return out


def _radial_unit_vectors(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = np.linalg.norm(positions[:, :2], axis=1)
    r_hat = np.zeros_like(positions)
    good = r > 1e-12
    r_hat[good, 0] = positions[good, 0] / r[good]
    r_hat[good, 1] = positions[good, 1] / r[good]
    return r, r_hat


def _factor_difference(measured: float, target: float) -> float:
    t = abs(float(target))
    m = abs(float(measured))
    if t < 1e-18:
        return 1.0 if m < 1e-18 else float("inf")
    ratio = m / t
    if ratio <= 0.0:
        return float("inf")
    return float(max(ratio, 1.0 / ratio))


def _collect_group_rows(
    mesh, *, option_key: str | tuple[str, ...], group: str
) -> np.ndarray:
    """Return mesh rows tagged by one or more vertex option/group pairs."""
    keys = (option_key,) if isinstance(option_key, str) else tuple(option_key)
    rows: list[int] = []
    for vid in mesh.vertex_ids:
        opts = getattr(mesh.vertices[int(vid)], "options", None) or {}
        if not any(opts.get(str(key)) == group for key in keys):
            continue
        row = mesh.vertex_index_to_row.get(int(vid))
        if row is not None:
            rows.append(int(row))
    out = np.asarray(rows, dtype=int)
    if out.size == 0:
        raise AssertionError(f"Missing or empty vertex group for {keys!r}={group!r}")
    return out


def _order_rows_by_angle(positions: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Return row indices sorted by azimuthal angle."""
    from modules.constraints.local_interface_shells import order_rows_by_angle

    return order_rows_by_angle(positions, rows)


def _collect_outer_radial_slope_samples(
    positions: np.ndarray,
    *,
    rim_rows_matched: np.ndarray,
    shell_count: int = 3,
) -> tuple[np.ndarray, np.ndarray, list[float], list[int]]:
    """Collect one-sided outer-shell samples for local radial slope estimation."""
    radii = np.linalg.norm(positions[:, :2], axis=1)
    rim_r = radii[rim_rows_matched]
    phi_rim = np.mod(
        np.arctan2(positions[rim_rows_matched, 1], positions[rim_rows_matched, 0]),
        2.0 * np.pi,
    )
    tol_base = max(1.0e-9, 1.0e-5 * max(1.0, float(np.max(radii))))
    unique_radii = np.unique(np.round(radii, 12))
    shell_radii = [
        float(rv) for rv in unique_radii if float(rv) > float(np.max(rim_r) + tol_base)
    ]
    if len(shell_radii) < max(1, int(shell_count)):
        raise AssertionError("Missing enough non-disk outer shells for slope fit.")

    use_radii = shell_radii[: int(shell_count)]
    sample_r = [rim_r.astype(float)]
    sample_h = [positions[rim_rows_matched, 2].astype(float)]
    used_counts: list[int] = [int(rim_rows_matched.size)]

    for radius in use_radii:
        tol = max(1.0e-9, 1.0e-5 * max(1.0, abs(radius)))
        shell_rows = np.flatnonzero(np.abs(radii - radius) <= tol)
        if shell_rows.size == 0:
            raise AssertionError("Encountered empty outer shell during slope fit.")
        phi_shell = np.mod(
            np.arctan2(positions[shell_rows, 1], positions[shell_rows, 0]), 2.0 * np.pi
        )
        dphi = np.abs(phi_shell[:, None] - phi_rim[None, :])
        dphi = np.minimum(dphi, 2.0 * np.pi - dphi)
        nearest = np.argmin(dphi, axis=0)
        matched_rows = shell_rows[nearest]
        sample_r.append(radii[matched_rows].astype(float))
        sample_h.append(positions[matched_rows, 2].astype(float))
        used_counts.append(int(matched_rows.size))

    r_matrix = np.stack(sample_r, axis=1)
    h_matrix = np.stack(sample_h, axis=1)
    return r_matrix, h_matrix, [float(v) for v in use_radii], used_counts


def _fit_outer_radial_slope_samples(
    positions: np.ndarray,
    *,
    rim_rows_matched: np.ndarray,
    shell_count: int = 3,
    estimator: str = "outer_linear_fit",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Estimate one-sided outer radial slope from the corrected outer-shell family."""
    r_matrix, h_matrix, use_radii, used_counts = _collect_outer_radial_slope_samples(
        positions,
        rim_rows_matched=rim_rows_matched,
        shell_count=shell_count,
    )
    phi = np.zeros(rim_rows_matched.size, dtype=float)
    estimator_mode = str(estimator).strip().lower()
    if estimator_mode == "outer_linear_fit":
        for idx in range(rim_rows_matched.size):
            coeff = np.polyfit(r_matrix[idx], h_matrix[idx], 1)
            phi[idx] = float(coeff[0])
    elif estimator_mode == "outer_multistencil_fd":
        x0 = r_matrix[:, 0]
        dx = r_matrix - x0[:, None]
        for idx in range(rim_rows_matched.size):
            vand = np.vander(dx[idx], N=dx.shape[1], increasing=True).T
            rhs = np.zeros(dx.shape[1], dtype=float)
            if rhs.size > 1:
                rhs[1] = 1.0
            weights = np.linalg.solve(vand, rhs)
            phi[idx] = float(np.dot(weights, h_matrix[idx]))
    else:
        raise ValueError(f"Unsupported outer slope estimator: {estimator}")

    return phi, {
        "outer_slope_estimator": estimator_mode,
        "outer_slope_shell_count": int(len(use_radii)),
        "outer_slope_shell_radii": [float(v) for v in use_radii],
        "outer_slope_sample_counts": [int(v) for v in used_counts],
    }


def _boundary_at_R_parity_metrics(
    mesh,
    *,
    theory_theta_value: float | None,
    outer_slope_estimator: str | None = None,
) -> dict[str, Any]:
    """Compute kink-angle and leaflet-tilt parity at the disk boundary r=R."""
    from modules.constraints.local_interface_shells import (
        build_local_interface_shell_data,
    )

    positions = mesh.positions_view()
    shell_data = build_local_interface_shell_data(mesh, positions=positions)
    disk_rows = shell_data.disk_rows
    rim_rows = shell_data.rim_rows
    outer_rows = shell_data.outer_rows
    rim_rows_matched = shell_data.rim_rows_matched
    disk_rows_matched = shell_data.disk_rows_matched
    r_hat_rim = shell_data.rim_r_hat
    r_hat_disk = shell_data.disk_r_hat
    kink_samples, slope_meta = _fit_outer_radial_slope_samples(
        positions,
        rim_rows_matched=rim_rows_matched,
        shell_count=3,
        estimator=str(
            outer_slope_estimator
            or mesh.global_parameters.get("boundary_outer_slope_estimator")
            or "outer_linear_fit"
        ),
    )
    tilt_in_rim = np.einsum(
        "ij,ij->i", mesh.tilts_in_view()[rim_rows_matched], r_hat_rim
    )
    tilt_out_rim = np.einsum(
        "ij,ij->i", mesh.tilts_out_view()[rim_rows_matched], r_hat_rim
    )
    tilt_in_disk = np.einsum(
        "ij,ij->i", mesh.tilts_in_view()[disk_rows_matched], r_hat_disk
    )

    out: dict[str, Any] = {
        "sample_count": int(outer_rows.size),
        "theory_model": "small_slope_half_split_proxy",
        "disk_source": "disk_boundary_group",
        "rim_source": "first_shell_outside_disk",
        "outer_source": "second_shell_outside_disk",
        "disk_count": int(disk_rows.size),
        "rim_count": int(rim_rows.size),
        "outer_count": int(outer_rows.size),
        "disk_radius": float(shell_data.disk_radius),
        "rim_radius": float(shell_data.rim_radius),
        "outer_radius": float(shell_data.outer_radius),
        "outer_slope_estimator": str(slope_meta["outer_slope_estimator"]),
        "outer_slope_shell_count": int(slope_meta["outer_slope_shell_count"]),
        "outer_slope_shell_radii": list(slope_meta["outer_slope_shell_radii"]),
        "kink_angle_mesh_median": float(np.median(kink_samples)),
        "kink_angle_mesh_mean": float(np.mean(kink_samples)),
        "tilt_in_mesh_median": float(np.median(tilt_in_rim)),
        "tilt_out_mesh_median": float(np.median(tilt_out_rim)),
        "tilt_in_disk_mesh_median": float(np.median(tilt_in_disk)),
        "tilt_out_minus_kink_mesh_median": float(
            np.median(tilt_out_rim - kink_samples)
        ),
        "tilt_in_plus_kink_minus_disk_mesh_median": float(
            np.median(tilt_in_rim + kink_samples - tilt_in_disk)
        ),
    }

    theory_theta = None if theory_theta_value is None else float(theory_theta_value)
    if theory_theta is None or not np.isfinite(theory_theta):
        out["available"] = False
        out["reason"] = "non_finite_theory_theta"
        return out

    half_theta = 0.5 * theory_theta
    out["available"] = True
    out["reason"] = "ok"
    out["kink_angle_theory"] = float(half_theta)
    out["tilt_in_theory"] = float(half_theta)
    out["tilt_out_theory"] = float(half_theta)
    out["kink_angle_factor"] = float(
        _factor_difference(out["kink_angle_mesh_median"], half_theta)
    )
    out["tilt_in_factor"] = float(
        _factor_difference(out["tilt_in_mesh_median"], half_theta)
    )
    out["tilt_out_factor"] = float(
        _factor_difference(out["tilt_out_mesh_median"], half_theta)
    )
    return out


def _profile_metrics(mesh, *, radius: float) -> dict[str, float]:
    positions = mesh.positions_view()
    r, r_hat = _radial_unit_vectors(positions)
    t_in = mesh.tilts_in_view()
    t_in_rad = np.einsum("ij,ij->i", t_in, r_hat)

    inner_mask = r < (0.60 * float(radius))
    rim_mask = (r >= (0.90 * float(radius))) & (r <= (1.10 * float(radius)))
    outer_mask = (r >= (2.00 * float(radius))) & (r <= (4.00 * float(radius)))
    if not np.any(rim_mask):
        raise AssertionError(
            "Rim profile band is empty; cannot evaluate benchmark profile."
        )
    if not np.any(outer_mask):
        raise AssertionError(
            "Outer profile band is empty; cannot evaluate benchmark profile."
        )

    def _median_abs(mask: np.ndarray) -> float:
        if not np.any(mask):
            return float("nan")
        return float(np.median(np.abs(t_in_rad[mask])))

    return {
        "inner_abs_median": _median_abs(inner_mask),
        "rim_abs_median": _median_abs(rim_mask),
        "outer_abs_median": _median_abs(outer_mask),
        "tilt_in_radial_max_abs": float(np.max(np.abs(t_in_rad))),
    }


def _rim_continuity_metrics(
    mesh,
    *,
    radius: float,
) -> dict[str, float]:
    """Compute rim continuity diagnostics by matching nearest angles across r=R."""
    positions = mesh.positions_view()
    r, r_hat = _radial_unit_vectors(positions)
    phi = np.mod(np.arctan2(positions[:, 1], positions[:, 0]), 2.0 * np.pi)
    t_in_rad = np.einsum("ij,ij->i", mesh.tilts_in_view(), r_hat)

    inner_candidates = r < (float(radius) - 1e-12)
    outer_candidates = r > (float(radius) + 1e-12)
    if not np.any(inner_candidates) or not np.any(outer_candidates):
        return {
            "matched_bins": 0,
            "jump_abs_median": float("nan"),
            "jump_abs_max": float("nan"),
            "jump_signed_mean": float("nan"),
        }

    r_in_shell = float(np.max(r[inner_candidates]))
    r_out_shell = float(np.min(r[outer_candidates]))
    tol_in = max(1e-9, 1e-5 * max(1.0, abs(r_in_shell)))
    tol_out = max(1e-9, 1e-5 * max(1.0, abs(r_out_shell)))
    inner_mask = np.abs(r - r_in_shell) <= tol_in
    outer_mask = np.abs(r - r_out_shell) <= tol_out
    inner_rows = np.flatnonzero(inner_mask)
    outer_rows = np.flatnonzero(outer_mask)
    if inner_rows.size == 0 or outer_rows.size == 0:
        return {
            "matched_bins": 0,
            "jump_abs_median": float("nan"),
            "jump_abs_max": float("nan"),
            "jump_signed_mean": float("nan"),
        }

    phi_in = phi[inner_rows]
    phi_out = phi[outer_rows]
    dphi = np.abs(phi_out[:, None] - phi_in[None, :])
    dphi = np.minimum(dphi, 2.0 * np.pi - dphi)
    nearest_in = np.argmin(dphi, axis=1)
    jumps = t_in_rad[outer_rows] - t_in_rad[inner_rows[nearest_in]]
    arr = np.asarray(jumps, dtype=float)
    return {
        "matched_bins": int(arr.size),
        "jump_abs_median": float(np.median(np.abs(arr))),
        "jump_abs_max": float(np.max(np.abs(arr))),
        "jump_signed_mean": float(np.mean(arr)),
    }


def _rim_boundary_realization_metrics(
    mesh,
    *,
    radius: float,
    theta_value: float,
) -> dict[str, float]:
    """Measure realized radial tilt on a rim shell vs imposed theta_B."""
    positions = mesh.positions_view()
    r, r_hat = _radial_unit_vectors(positions)
    shell_tol = max(1e-6, 0.02 * float(radius))
    rim_mask = np.abs(r - float(radius)) <= shell_tol
    rows = np.flatnonzero(rim_mask)
    if rows.size == 0:
        return {
            "rim_samples": 0,
            "rim_theta_error_abs_median": float("nan"),
            "rim_theta_error_abs_max": float("nan"),
            "rim_theta_realized_median": float("nan"),
        }
    t_rad = np.einsum("ij,ij->i", mesh.tilts_in_view()[rows], r_hat[rows])
    err = t_rad - float(theta_value)
    return {
        "rim_samples": int(rows.size),
        "rim_theta_error_abs_median": float(np.median(np.abs(err))),
        "rim_theta_error_abs_max": float(np.max(np.abs(err))),
        "rim_theta_realized_median": float(np.median(t_rad)),
    }


def _leakage_metrics(
    mesh,
    *,
    radius: float,
    lambda_value: float | None = None,
    rim_half_width_lambda: float = 0.0,
    outer_near_width_lambda: float = 0.0,
) -> dict[str, float]:
    """Report azimuthal leakage t_phi relative to radial component t_r."""
    positions = mesh.positions_view()
    r, r_hat = _radial_unit_vectors(positions)
    phi_hat = np.zeros_like(positions)
    good = r > 1e-12
    phi_hat[good, 0] = -positions[good, 1] / r[good]
    phi_hat[good, 1] = positions[good, 0] / r[good]
    t_in = mesh.tilts_in_view()
    t_rad = np.einsum("ij,ij->i", t_in, r_hat)
    t_phi = np.einsum("ij,ij->i", t_in, phi_hat)

    def _ratio(mask: np.ndarray) -> float:
        if not np.any(mask):
            return float("nan")
        num = float(np.median(np.abs(t_phi[mask])))
        den = float(np.median(np.abs(t_rad[mask])))
        return float(num / max(den, 1e-18))

    def _median_abs(values: np.ndarray, mask: np.ndarray) -> float:
        if not np.any(mask):
            return float("nan")
        return float(np.median(np.abs(values[mask])))

    inner_mask = r < float(radius)
    outer_mask = r > float(radius)
    result = {
        "inner_tphi_over_trad_median": _ratio(inner_mask),
        "outer_tphi_over_trad_median": _ratio(outer_mask),
    }
    if lambda_value is None:
        return result

    rim_w = max(0.0, float(rim_half_width_lambda) * float(lambda_value))
    outer_near_w = max(0.0, float(outer_near_width_lambda) * float(lambda_value))
    disk_core_end = max(0.0, float(radius) - rim_w)
    rim_end = max(disk_core_end, float(radius) + rim_w)
    outer_near_end = max(rim_end, float(radius) + outer_near_w)
    bands = {
        "disk_core": r < disk_core_end,
        "rim_band": (r >= disk_core_end) & (r <= rim_end),
        "outer_near": (r > rim_end) & (r <= outer_near_end),
        "outer_far": r > outer_near_end,
    }
    for name, mask in bands.items():
        result[f"{name}_tphi_abs_median"] = _median_abs(t_phi, mask)
        result[f"{name}_trad_abs_median"] = _median_abs(t_rad, mask)
        result[f"{name}_tphi_over_trad_median"] = _ratio(mask)
    return result


def _contact_diagnostics(
    *,
    breakdown: dict[str, float],
    theory,
    radius: float,
) -> dict[str, float]:
    """Return contact energy diagnostics in both absolute and per-length units."""
    perimeter = 2.0 * np.pi * float(radius)
    if perimeter <= 0.0:
        raise ValueError("radius must be > 0 for contact diagnostics.")
    mesh_contact = float(breakdown.get("tilt_thetaB_contact_in", 0.0))
    theory_contact = float(theory.contact)
    return {
        "mesh_contact_energy": mesh_contact,
        "theory_contact_energy": theory_contact,
        "mesh_contact_per_length": float(mesh_contact / perimeter),
        "theory_contact_per_length": float(theory_contact / perimeter),
        "contact_factor": float(_factor_difference(mesh_contact, theory_contact)),
    }


def _flat_benchmark_reference_profiles(
    *,
    parameterization: str,
    smoothness_model: str,
) -> dict[str, str]:
    """Describe the continuum reference families relevant to this benchmark."""
    mode = str(parameterization).strip().lower()
    smooth = str(smoothness_model).strip().lower()
    refs = {
        "continuum_field_model": (
            "vector_field_radial_amplitude"
            if mode == "kh_physical"
            else "scalar_amplitude"
        ),
        "scalar_tex_profile": "I0_inside_K0_outside",
    }
    if mode == "kh_physical":
        refs["combined_reference_profile"] = "I1_inside_K1_outside"
        refs["smoothness_only_reference_profile"] = (
            "r_over_R_inside_R_over_r_outside"
            if smooth == "dirichlet"
            else "not_applicable_for_splay_twist"
        )
    else:
        refs["combined_reference_profile"] = "I0_inside_K0_outside"
        refs["smoothness_only_reference_profile"] = "not_applicable_for_scalar_lane"
    return refs
