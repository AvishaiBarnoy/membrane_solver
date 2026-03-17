"""Compact curved 3D audit wrapper for the free-z benchmark lane."""

from __future__ import annotations

from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE = (
    ROOT / "tests" / "fixtures" / "kozlov_1disk_3d_free_disk_theory_parity.yaml"
)


def _fixture_label(path: Path | str) -> str:
    """Return a stable fixture label for reports."""

    fixture_path = Path(path)
    if not fixture_path.is_absolute():
        fixture_path = (ROOT / fixture_path).resolve()
    try:
        return str(fixture_path.relative_to(ROOT))
    except ValueError:
        return str(fixture_path)


def _compact_boundary(boundary: dict[str, Any] | None) -> dict[str, Any] | None:
    """Normalize boundary-at-R metrics to a compact smoke-report shape."""

    if boundary is None:
        return None
    return {
        "theory_model": str(boundary.get("theory_model", "unavailable")),
        "available": bool(boundary.get("available", False)),
        "reason": str(boundary.get("reason", "unavailable")),
        "sample_count": int(boundary.get("sample_count", 0)),
        "disk_source": str(boundary.get("disk_source", "unavailable")),
        "rim_source": str(boundary.get("rim_source", "unavailable")),
        "outer_source": str(boundary.get("outer_source", "unavailable")),
        "disk_count": int(boundary.get("disk_count", 0)),
        "rim_count": int(boundary.get("rim_count", 0)),
        "outer_count": int(boundary.get("outer_count", 0)),
        "disk_radius": float(boundary.get("disk_radius", float("nan"))),
        "rim_radius": float(boundary.get("rim_radius", float("nan"))),
        "outer_radius": float(boundary.get("outer_radius", float("nan"))),
        "kink_angle_mesh": float(boundary.get("kink_angle_mesh_median", float("nan"))),
        "kink_angle_theory": float(boundary.get("kink_angle_theory", float("nan"))),
        "kink_angle_factor": float(boundary.get("kink_angle_factor", float("nan"))),
        "tilt_in_mesh": float(boundary.get("tilt_in_mesh_median", float("nan"))),
        "tilt_in_theory": float(boundary.get("tilt_in_theory", float("nan"))),
        "tilt_in_factor": float(boundary.get("tilt_in_factor", float("nan"))),
        "tilt_out_mesh": float(boundary.get("tilt_out_mesh_median", float("nan"))),
        "tilt_out_theory": float(boundary.get("tilt_out_theory", float("nan"))),
        "tilt_out_factor": float(boundary.get("tilt_out_factor", float("nan"))),
    }


def run_flat_disk_curved_3d_audit(
    *,
    fixture: Path | str = DEFAULT_FIXTURE,
    refine_level: int = 2,
    outer_mode: str = "free",
    smoothness_model: str = "splay_twist",
    theta_mode: str = "optimize",
    theta_initial: float = 0.12,
    theta_optimize_steps: int = 8,
    theta_optimize_every: int = 1,
    theta_optimize_delta: float = 0.01,
    theta_optimize_inner_steps: int = 20,
    kappa_physical: float = 10.0,
    kappa_t_physical: float = 10.0,
    length_scale_nm: float = 15.0,
    radius_nm: float = 7.0,
    drive_physical: float = (2.0 / 0.7),
    z_gauge: str = "mean_zero",
    curved_acceptance_profile: str = "fast",
    include_sections: bool = True,
) -> dict[str, Any]:
    """Run the supported curved benchmark lane and emit a compact audit report."""

    from tools.reproduce_flat_disk_one_leaflet import (
        run_flat_disk_one_leaflet_benchmark,
    )

    report = run_flat_disk_one_leaflet_benchmark(
        fixture=fixture,
        refine_level=int(refine_level),
        outer_mode=str(outer_mode),
        geometry_lane="free_z",
        z_gauge=str(z_gauge),
        smoothness_model=str(smoothness_model),
        parameterization="kh_physical",
        theta_mode=str(theta_mode),
        theta_initial=float(theta_initial),
        theta_optimize_steps=int(theta_optimize_steps),
        theta_optimize_every=int(theta_optimize_every),
        theta_optimize_delta=float(theta_optimize_delta),
        theta_optimize_inner_steps=int(theta_optimize_inner_steps),
        kappa_physical=float(kappa_physical),
        kappa_t_physical=float(kappa_t_physical),
        length_scale_nm=float(length_scale_nm),
        radius_nm=float(radius_nm),
        drive_physical=float(drive_physical),
        curved_acceptance_profile=str(curved_acceptance_profile),
    )

    return {
        "meta": {
            "mode": "curved_3d_audit_smoke",
            "fixture": _fixture_label(fixture),
            "geometry_lane": str(
                report["meta"].get(
                    "geometry_lane_requested", report["meta"]["geometry_lane"]
                )
            ),
            "z_gauge": str(
                report["meta"].get("z_gauge_requested", report["meta"]["z_gauge"])
            ),
            "geometry_lane_effective": str(report["meta"]["geometry_lane"]),
            "z_gauge_effective": str(report["meta"]["z_gauge"]),
            "curved_acceptance_profile": str(
                report["meta"].get("curved_acceptance_profile", "full")
            ),
            "theory_source": str(report["meta"]["theory_source"]),
            "theory_model": str(report["meta"]["theory_model"]),
            "sections_requested": bool(include_sections),
            "sections_available": False,
        },
        "parity": {
            "theta_star_mesh": float(report["mesh"]["theta_star"]),
            "theta_star_theory": float(report["theory"]["theta_star"]),
            "theta_factor": float(report["parity"]["theta_factor"]),
            "total_energy_mesh": float(report["mesh"]["total_energy"]),
            "total_energy_theory": float(report["theory"]["total"]),
            "energy_factor": float(report["parity"]["energy_factor"]),
        },
        "curvature": {
            "h_mean": float(report["diagnostics"]["curvature"]["h_mean"]),
            "h_p95": float(report["diagnostics"]["curvature"]["h_p95"]),
            "h_max": float(report["diagnostics"]["curvature"]["h_max"]),
        },
        "boundary_at_R": _compact_boundary(report["parity"].get("boundary_at_R")),
    }
