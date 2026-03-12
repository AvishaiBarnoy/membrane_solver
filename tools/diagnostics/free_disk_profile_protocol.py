#!/usr/bin/env python3
"""Reusable two-stage protocol for free-disk coupled profile diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from geometry.geom_io import load_data, parse_geometry
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.gradient_descent import GradientDescent

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FREE_DISK_FIXTURE = (
    ROOT / "tests" / "fixtures" / "kozlov_1disk_3d_free_disk_theory_parity.yaml"
)
DEFAULT_FREE_DISK_CURVED_BILAYER_SOURCE = (
    ROOT
    / "meshes"
    / "caveolin"
    / "kozlov_1disk_3d_tensionless_single_leaflet_profile_hard_rim_R12_free_disk.yaml"
)


def _configure_theta_scan(mesh) -> Minimizer:
    gp = mesh.global_parameters
    gp.set("tilt_solve_mode", "coupled")
    gp.set("tilt_solver", "gd")
    gp.set("tilt_step_size", 0.15)
    gp.set("tilt_inner_steps", 10)
    gp.set("tilt_tol", 1e-8)
    gp.set("tilt_kkt_projection_during_relaxation", False)
    gp.set("tilt_thetaB_optimize", True)
    gp.set("tilt_thetaB_optimize_every", 1)
    gp.set("tilt_thetaB_optimize_delta", 0.02)
    gp.set("tilt_thetaB_optimize_inner_steps", 2)
    gp.set("step_size_mode", "fixed")
    gp.set("step_size", 1.0e-3)
    return Minimizer(
        mesh,
        gp,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
        tol=1e-6,
    )


def _configure_shape_relax(mesh, *, theta_b: float) -> Minimizer:
    gp = mesh.global_parameters
    gp.set("tilt_solve_mode", "coupled")
    gp.set("tilt_solver", "gd")
    gp.set("tilt_step_size", 0.15)
    gp.set("tilt_inner_steps", 10)
    gp.set("tilt_tol", 1e-8)
    gp.set("tilt_kkt_projection_during_relaxation", False)
    gp.set("tilt_thetaB_optimize", False)
    gp.set("tilt_thetaB_value", float(theta_b))
    gp.set("step_size_mode", "fixed")
    gp.set("step_size", 0.01)
    return Minimizer(
        mesh,
        gp,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
        tol=1e-6,
    )


def load_free_disk_theory_mesh(path: str | Path | None = None):
    """Return the canonical free-disk theory fixture as a parsed mesh."""
    mesh_path = Path(path) if path is not None else DEFAULT_FREE_DISK_FIXTURE
    return parse_geometry(load_data(str(mesh_path)))


def load_free_disk_curved_bilayer_mesh(path: str | Path | None = None):
    """Return the refined curved free-disk mesh configured for bilayer parity.

    This reuses the refined free-disk geometry with the physical rim at
    ``R = 7/15`` and a resolved first free ring just outside the disk. The
    returned mesh is adjusted in-memory so the outer leaflet is present on the
    disk and the truncated far ring can slide collectively along ``z`` while
    keeping its radius fixed.
    """
    mesh_path = (
        Path(path) if path is not None else DEFAULT_FREE_DISK_CURVED_BILAYER_SOURCE
    )
    mesh = parse_geometry(load_data(str(mesh_path)))
    gp = mesh.global_parameters
    gp.set("leaflet_out_absent_presets", [])

    defs = getattr(mesh, "definitions", None)
    if isinstance(defs, dict):
        outer_def = defs.get("outer_rim")
        if isinstance(outer_def, dict):
            outer_def["pin_to_plane_mode"] = "slide"
            outer_def["pin_to_plane_group"] = "outer_height_gauge"
            outer_def["pin_to_circle_mode"] = "slide"
            outer_def["pin_to_circle_group"] = "outer"
            outer_def["pin_to_circle_normal"] = [0.0, 0.0, 1.0]
            outer_def["pin_to_circle_point"] = [0.0, 0.0, 0.0]

    for vid in mesh.vertex_ids:
        vertex = mesh.vertices[int(vid)]
        opts = getattr(vertex, "options", None) or {}
        if (
            opts.get("preset") == "outer_rim"
            or opts.get("pin_to_circle_group") == "outer"
        ):
            opts["pin_to_plane_mode"] = "slide"
            opts["pin_to_plane_group"] = "outer_height_gauge"
            opts["pin_to_circle_mode"] = "slide"
            opts["pin_to_circle_group"] = "outer"
            opts["pin_to_circle_normal"] = [0.0, 0.0, 1.0]
            opts["pin_to_circle_point"] = [0.0, 0.0, 0.0]
        vertex.options = opts
    return mesh


def optimize_free_disk_theta_b(mesh, *, scans: int = 4) -> float:
    """Optimize the theory scalar drive on the flat free-disk lane."""
    minim = _configure_theta_scan(mesh)
    tilt_mode = str(mesh.global_parameters.get("tilt_solve_mode") or "coupled")
    for i in range(int(scans)):
        minim._relax_leaflet_tilts(
            positions=mesh.positions_view(),
            mode=tilt_mode,
        )
        minim._optimize_thetaB_scalar(tilt_mode=tilt_mode, iteration=i)
    return float(mesh.global_parameters.get("tilt_thetaB_value") or 0.0)


def _physical_rim_and_first_shell_radius(mesh) -> tuple[float, float]:
    """Return the physical rim radius and the first free-shell radius."""
    positions = mesh.positions_view()
    r = np.linalg.norm(positions[:, :2], axis=1)
    rim_rows: list[int] = []
    for vid in mesh.vertex_ids:
        opts = getattr(mesh.vertices[int(vid)], "options", None) or {}
        if opts.get("rim_slope_match_group") == "rim":
            rim_rows.append(mesh.vertex_index_to_row[int(vid)])
    if not rim_rows:
        raise AssertionError("No rim_slope_match_group='rim' rows found")

    # Refined meshes can carry rim-tagged midpoint rows slightly inside the
    # physical shared rim. Use the outer edge of that tagged band so the first
    # selected shell is truly outside the rim.
    rim_radius = float(np.max(r[np.asarray(rim_rows, dtype=int)]))
    shell_radii = np.unique(np.round(r[r > rim_radius + 1.0e-3], 3))
    if shell_radii.size == 0:
        raise AssertionError("No free shell found outside the rim")
    return rim_radius, float(shell_radii[0])


def activate_local_outer_shell(mesh, *, z_bump: float = 1.5e-4) -> float:
    """Tag the first shell outside the rim as the local outer slope ring.

    Returns the activated shell radius.
    """
    positions = mesh.positions_view()
    r = np.linalg.norm(positions[:, :2], axis=1)
    rim_radius, shell_radius = _physical_rim_and_first_shell_radius(mesh)

    shell_rows = np.where(np.isclose(r, shell_radius, atol=1.0e-3))[0]
    if shell_rows.size == 0:
        raise AssertionError("No rows found on the first shell outside the rim")

    for row in shell_rows:
        vid = int(mesh.vertex_ids[int(row)])
        mesh.vertices[vid].options["rim_slope_match_group"] = "outer"
        mesh.vertices[vid].position[2] = float(z_bump)

    mesh.build_position_cache()
    return shell_radius


def configure_free_disk_curved_bilayer_stage2(
    mesh,
    *,
    theta_b: float,
    z_bump: float | None = None,
) -> float:
    """Configure the refined curved bilayer mesh for stage-2 parity checks.

    The physical disk edge remains the ``rim`` group at ``R = 7/15``. The first
    free shell outside that shared rim is tagged as the local ``outer`` ring so
    near-rim parity can be measured at ``R+epsilon``.
    """
    gp = mesh.global_parameters
    gp.set("tilt_thetaB_group_in", "rim")
    gp.set("rim_slope_match_group", "rim")
    gp.set("rim_slope_match_outer_group", "outer")
    gp.set("rim_slope_match_disk_group", "disk")
    gp.set("rim_slope_match_mode", "shared_rim_staggered_v1")
    # In the shared-rim discretization the inner leaflet is first free on the
    # outer support ring, so its base-term boundary should exclude that ring.
    gp.set("bending_tilt_base_term_boundary_group_in", "outer")
    gp.set("bending_tilt_base_term_boundary_group_out", "rim")
    gp.set("rim_slope_match_strength", 0.0)
    gp.set("tilt_thetaB_optimize", False)
    gp.set("tilt_thetaB_value", float(theta_b))
    for vid in mesh.vertex_ids:
        vertex = mesh.vertices[int(vid)]
        opts = getattr(vertex, "options", None) or {}
        if opts.get("rim_slope_match_group") == "outer":
            opts.pop("rim_slope_match_group", None)
        vertex.options = opts
    if z_bump is None:
        rim_radius, shell_radius = _physical_rim_and_first_shell_radius(mesh)
        z_bump = 0.5 * float(theta_b) * float(shell_radius - rim_radius)
    return activate_local_outer_shell(mesh, z_bump=float(z_bump))


def run_free_disk_two_stage_profile_protocol(
    *,
    path: str | Path | None = None,
    theta_scans: int = 4,
    shape_steps: int = 40,
    z_bump: float = 1.5e-4,
):
    """Return ``(mesh, theta_b)`` after the approved two-stage profile protocol."""
    theta_mesh = load_free_disk_theory_mesh(path)
    theta_b = optimize_free_disk_theta_b(theta_mesh, scans=theta_scans)
    if theta_b <= 0.0:
        raise AssertionError("thetaB optimization did not produce a positive drive")

    mesh = load_free_disk_theory_mesh(path)
    activate_local_outer_shell(mesh, z_bump=z_bump)
    minim = _configure_shape_relax(mesh, theta_b=theta_b)
    minim.minimize(n_steps=int(shape_steps))
    return mesh, theta_b


def run_free_disk_curved_bilayer_protocol(
    *,
    theta_scans: int = 4,
    shape_steps: int = 60,
    z_bump: float | None = None,
    theta_path: str | Path | None = None,
    curved_path: str | Path | None = None,
):
    """Return ``(mesh, theta_b)`` for the refined curved bilayer stage-2 lane."""
    theta_mesh = load_free_disk_theory_mesh(theta_path)
    theta_b = optimize_free_disk_theta_b(theta_mesh, scans=theta_scans)
    if theta_b <= 0.0:
        raise AssertionError("thetaB optimization did not produce a positive drive")

    mesh = load_free_disk_curved_bilayer_mesh(curved_path)
    configure_free_disk_curved_bilayer_stage2(mesh, theta_b=theta_b, z_bump=z_bump)
    minim = _configure_shape_relax(mesh, theta_b=theta_b)
    minim.minimize(n_steps=int(shape_steps))
    return mesh, theta_b


def measure_free_disk_curved_bilayer_near_rim(
    mesh,
    *,
    theta_b: float,
    rim_radius: float = 7.0 / 15.0,
) -> dict[str, float]:
    """Return near-rim shared-rim observables for a curved bilayer run."""
    positions = mesh.positions_view()
    r = np.linalg.norm(positions[:, :2], axis=1)
    r_hat = np.zeros_like(positions)
    good = r > 1.0e-12
    r_hat[good, 0] = positions[good, 0] / r[good]
    r_hat[good, 1] = positions[good, 1] / r[good]

    theta_in = np.einsum("ij,ij->i", mesh.tilts_in_view(), r_hat)
    theta_out = np.einsum("ij,ij->i", mesh.tilts_out_view(), r_hat)

    free_radii = sorted(
        {
            round(float(rr), 6)
            for rr in r
            if rr > float(rim_radius) + 1.0e-6 and rr < 12.0 - 1.0e-6
        }
    )
    if not free_radii:
        raise AssertionError("No free ring found outside the physical disk edge")
    ring_r = float(free_radii[0])

    disk_rows = np.where(np.isclose(r, float(rim_radius), atol=1.0e-6))[0]
    outer_rows = np.where(np.isclose(r, ring_r, atol=1.0e-6))[0]
    if disk_rows.size == 0:
        raise AssertionError(
            f"No rows found near the physical disk edge r={rim_radius}"
        )
    if outer_rows.size == 0:
        raise AssertionError(f"No rows found near the first free ring r={ring_r}")

    disk_z = float(np.median(positions[disk_rows, 2]))
    outer_z = float(np.median(positions[outer_rows, 2]))
    dr = float(np.median(r[outer_rows]) - np.median(r[disk_rows]))
    if abs(dr) <= 1.0e-12:
        raise AssertionError("Near-rim slope estimate has zero radial spacing")

    theta_disk = float(np.median(theta_in[disk_rows]))
    theta_outer_in = float(np.median(theta_in[outer_rows]))
    theta_outer_out = float(np.median(theta_out[outer_rows]))
    phi = float((outer_z - disk_z) / dr)
    phi_abs = abs(phi)
    closure = float(theta_outer_in + theta_outer_out)

    return {
        "theta_b": float(theta_b),
        "rim_radius": float(rim_radius),
        "ring_r": ring_r,
        "theta_disk": theta_disk,
        "theta_outer_in": theta_outer_in,
        "theta_outer_out": theta_outer_out,
        "phi": phi,
        "phi_abs": phi_abs,
        "target_half_theta": 0.5 * float(theta_b),
        "closure": closure,
        "closure_error": float(closure - theta_b),
        "theta_out_phi_gap": float(theta_outer_out - phi),
        "phi_deficit": float(theta_b - (2.0 * phi_abs)),
        "z_span": float(np.ptp(positions[:, 2])),
    }


def run_free_disk_curved_bilayer_theta_sweep(
    theta_values: list[float] | tuple[float, ...] | np.ndarray,
    *,
    shape_steps: int = 60,
    z_bump: float | None = None,
    curved_path: str | Path | None = None,
) -> list[dict[str, float]]:
    """Run the curved bilayer stage-2 lane for imposed ``thetaB`` values."""
    rows: list[dict[str, float]] = []
    for theta_b in np.asarray(theta_values, dtype=float):
        mesh = load_free_disk_curved_bilayer_mesh(curved_path)
        configure_free_disk_curved_bilayer_stage2(
            mesh,
            theta_b=float(theta_b),
            z_bump=z_bump,
        )
        minim = _configure_shape_relax(mesh, theta_b=float(theta_b))
        minim.minimize(n_steps=int(shape_steps))
        rows.append(
            measure_free_disk_curved_bilayer_near_rim(mesh, theta_b=float(theta_b))
        )
    return rows


__all__ = [
    "DEFAULT_FREE_DISK_FIXTURE",
    "DEFAULT_FREE_DISK_CURVED_BILAYER_SOURCE",
    "activate_local_outer_shell",
    "configure_free_disk_curved_bilayer_stage2",
    "load_free_disk_curved_bilayer_mesh",
    "load_free_disk_theory_mesh",
    "measure_free_disk_curved_bilayer_near_rim",
    "optimize_free_disk_theta_b",
    "run_free_disk_curved_bilayer_protocol",
    "run_free_disk_curved_bilayer_theta_sweep",
    "run_free_disk_two_stage_profile_protocol",
]
