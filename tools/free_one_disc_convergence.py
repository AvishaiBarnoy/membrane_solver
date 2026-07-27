#!/usr/bin/env python3
"""Canonical free-one-disc field and convergence validation.

The lane built here has one radial-only trace ring at ``R + epsilon`` and
otherwise unconstrained refinement rings between the disc and the pre-existing
outer membrane.  Constructed support shells and height constraints are not
used.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml
from scipy.special import i1, k1

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from commands.executor import execute_command_line  # noqa: E402
from runtime.topology import detect_vertex_edge_collisions  # noqa: E402
from tools.diagnostics.curved_1disk_shared_rim_phi_target_audit import (  # noqa: E402
    THEORY_THETA_B,
)
from tools.reproduce_theory_parity import _build_context  # noqa: E402
from tools.theory_parity_interface_profiles import (  # noqa: E402
    build_free_outer_refinement_fixture,
)

DEFAULT_BASE_FIXTURE = (
    ROOT / "tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity.yaml"
)
DEFAULT_PROTOCOL = ("g10", "t2e-3", "g20")


@dataclass(frozen=True)
class FreeOneDiscCase:
    label: str
    trace_epsilon: float
    near_spacing: float
    outer_radius: float
    refinement_passes: int = 0


def _vertex_radius(vertex: Sequence[Any]) -> float:
    return float(np.hypot(float(vertex[0]), float(vertex[1])))


def _group_radii(doc: dict[str, Any], group: str) -> list[float]:
    radii: list[float] = []
    for vertex in doc.get("vertices", []):
        if len(vertex) < 4 or not isinstance(vertex[3], dict):
            continue
        opts = vertex[3]
        if (
            opts.get("rim_slope_match_group") == group
            or opts.get("pin_to_circle_group") == group
        ):
            radii.append(_vertex_radius(vertex))
    return radii


def _rescale_outer_domain(
    doc: dict[str, Any], *, disk_radius: float, outer_radius: float
) -> dict[str, Any]:
    current_outer = max(_vertex_radius(v) for v in doc["vertices"])
    if outer_radius <= disk_radius:
        raise ValueError("outer_radius must exceed the disc radius")
    if np.isclose(current_outer, outer_radius, rtol=0.0, atol=1.0e-12):
        return doc
    scale = (float(outer_radius) - disk_radius) / (current_outer - disk_radius)
    for vertex in doc["vertices"]:
        radius = _vertex_radius(vertex)
        if radius <= disk_radius + 1.0e-12:
            continue
        target = disk_radius + scale * (radius - disk_radius)
        factor = target / radius
        vertex[0] = float(vertex[0]) * factor
        vertex[1] = float(vertex[1]) * factor
        if len(vertex) >= 4 and isinstance(vertex[3], dict):
            opts = vertex[3]
            if opts.get("pin_to_circle_radius") is not None:
                opts["pin_to_circle_radius"] = float(target)
    gp = dict(doc.get("global_parameters") or {})
    if gp.get("pin_to_circle_radius") is not None:
        gp["pin_to_circle_radius"] = float(outer_radius)
    doc["global_parameters"] = gp
    return doc


def shape_regular_free_radii(
    *,
    trace_radius: float,
    first_coarse_radius: float,
    target_spacing: float,
) -> list[float]:
    """Return uniform free-ring radii without a terminal radial sliver."""
    span = float(first_coarse_radius) - float(trace_radius)
    if span <= 0.0:
        raise ValueError("first coarse radius must exceed the trace radius")
    if target_spacing <= 0.0:
        raise ValueError("target_spacing must be positive")
    intervals = max(1, int(np.ceil(span / float(target_spacing))))
    radii = np.linspace(trace_radius, first_coarse_radius, intervals + 1)
    return [float(value) for value in radii[1:-1]]


def build_canonical_free_one_disc_fixture(
    *,
    base_doc: dict[str, Any],
    case: FreeOneDiscCase,
    theta_b: float = THEORY_THETA_B,
    seed_theory: bool = True,
) -> dict[str, Any]:
    """Build a fixed-theta free membrane without constructed support shells."""
    doc = yaml.safe_load(yaml.safe_dump(base_doc, sort_keys=False))
    disk_radii = _group_radii(doc, "disk")
    if not disk_radii:
        raise ValueError("canonical free lane requires a tagged disc boundary")
    disk_radius = float(np.median(disk_radii))
    doc = _rescale_outer_domain(
        doc, disk_radius=disk_radius, outer_radius=float(case.outer_radius)
    )
    outer_definition = dict((doc.get("definitions") or {}).get("outer_rim") or {})
    outer_definition["constraints"] = [
        name
        for name in list(outer_definition.get("constraints") or [])
        if name != "pin_to_plane"
    ]
    outer_definition["pin_to_circle_mode"] = "slide"
    doc["definitions"]["outer_rim"] = outer_definition
    for vertex in doc["vertices"]:
        if len(vertex) < 4 or not isinstance(vertex[3], dict):
            continue
        opts = vertex[3]
        if opts.get("pin_to_circle_group") != "outer":
            continue
        opts["constraints"] = [
            name
            for name in list(opts.get("constraints") or [])
            if name != "pin_to_plane"
        ]
        opts["pin_to_circle_mode"] = "slide"

    membrane_radii = sorted(
        {
            round(_vertex_radius(vertex), 12)
            for vertex in doc["vertices"]
            if _vertex_radius(vertex) > disk_radius + 1.0e-10
        }
    )
    if not membrane_radii:
        raise ValueError("canonical free lane requires an outer membrane")
    first_coarse_radius = float(membrane_radii[0])
    trace_radius = disk_radius + float(case.trace_epsilon)
    free_radii = shape_regular_free_radii(
        trace_radius=trace_radius,
        first_coarse_radius=first_coarse_radius,
        target_spacing=float(case.near_spacing),
    )
    doc = build_free_outer_refinement_fixture(
        base_doc=doc,
        label=str(case.label),
        trace_radius=trace_radius,
        free_radii=free_radii,
        planar_geometry=False,
    )
    gp = dict(doc.get("global_parameters") or {})
    gp["tilt_thetaB_optimize"] = False
    gp["tilt_thetaB_value"] = float(theta_b)
    gp["free_one_disc_validation_lane"] = True
    gp["free_one_disc_trace_epsilon"] = float(case.trace_epsilon)
    gp["free_one_disc_near_spacing"] = float(case.near_spacing)
    gp["free_one_disc_outer_radius"] = float(case.outer_radius)
    gp["free_one_disc_refinement_passes"] = int(case.refinement_passes)
    gp["free_one_disc_theory_seeded"] = bool(seed_theory)
    doc["global_parameters"] = gp
    if seed_theory:
        lam_in = float(
            np.sqrt(
                float(gp.get("tilt_modulus_in") or 0.0)
                / float(gp.get("bending_modulus_in") or 1.0)
            )
        )
        lam_out = float(
            np.sqrt(
                float(gp.get("tilt_modulus_out") or 0.0)
                / float(gp.get("bending_modulus_out") or 1.0)
            )
        )
        half_theta = 0.5 * float(theta_b)
        for vertex in doc["vertices"]:
            radius = _vertex_radius(vertex)
            if len(vertex) < 4 or not isinstance(vertex[3], dict):
                vertex.append({})
            opts = vertex[3]
            radial = np.zeros(3, dtype=float)
            if radius > 1.0e-12:
                radial[:2] = np.asarray(vertex[:2], dtype=float) / radius
            if radius <= disk_radius + 1.0e-10:
                amplitude = (
                    float(theta_b) * i1(lam_in * radius) / i1(lam_in * disk_radius)
                    if radius > 1.0e-12
                    else 0.0
                )
                opts["tilt_in"] = (amplitude * radial).tolist()
                opts["tilt_out"] = [0.0, 0.0, 0.0]
            else:
                amplitude = (
                    half_theta * k1(lam_out * radius) / k1(lam_out * disk_radius)
                )
                opts["tilt_in"] = (amplitude * radial).tolist()
                opts["tilt_out"] = (amplitude * radial).tolist()
                vertex[2] = float(
                    half_theta * disk_radius * np.log(radius / disk_radius)
                )
    return doc


def _shell_profile(
    mesh,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positions = mesh.positions_view()
    tilts_in = mesh.tilts_in_view()
    tilts_out = mesh.tilts_out_view()
    normals = mesh.vertex_normals(positions=positions)
    radii = np.linalg.norm(positions[:, :2], axis=1)
    shell_r: list[float] = []
    shell_z: list[float] = []
    shell_in: list[float] = []
    shell_out: list[float] = []
    shell_leak: list[float] = []
    for radius in sorted({round(float(value), 10) for value in radii if value > 0.0}):
        rows = np.flatnonzero(np.isclose(radii, radius, atol=5.0e-9))
        radial = positions[rows].copy()
        radial[:, 2] = 0.0
        radial /= np.linalg.norm(radial, axis=1)[:, None]
        tangent_radial = (
            radial
            - np.einsum("ij,ij->i", radial, normals[rows])[:, None] * normals[rows]
        )
        tangent_radial /= np.linalg.norm(tangent_radial, axis=1)[:, None]
        tin_r = np.einsum("ij,ij->i", tilts_in[rows], tangent_radial)
        tout_r = np.einsum("ij,ij->i", tilts_out[rows], tangent_radial)
        tin_leak = tilts_in[rows] - tin_r[:, None] * tangent_radial
        tout_leak = tilts_out[rows] - tout_r[:, None] * tangent_radial
        shell_r.append(float(np.median(radii[rows])))
        shell_z.append(float(np.median(positions[rows, 2])))
        shell_in.append(float(np.median(tin_r)))
        shell_out.append(float(np.median(tout_r)))
        shell_leak.append(
            float(
                max(
                    np.median(np.linalg.norm(tin_leak, axis=1)),
                    np.median(np.linalg.norm(tout_leak, axis=1)),
                )
            )
        )
    return tuple(
        np.asarray(values, dtype=float)
        for values in (shell_r, shell_z, shell_in, shell_out, shell_leak)
    )


def _error_metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    residual = np.asarray(actual) - np.asarray(expected)
    expected_norm = float(np.linalg.norm(expected))
    scale = float(np.max(np.abs(expected))) if expected.size else 0.0
    denom = max(expected_norm, 1.0e-30)
    cosine_denom = float(np.linalg.norm(actual) * np.linalg.norm(expected))
    cosine = (
        float(np.dot(actual, expected) / cosine_denom)
        if cosine_denom > 1.0e-30
        else float("nan")
    )
    return {
        "relative_l2": float(np.linalg.norm(residual) / denom),
        "relative_linf": float(np.max(np.abs(residual)) / max(scale, 1.0e-30)),
        "rmse": float(np.sqrt(np.mean(residual * residual))),
        "cosine": cosine,
    }


def fixed_theta_field_agreement(mesh, *, theta_b: float) -> dict[str, Any]:
    """Compare complete axisymmetric fields against the tensionless theory."""
    gp = mesh.global_parameters
    disk_radius = float(gp.get("theory_radius") or 0.0)
    if disk_radius <= 0.0:
        raise ValueError("theory_radius must be set")
    trace_radius = float(gp.get("parity_trace_layer_radius") or disk_radius)
    lam = float(
        np.sqrt(
            float(gp.get("tilt_modulus_out") or 0.0)
            / float(gp.get("bending_modulus_out") or 1.0)
        )
    )
    shell_r, shell_z, shell_in, shell_out, shell_leak = _shell_profile(mesh)
    outer_mask = shell_r >= trace_radius - 1.0e-9
    r = shell_r[outer_mask]
    z = shell_z[outer_mask]
    tin = shell_in[outer_mask]
    tout = shell_out[outer_mask]
    leak = shell_leak[outer_mask]

    # Exclude the fixed far boundary from the continuum profile comparison.
    if r.size > 2:
        r, z, tin, tout, leak = r[:-1], z[:-1], tin[:-1], tout[:-1], leak[:-1]
    phi_r = np.gradient(z, r, edge_order=1)
    phi_boundary = 0.5 * float(theta_b)
    expected_phi = phi_boundary * disk_radius / r
    expected_tilt = phi_boundary * k1(lam * r) / k1(lam * disk_radius)
    expected_z_shape = phi_boundary * disk_radius * np.log(r / disk_radius)
    expected_z = expected_z_shape + float(np.mean(z - expected_z_shape))

    inner_mask = (shell_r > 0.0) & (shell_r <= disk_radius + 1.0e-9)
    r_inner = shell_r[inner_mask]
    tin_inner = shell_in[inner_mask]
    expected_inner = float(theta_b) * i1(lam * r_inner) / i1(lam * disk_radius)
    tilt_scale = max(float(np.max(np.abs(expected_tilt))), 1.0e-30)
    return {
        "theta_B": float(theta_b),
        "lambda": lam,
        "disk_radius": disk_radius,
        "trace_radius": trace_radius,
        "outer_shell_count": int(r.size),
        "inner_shell_count": int(r_inner.size),
        "phi": _error_metrics(phi_r, expected_phi),
        "z": _error_metrics(z, expected_z),
        "t_in_outer": _error_metrics(tin, expected_tilt),
        "t_out_outer": _error_metrics(tout, expected_tilt),
        "t_in_disc": _error_metrics(tin_inner, expected_inner),
        "vector": {
            "max_tangential_leak_relative": float(np.max(leak) / tilt_scale),
            "median_tangential_leak_relative": float(np.median(leak) / tilt_scale),
        },
    }


def default_convergence_cases(*, outer_radius: float = 12.0) -> list[FreeOneDiscCase]:
    """Return radial, topology-stress, and domain convergence families.

    ``r`` refines triangles rather than independently resampling the circular
    rings.  Those cases therefore audit whether generic refinement remains
    admissible; they are not labeled as angular convergence.
    """
    cases = [
        FreeOneDiscCase("radial_h020", 0.02, 0.02, outer_radius),
        FreeOneDiscCase("radial_h010", 0.01, 0.01, outer_radius),
        FreeOneDiscCase("radial_h005", 0.005, 0.005, outer_radius),
        FreeOneDiscCase("topology_refine0", 0.01, 0.01, outer_radius, 0),
        FreeOneDiscCase("topology_refine1", 0.01, 0.01, outer_radius, 1),
        FreeOneDiscCase("domain_r8", 0.01, 0.01, 8.0),
        FreeOneDiscCase("domain_r12", 0.01, 0.01, 12.0),
        FreeOneDiscCase("domain_r16", 0.01, 0.01, 16.0),
    ]
    return cases


def run_case(
    *,
    base_doc: dict[str, Any],
    case: FreeOneDiscCase,
    theta_b: float,
    protocol: Sequence[str],
) -> dict[str, Any]:
    doc = build_canonical_free_one_disc_fixture(
        base_doc=base_doc, case=case, theta_b=theta_b
    )
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as stream:
        yaml.safe_dump(doc, stream, sort_keys=False)
        path = Path(stream.name)
    try:
        context = _build_context(path)
        for _ in range(int(case.refinement_passes)):
            execute_command_line(context, "r")
        initial_collisions = detect_vertex_edge_collisions(
            context.mesh, threshold=1.0e-3
        )
        for command in protocol:
            execute_command_line(context, str(command))
        final_collisions = detect_vertex_edge_collisions(context.mesh, threshold=1.0e-3)
        agreement = fixed_theta_field_agreement(context.mesh, theta_b=theta_b)
        energy = float(context.minimizer.compute_energy())
        projected_energy, projected_gradient = (
            context.minimizer.compute_energy_and_gradient_array()
        )
        return {
            "case": asdict(case),
            "protocol": list(protocol),
            "energy": energy,
            "gradient_energy": float(projected_energy),
            "projected_shape_gradient_norm": float(np.linalg.norm(projected_gradient)),
            "topology": {
                "valid": not initial_collisions and not final_collisions,
                "initial_vertex_edge_collision_count": len(initial_collisions),
                "final_vertex_edge_collision_count": len(final_collisions),
            },
            "agreement": agreement,
        }
    finally:
        path.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE_FIXTURE)
    parser.add_argument("--theta", type=float, default=THEORY_THETA_B)
    parser.add_argument("--protocol", nargs="*", default=list(DEFAULT_PROTOCOL))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    base_doc = yaml.safe_load(args.base.read_text(encoding="utf-8")) or {}
    cases = default_convergence_cases()
    if args.quick:
        cases = [cases[1]]
    report = {
        "meta": {
            "base": str(args.base),
            "theta_B": float(args.theta),
            "protocol": list(args.protocol),
            "constructed_support_shells": False,
            "outer_height_fixed": False,
            "angular_convergence_status": (
                "not implemented: generic triangle refinement is reported only "
                "as a topology stress test"
            ),
        },
        "cases": [
            run_case(
                base_doc=base_doc,
                case=case,
                theta_b=float(args.theta),
                protocol=tuple(args.protocol),
            )
            for case in cases
        ],
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out is None:
        print(text)
    else:
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
