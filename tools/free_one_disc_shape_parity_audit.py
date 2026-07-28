#!/usr/bin/env python3
"""Audit the remaining free one-disc shape-parity residual."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from commands.executor import execute_command_line  # noqa: E402
from runtime.projections.curved_disk import (  # noqa: E402
    project_curved_free_disk_shape_dofs,
)
from tools.free_one_disc_convergence import (  # noqa: E402
    DEFAULT_BASE_FIXTURE,
    THEORY_THETA_B,
    FreeOneDiscCase,
    build_canonical_free_one_disc_fixture,
)
from tools.free_one_disc_protocol_comparison import (  # noqa: E402
    coupled_stationarity,
    prepare_feasible_state,
)
from tools.reproduce_theory_parity import _build_context  # noqa: E402

DEFAULT_PROTOCOL = ("gd", "g10", "t2e-3", "g20")
DEFAULT_EPSILONS = (1.0e-5, 3.0e-6)


def _invalidate(mesh) -> None:
    mesh._curvature_cache = {}
    mesh._curvature_version = -1
    mesh.increment_version()


def _snapshot(context) -> dict[str, Any]:
    return {
        "positions": context.mesh.positions_view().copy(),
        "tilts_in": context.mesh.tilts_in_view().copy(),
        "tilts_out": context.mesh.tilts_out_view().copy(),
        "global_parameters": context.minimizer.global_params.to_dict().copy(),
    }


def _restore(context, state: dict[str, Any]) -> None:
    mesh = context.mesh
    for row, vertex_id in enumerate(mesh.vertex_ids):
        mesh.vertices[int(vertex_id)].position[:] = state["positions"][row]
    mesh.set_tilts_in_from_array(state["tilts_in"])
    mesh.set_tilts_out_from_array(state["tilts_out"])
    context.minimizer.global_params._params = dict(state["global_parameters"])
    _invalidate(mesh)


def _shape_gradients(context) -> dict[str, np.ndarray | float]:
    minimizer = context.minimizer
    mesh = context.mesh
    positions = mesh.positions_view()
    tilts_in = mesh.tilts_in_view()
    tilts_out = mesh.tilts_out_view()
    minimizer._sync_evaluation_manager()
    energy, raw = minimizer._evaluation_manager.compute_energy_and_gradient_array(
        positions=positions
    )
    raw = np.asarray(raw, dtype=float).copy()
    tilt_in = np.zeros_like(raw)
    tilt_out = np.zeros_like(raw)
    minimizer._compute_energy_and_leaflet_tilt_gradients_array(
        positions=positions,
        tilts_in=tilts_in,
        tilts_out=tilts_out,
        tilt_in_grad_arr=tilt_in,
        tilt_out_grad_arr=tilt_out,
        tilt_only=True,
    )
    projected = raw.copy()
    projected_in = tilt_in.copy()
    projected_out = tilt_out.copy()
    minimizer.constraint_manager.apply_joint_gradient_modifications_array(
        projected,
        projected_in,
        projected_out,
        mesh,
        minimizer.global_params,
        positions=positions,
        tilts_in=tilts_in,
        tilts_out=tilts_out,
    )
    projected[mesh.fixed_mask] = 0.0
    project_curved_free_disk_shape_dofs(mesh, minimizer.global_params, projected)
    return {
        "energy": float(energy),
        "raw": raw,
        "tilt_in": tilt_in,
        "tilt_out": tilt_out,
        "projected": projected,
        "projected_tilt_in": projected_in,
        "projected_tilt_out": projected_out,
    }


def _unit(values: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(values))
    return values / norm if norm > 0.0 else np.zeros_like(values)


def _theory_log_direction(context) -> np.ndarray:
    positions = context.mesh.positions_view()
    radius = float(context.mesh.global_parameters.get("theory_radius"))
    radii = np.linalg.norm(positions[:, :2], axis=1)
    direction = np.zeros_like(positions)
    outer = radii > radius + 1.0e-9
    direction[outer, 2] = radius * np.log(radii[outer] / radius)
    return _unit(direction)


def _evaluate_direction(
    context,
    *,
    state: dict[str, Any],
    direction: np.ndarray,
    epsilon: float,
    enforce: bool,
    relax_tilts: bool,
) -> dict[str, Any]:
    minimizer = context.minimizer
    mesh = context.mesh
    energies: list[float] = []
    breakdowns: list[dict[str, float]] = []
    states: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for sign in (1.0, -1.0):
        _restore(context, state)
        trial = state["positions"] + sign * float(epsilon) * direction
        for row, vertex_id in enumerate(mesh.vertex_ids):
            mesh.vertices[int(vertex_id)].position[:] = trial[row]
        _invalidate(mesh)
        if enforce:
            minimizer._enforce_constraints()
            _invalidate(mesh)
        if relax_tilts:
            tilt_mode = str(
                minimizer.global_params.get("tilt_solve_mode", "fixed") or "fixed"
            )
            minimizer._relax_leaflet_tilts(
                positions=mesh.positions_view(), mode=tilt_mode
            )
        breakdown = minimizer.compute_energy_breakdown()
        breakdowns.append({str(key): float(value) for key, value in breakdown.items()})
        energies.append(float(sum(breakdown.values())))
        states.append(
            (
                mesh.positions_view().copy(),
                mesh.tilts_in_view().copy(),
                mesh.tilts_out_view().copy(),
            )
        )
    tangent = tuple(
        (states[0][index] - states[1][index]) / (2.0 * float(epsilon))
        for index in range(3)
    )
    module_slopes = {
        module: float(
            (breakdowns[0].get(module, 0.0) - breakdowns[1].get(module, 0.0))
            / (2.0 * float(epsilon))
        )
        for module in sorted(set(breakdowns[0]) | set(breakdowns[1]))
    }
    _restore(context, state)
    return {
        "epsilon": float(epsilon),
        "enforce_constraints": bool(enforce),
        "relax_tilts": bool(relax_tilts),
        "fd_slope": float((energies[0] - energies[1]) / (2.0 * float(epsilon))),
        "plus_energy": energies[0],
        "minus_energy": energies[1],
        "position_tangent": tangent[0],
        "tilt_in_tangent": tangent[1],
        "tilt_out_tangent": tangent[2],
        "module_fd_slopes": module_slopes,
    }


def _module_analytic_slopes(
    context, *, state: dict[str, Any], position_tangent: np.ndarray
) -> dict[str, float]:
    minimizer = context.minimizer
    mesh = context.mesh
    slopes: dict[str, float] = {}
    for name, module in zip(minimizer.energy_module_names, minimizer.energy_modules):
        _restore(context, state)
        module_gradient = np.zeros_like(state["positions"])
        kwargs = {
            "positions": state["positions"],
            "index_map": mesh.vertex_index_to_row,
            "grad_arr": module_gradient,
            "tilts_in": state["tilts_in"],
            "tilts_out": state["tilts_out"],
            "tilt_in_grad_arr": None,
            "tilt_out_grad_arr": None,
        }
        try:
            minimizer._evaluation_manager._call_module_array(module, **kwargs)
        except TypeError:
            minimizer._evaluation_manager._call_module_array(
                module,
                positions=state["positions"],
                index_map=mesh.vertex_index_to_row,
                grad_arr=module_gradient,
            )
        scale = float(minimizer._experimental_energy_scale_for_module(str(name)))
        slopes[str(name)] = float(scale * np.sum(module_gradient * position_tangent))
    _restore(context, state)
    return slopes


def _shell_gradient_summary(context, gradient: np.ndarray) -> list[dict[str, Any]]:
    radii = np.linalg.norm(context.mesh.positions_view()[:, :2], axis=1)
    rows = []
    for radius in sorted({round(float(value), 10) for value in radii}):
        mask = np.isclose(radii, radius, atol=5.0e-9)
        rows.append(
            {
                "radius": float(radius),
                "rows": int(np.count_nonzero(mask)),
                "gradient_l2": float(np.linalg.norm(gradient[mask])),
                "median_gradient_z": float(np.median(gradient[mask, 2])),
            }
        )
    return sorted(rows, key=lambda row: row["gradient_l2"], reverse=True)


def run_shape_parity_audit(
    *,
    angular_sectors: int = 12,
    near_spacing: float = 0.01,
    epsilons: Sequence[float] = DEFAULT_EPSILONS,
) -> dict[str, Any]:
    base_doc = yaml.safe_load(DEFAULT_BASE_FIXTURE.read_text(encoding="utf-8"))
    fixture = build_canonical_free_one_disc_fixture(
        base_doc=base_doc,
        case=FreeOneDiscCase(
            "shape_parity_audit",
            trace_epsilon=float(near_spacing),
            near_spacing=float(near_spacing),
            outer_radius=12.0,
            angular_sectors=int(angular_sectors),
        ),
        theta_b=THEORY_THETA_B,
    )
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as stream:
        yaml.safe_dump(fixture, stream, sort_keys=False)
        path = Path(stream.name)
    try:
        context = _build_context(path)
        prepare_feasible_state(context)
        for command in DEFAULT_PROTOCOL:
            execute_command_line(context, command)
        stationarity = coupled_stationarity(context)
        gradients = _shape_gradients(context)
        state = _snapshot(context)
        projected_direction = _unit(-np.asarray(gradients["projected"]))
        log_direction = _theory_log_direction(context)

        projected_probes = []
        for epsilon in epsilons:
            for enforce, relax in (
                (False, False),
                (True, False),
                (True, True),
            ):
                probe = _evaluate_direction(
                    context,
                    state=state,
                    direction=projected_direction,
                    epsilon=float(epsilon),
                    enforce=enforce,
                    relax_tilts=relax,
                )
                raw_dot = float(
                    np.sum(np.asarray(gradients["raw"]) * probe["position_tangent"])
                    + np.sum(
                        np.asarray(gradients["tilt_in"]) * probe["tilt_in_tangent"]
                    )
                    + np.sum(
                        np.asarray(gradients["tilt_out"]) * probe["tilt_out_tangent"]
                    )
                )
                projected_dot = float(
                    np.sum(
                        np.asarray(gradients["projected"]) * probe["position_tangent"]
                    )
                    + np.sum(
                        np.asarray(gradients["projected_tilt_in"])
                        * probe["tilt_in_tangent"]
                    )
                    + np.sum(
                        np.asarray(gradients["projected_tilt_out"])
                        * probe["tilt_out_tangent"]
                    )
                )
                probe["raw_gradient_dot_tangent"] = raw_dot
                probe["projected_gradient_dot_tangent"] = projected_dot
                for key in (
                    "position_tangent",
                    "tilt_in_tangent",
                    "tilt_out_tangent",
                ):
                    probe.pop(key)
                projected_probes.append(probe)

        module_probe = _evaluate_direction(
            context,
            state=state,
            direction=projected_direction,
            epsilon=float(min(epsilons)),
            enforce=True,
            relax_tilts=False,
        )
        module_analytic = _module_analytic_slopes(
            context,
            state=state,
            position_tangent=module_probe["position_tangent"],
        )
        module_rows = {
            name: {
                "fd_slope": float(module_probe["module_fd_slopes"].get(name, 0.0)),
                "analytic_shape_slope": float(module_analytic.get(name, 0.0)),
                "fd_minus_analytic": float(
                    module_probe["module_fd_slopes"].get(name, 0.0)
                    - module_analytic.get(name, 0.0)
                ),
            }
            for name in sorted(
                set(module_probe["module_fd_slopes"]) | set(module_analytic)
            )
        }
        log_probes = [
            _evaluate_direction(
                context,
                state=state,
                direction=log_direction,
                epsilon=float(epsilon),
                enforce=True,
                relax_tilts=True,
            )
            for epsilon in epsilons
        ]
        for probe in log_probes:
            for key in (
                "position_tangent",
                "tilt_in_tangent",
                "tilt_out_tangent",
                "module_fd_slopes",
            ):
                probe.pop(key)
        _restore(context, state)
        return {
            "meta": {
                "angular_sectors": int(angular_sectors),
                "near_spacing": float(near_spacing),
                "epsilons": [float(value) for value in epsilons],
                "protocol": list(DEFAULT_PROTOCOL),
            },
            "stationarity": stationarity,
            "shape_gradient": {
                "raw_l2": float(np.linalg.norm(gradients["raw"])),
                "projected_l2": float(np.linalg.norm(gradients["projected"])),
                "dominant_shells": _shell_gradient_summary(
                    context, np.asarray(gradients["projected"])
                )[:12],
            },
            "projected_descent_probes": projected_probes,
            "module_shape_derivatives": module_rows,
            "theory_log_mode_probes": log_probes,
        }
    finally:
        path.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--angular-sectors", type=int, default=12)
    parser.add_argument("--near-spacing", type=float, default=0.01)
    parser.add_argument("--epsilon", type=float, action="append")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    report = run_shape_parity_audit(
        angular_sectors=int(args.angular_sectors),
        near_spacing=float(args.near_spacing),
        epsilons=tuple(args.epsilon) if args.epsilon else DEFAULT_EPSILONS,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out is None:
        print(text)
    else:
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
