#!/usr/bin/env python3
"""Compare fixed-mesh minimization protocols for the free one-disc lane."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from commands.executor import execute_command_line  # noqa: E402
from tools.free_one_disc_convergence import (  # noqa: E402
    DEFAULT_BASE_FIXTURE,
    THEORY_THETA_B,
    FreeOneDiscCase,
    build_canonical_free_one_disc_fixture,
    fixed_theta_field_agreement,
)
from tools.reproduce_theory_parity import _build_context  # noqa: E402


@dataclass(frozen=True)
class MinimizationProtocol:
    label: str
    commands: tuple[str, ...]


DEFAULT_PROTOCOLS = (
    MinimizationProtocol("gd_fixed_reference", ("gd", "g10", "t2e-3", "g20")),
    MinimizationProtocol(
        "gd_finalize_each_step", ("gd", "tf", *tuple("g1" for _ in range(30)))
    ),
    MinimizationProtocol("gd_adaptive", ("gd", "tf", "g50")),
    MinimizationProtocol("cg_adaptive", ("cg", "tf", "g50")),
    MinimizationProtocol("bfgs_adaptive", ("bfgs", "tf", "g30")),
    MinimizationProtocol("hessian_hybrid", ("gd", "tf", "g10", "hessian 5", "g20")),
)


def _block_metrics(array: np.ndarray) -> dict[str, float]:
    flat = np.asarray(array, dtype=float).reshape(-1)
    return {
        "l2": float(np.linalg.norm(flat)),
        "rms": float(np.sqrt(np.mean(flat * flat))) if flat.size else 0.0,
        "linf": float(np.max(np.abs(flat))) if flat.size else 0.0,
    }


def coupled_stationarity(context) -> dict[str, Any]:
    """Return the projected shape and leaflet-tilt residual blocks."""
    minimizer = context.minimizer
    mesh = context.mesh
    positions = mesh.positions_view()
    tilts_in = mesh.tilts_in_view()
    tilts_out = mesh.tilts_out_view()
    minimizer._sync_evaluation_manager()
    energy, shape_grad = (
        minimizer._evaluation_manager.compute_energy_and_gradient_array(
            positions=positions
        )
    )
    tilt_in_grad = np.zeros_like(shape_grad)
    tilt_out_grad = np.zeros_like(shape_grad)
    tilt_energy = minimizer._compute_energy_and_leaflet_tilt_gradients_array(
        positions=positions,
        tilts_in=tilts_in,
        tilts_out=tilts_out,
        tilt_in_grad_arr=tilt_in_grad,
        tilt_out_grad_arr=tilt_out_grad,
        tilt_only=True,
    )
    if hasattr(
        minimizer.constraint_manager, "apply_joint_gradient_modifications_array"
    ):
        minimizer.constraint_manager.apply_joint_gradient_modifications_array(
            shape_grad,
            tilt_in_grad,
            tilt_out_grad,
            mesh,
            minimizer.global_params,
            positions=positions,
            tilts_in=tilts_in,
            tilts_out=tilts_out,
        )
    else:
        minimizer.constraint_manager.apply_gradient_modifications_array(
            shape_grad, mesh, minimizer.global_params
        )
        minimizer.constraint_manager.apply_tilt_gradient_modifications_array(
            tilt_in_grad,
            tilt_out_grad,
            mesh,
            minimizer.global_params,
            positions=positions,
            tilts_in=tilts_in,
            tilts_out=tilts_out,
        )
    shape_grad[mesh.fixed_mask] = 0.0
    tilt_in_grad[minimizer._tilt_fixed_mask_in()] = 0.0
    tilt_out_grad[minimizer._tilt_fixed_mask_out()] = 0.0
    combined = np.concatenate(
        (shape_grad.reshape(-1), tilt_in_grad.reshape(-1), tilt_out_grad.reshape(-1))
    )
    result = {
        "energy": float(energy),
        "tilt_energy_crosscheck": float(tilt_energy),
        "shape": _block_metrics(shape_grad),
        "tilt_in": _block_metrics(tilt_in_grad),
        "tilt_out": _block_metrics(tilt_out_grad),
        "combined": _block_metrics(combined),
    }
    # Gradient evaluation populates geometry/curvature caches. Diagnostics must
    # not make the next relaxation depend on whether an audit was requested.
    mesh._curvature_cache = {}
    mesh._curvature_version = -1
    return result


def prepare_feasible_state(context) -> None:
    """Project the parsed seed onto constraints and relax tilts once."""
    minimizer = context.minimizer
    minimizer._enforce_constraints()
    context.mesh.increment_version()
    tilt_mode = str(minimizer.global_params.get("tilt_solve_mode", "fixed") or "fixed")
    minimizer._relax_leaflet_tilts(
        positions=context.mesh.positions_view(), mode=tilt_mode
    )


def run_protocol(
    *,
    fixture: dict[str, Any],
    protocol: MinimizationProtocol,
    theta_b: float,
) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as stream:
        yaml.safe_dump(fixture, stream, sort_keys=False)
        path = Path(stream.name)
    try:
        context = _build_context(path)
        prepare_feasible_state(context)
        initial = coupled_stationarity(context)
        checkpoints = []
        for command in protocol.commands:
            execute_command_line(context, command)
            checkpoints.append(
                {
                    "command": command,
                    "stationarity": coupled_stationarity(context),
                }
            )
        return {
            "label": protocol.label,
            "commands": list(protocol.commands),
            "initial_stationarity": initial,
            "checkpoints": checkpoints,
            "final_stationarity": coupled_stationarity(context),
            "agreement": fixed_theta_field_agreement(
                context.mesh, theta_b=float(theta_b)
            ),
        }
    finally:
        path.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE_FIXTURE)
    parser.add_argument("--theta", type=float, default=THEORY_THETA_B)
    parser.add_argument("--angular-sectors", type=int, default=24)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    base_doc = yaml.safe_load(args.base.read_text(encoding="utf-8")) or {}
    case = FreeOneDiscCase(
        "protocol_comparison",
        trace_epsilon=0.01,
        near_spacing=0.01,
        outer_radius=12.0,
        angular_sectors=int(args.angular_sectors),
    )
    fixture = build_canonical_free_one_disc_fixture(
        base_doc=base_doc, case=case, theta_b=float(args.theta)
    )
    protocols = DEFAULT_PROTOCOLS[:2] if args.quick else DEFAULT_PROTOCOLS
    report = {
        "meta": {
            "base": str(args.base),
            "theta_B": float(args.theta),
            "case": case.__dict__,
            "identical_initial_fixture": True,
            "mesh_operations_allowed": False,
        },
        "protocols": [
            run_protocol(
                fixture=fixture,
                protocol=protocol,
                theta_b=float(args.theta),
            )
            for protocol in protocols
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
