"""Helper utilities extracted from ``runtime.minimizer``.

These helpers keep behavior-preserving logic centralized while letting
``Minimizer`` stay focused on optimization flow.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any, Callable

import numpy as np


def capture_diagnostic_state(
    mesh, global_params, *, uses_leaflet_tilts: bool
) -> dict[str, Any]:
    """Capture mutable state that debug diagnostics must not persist."""
    snapshot: dict[str, Any] = {
        "global_params": dict(global_params.to_dict()),
        "tilts": None,
        "tilts_in": None,
        "tilts_out": None,
    }
    if uses_leaflet_tilts:
        snapshot["tilts_in"] = mesh.tilts_in_view().copy(order="F")
        snapshot["tilts_out"] = mesh.tilts_out_view().copy(order="F")
    else:
        snapshot["tilts"] = mesh.tilts_view().copy(order="F")
    return snapshot


def restore_diagnostic_state(mesh, global_params, snapshot: dict[str, Any]) -> None:
    """Restore mutable state after debug diagnostics."""
    if snapshot.get("tilts_in") is not None:
        if not np.array_equal(mesh.tilts_in_view(), snapshot["tilts_in"]):
            mesh.set_tilts_in_from_array(snapshot["tilts_in"])
        if not np.array_equal(mesh.tilts_out_view(), snapshot["tilts_out"]):
            mesh.set_tilts_out_from_array(snapshot["tilts_out"])
    elif snapshot.get("tilts") is not None:
        if not np.array_equal(mesh.tilts_view(), snapshot["tilts"]):
            mesh.set_tilts_from_array(snapshot["tilts"])

    if snapshot.get("global_params") != global_params.to_dict():
        global_params._params = dict(snapshot["global_params"])


def get_cached_tilt_fixed_mask(
    *,
    mesh,
    flag_attr: str,
    cached_mask: np.ndarray | None,
    cached_flags_version: int,
    cached_vertex_version: int,
) -> tuple[np.ndarray, int, int]:
    """Return cached tilt fixed mask (or recompute when mesh versions changed)."""
    flags_version = mesh._tilt_fixed_flags_version
    vertex_version = mesh._vertex_ids_version
    if (
        cached_mask is not None
        and cached_flags_version == flags_version
        and cached_vertex_version == vertex_version
        and len(cached_mask) == len(mesh.vertices)
    ):
        return cached_mask, flags_version, vertex_version

    mask = np.array(
        [
            bool(getattr(mesh.vertices[int(vid)], flag_attr, False))
            for vid in mesh.vertex_ids
        ],
        dtype=bool,
    )
    return mask, flags_version, vertex_version


def build_reduced_line_search_energy_fn(
    *,
    mesh,
    global_params,
    reduced_steps: int,
    uses_leaflet_tilts: bool,
    projected_energy_fn: Callable[[], float],
    compute_energy_fn: Callable[[], float],
    projected_energy_at_positions_fn: Callable[[np.ndarray], float] | None = None,
    compute_energy_at_positions_fn: Callable[[np.ndarray], float] | None = None,
    relax_tilts_fn: Callable[..., None],
    relax_leaflet_tilts_fn: Callable[..., None],
    set_leaflet_tilts_fast_fn: Callable[[np.ndarray, np.ndarray], None],
    logger_obj: logging.Logger,
) -> Callable[[], float]:
    """Build reduced-energy line-search callback with temporary tilt overrides."""
    energy_fn, _ = _build_reduced_line_search_callbacks(
        mesh=mesh,
        global_params=global_params,
        reduced_steps=reduced_steps,
        uses_leaflet_tilts=uses_leaflet_tilts,
        projected_energy_fn=projected_energy_fn,
        compute_energy_fn=compute_energy_fn,
        projected_energy_at_positions_fn=projected_energy_at_positions_fn,
        compute_energy_at_positions_fn=compute_energy_at_positions_fn,
        relax_tilts_fn=relax_tilts_fn,
        relax_leaflet_tilts_fn=relax_leaflet_tilts_fn,
        set_leaflet_tilts_fast_fn=set_leaflet_tilts_fast_fn,
        logger_obj=logger_obj,
    )
    return energy_fn


def build_reduced_line_search_trial_energy_fn(
    *,
    mesh,
    global_params,
    reduced_steps: int,
    uses_leaflet_tilts: bool,
    projected_energy_fn: Callable[[], float],
    compute_energy_fn: Callable[[], float],
    projected_energy_at_positions_fn: Callable[[np.ndarray], float] | None = None,
    compute_energy_at_positions_fn: Callable[[np.ndarray], float] | None = None,
    relax_tilts_fn: Callable[..., None],
    relax_leaflet_tilts_fn: Callable[..., None],
    set_leaflet_tilts_fast_fn: Callable[[np.ndarray, np.ndarray], None],
    logger_obj: logging.Logger,
) -> Callable[[np.ndarray], float] | None:
    """Build reduced-energy trial callback for explicit trial positions."""
    _, trial_energy_fn = _build_reduced_line_search_callbacks(
        mesh=mesh,
        global_params=global_params,
        reduced_steps=reduced_steps,
        uses_leaflet_tilts=uses_leaflet_tilts,
        projected_energy_fn=projected_energy_fn,
        compute_energy_fn=compute_energy_fn,
        projected_energy_at_positions_fn=projected_energy_at_positions_fn,
        compute_energy_at_positions_fn=compute_energy_at_positions_fn,
        relax_tilts_fn=relax_tilts_fn,
        relax_leaflet_tilts_fn=relax_leaflet_tilts_fn,
        set_leaflet_tilts_fast_fn=set_leaflet_tilts_fast_fn,
        logger_obj=logger_obj,
    )
    return trial_energy_fn


def _build_reduced_line_search_callbacks(
    *,
    mesh,
    global_params,
    reduced_steps: int,
    uses_leaflet_tilts: bool,
    projected_energy_fn: Callable[[], float],
    compute_energy_fn: Callable[[], float],
    projected_energy_at_positions_fn: Callable[[np.ndarray], float] | None = None,
    compute_energy_at_positions_fn: Callable[[np.ndarray], float] | None = None,
    relax_tilts_fn: Callable[..., None],
    relax_leaflet_tilts_fn: Callable[..., None],
    set_leaflet_tilts_fast_fn: Callable[[np.ndarray, np.ndarray], None],
    logger_obj: logging.Logger,
) -> tuple[Callable[[], float], Callable[[np.ndarray], float] | None]:
    """Build reduced-energy callbacks for current-state and trial positions."""
    gp = global_params
    override_keys = ("tilt_inner_steps", "tilt_coupled_steps", "tilt_cg_max_iters")
    override_present = {key: (key in gp) for key in override_keys}
    override_old = {key: gp.get(key) for key in override_keys}

    guard_factor = float(gp.get("tilt_relax_energy_guard_factor", 0.0) or 0.0)
    guard_min = float(gp.get("tilt_relax_energy_guard_min", 0.0) or 0.0)

    def _restore_overrides() -> None:
        for key in override_keys:
            if override_present.get(key, False):
                gp.set(key, override_old[key])
            else:
                gp.unset(key)

    def _capture_tilt_state() -> dict[str, np.ndarray]:
        if uses_leaflet_tilts:
            return {
                "tilts_in": mesh.tilts_in_view().copy(order="F"),
                "tilts_out": mesh.tilts_out_view().copy(order="F"),
            }
        return {"tilts": mesh.tilts_view().copy(order="F")}

    def _restore_tilt_state(snapshot: dict[str, np.ndarray]) -> None:
        if "tilts_in" in snapshot and "tilts_out" in snapshot:
            set_leaflet_tilts_fast_fn(snapshot["tilts_in"], snapshot["tilts_out"])
            return
        if "tilts" in snapshot:
            mesh.set_tilts_from_array(snapshot["tilts"])

    def _evaluate(positions: np.ndarray | None) -> float:
        tilt_mode = str(gp.get("tilt_solve_mode", "fixed") or "fixed")
        mode_norm = tilt_mode.strip().lower()
        if positions is None:
            positions_arg = mesh.positions_view()
            projected_eval = projected_energy_fn
            raw_eval = compute_energy_fn
            cache_scope = nullcontext()
        else:
            if (
                projected_energy_at_positions_fn is None
                or compute_energy_at_positions_fn is None
            ):
                raise ValueError(
                    "Reduced trial-energy callback requires explicit-position evaluators."
                )
            positions_arg = np.asarray(positions, dtype=float)

            def projected_eval() -> float:
                return float(projected_energy_at_positions_fn(positions_arg))

            def raw_eval() -> float:
                return float(compute_energy_at_positions_fn(positions_arg))

            cache_scope = mesh.geometry_freeze(positions_arg)

        with cache_scope:
            if mode_norm in ("", "none", "off", "false", "fixed"):
                return float(projected_eval())

            try:
                gp.set("tilt_inner_steps", reduced_steps)
                gp.set("tilt_coupled_steps", reduced_steps)
                gp.set("tilt_cg_max_iters", reduced_steps)

                tilt_snapshot = None
                pre_e = None
                if guard_factor > 0.0:
                    tilt_snapshot = _capture_tilt_state()
                    pre_e = float(raw_eval())

                if uses_leaflet_tilts:
                    relax_leaflet_tilts_fn(positions=positions_arg, mode=tilt_mode)
                else:
                    relax_tilts_fn(positions=positions_arg, mode=tilt_mode)

                e = float(projected_eval())

                if guard_factor > 0.0:
                    assert pre_e is not None
                    threshold = max(guard_min, abs(pre_e) * guard_factor)
                    if e > threshold:
                        assert tilt_snapshot is not None
                        _restore_tilt_state(tilt_snapshot)
                        if logger_obj.isEnabledFor(logging.DEBUG):
                            logger_obj.debug(
                                "Line-search tilt guard: E %.6g -> %.6g "
                                "(threshold %.6g); rolling back tilts.",
                                pre_e,
                                e,
                                threshold,
                            )
                        return float(pre_e)

                return e
            finally:
                _restore_overrides()

    def energy_fn() -> float:
        return _evaluate(None)

    trial_energy_fn = None
    if (
        projected_energy_at_positions_fn is not None
        and compute_energy_at_positions_fn is not None
    ):

        def _trial_energy_fn(positions: np.ndarray) -> float:
            return _evaluate(positions)

        trial_energy_fn = _trial_energy_fn

    return energy_fn, trial_energy_fn
