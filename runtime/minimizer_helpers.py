"""Helper utilities extracted from ``runtime.minimizer``.

These helpers keep behavior-preserving logic centralized while letting
``Minimizer`` stay focused on optimization flow.
"""

from __future__ import annotations

import logging
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
    relax_tilts_fn: Callable[..., None],
    relax_leaflet_tilts_fn: Callable[..., None],
    set_leaflet_tilts_fast_fn: Callable[[np.ndarray, np.ndarray], None],
    logger_obj: logging.Logger,
) -> Callable[[], float]:
    """Build reduced-energy line-search callback with temporary tilt overrides."""
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

    def energy_fn() -> float:
        tilt_mode = str(gp.get("tilt_solve_mode", "fixed") or "fixed")
        mode_norm = tilt_mode.strip().lower()
        if mode_norm in ("", "none", "off", "false", "fixed"):
            return float(projected_energy_fn())

        positions = mesh.positions_view()
        try:
            gp.set("tilt_inner_steps", reduced_steps)
            gp.set("tilt_coupled_steps", reduced_steps)
            gp.set("tilt_cg_max_iters", reduced_steps)

            if guard_factor > 0.0:
                pre_tin = mesh.tilts_in_view().copy(order="F")
                pre_tout = mesh.tilts_out_view().copy(order="F")
                pre_e = float(compute_energy_fn())

            if uses_leaflet_tilts:
                relax_leaflet_tilts_fn(positions=positions, mode=tilt_mode)
            else:
                relax_tilts_fn(positions=positions, mode=tilt_mode)

            e = float(projected_energy_fn())

            if guard_factor > 0.0:
                threshold = max(guard_min, abs(pre_e) * guard_factor)
                if e > threshold:
                    set_leaflet_tilts_fast_fn(pre_tin, pre_tout)
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

    return energy_fn
