import logging

import numpy as np

from runtime.equiangulation import equiangulate_iteration

logger = logging.getLogger("membrane_solver")


def _triangle_aspect_percentile(minimizer, percentile: float = 90.0) -> float:
    """Return triangle aspect-ratio percentile ``h_max/h_min`` for current mesh."""
    tri_rows, _ = minimizer.mesh.triangle_row_cache()
    if tri_rows is None or len(tri_rows) == 0:
        return float("nan")
    pos = minimizer.mesh.positions_view()
    tri = pos[np.asarray(tri_rows, dtype=int)]
    e01 = np.linalg.norm(tri[:, 0] - tri[:, 1], axis=1)
    e12 = np.linalg.norm(tri[:, 1] - tri[:, 2], axis=1)
    e20 = np.linalg.norm(tri[:, 2] - tri[:, 0], axis=1)
    h_max = np.maximum.reduce([e01, e12, e20])
    h_min = np.minimum.reduce([e01, e12, e20])
    ratio = h_max / np.maximum(h_min, 1e-18)
    return float(np.percentile(ratio, float(percentile)))


def _maybe_auto_mesh_quality_repair(minimizer, *, iteration: int) -> bool:
    """Optionally run bounded mesh-quality repair via equiangulation passes."""
    if not bool(minimizer.global_params.get("mesh_quality_auto_repair_enabled", False)):
        return False
    every = int(minimizer.global_params.get("mesh_quality_auto_repair_every", 0) or 0)
    if every <= 0 or ((int(iteration) + 1) % every) != 0:
        return False

    threshold = float(
        minimizer.global_params.get("mesh_quality_aspect_threshold", 0.0) or 0.0
    )
    if threshold <= 0.0:
        return False
    perc = float(
        minimizer.global_params.get("mesh_quality_aspect_percentile", 90.0) or 90.0
    )
    max_passes = int(
        minimizer.global_params.get("mesh_quality_max_repair_passes", 1) or 1
    )
    if max_passes <= 0:
        return False

    aspect_p = _triangle_aspect_percentile(minimizer, percentile=perc)
    if not np.isfinite(aspect_p) or aspect_p <= threshold:
        return False

    changed_any = False
    for _ in range(max_passes):
        new_mesh, changed = equiangulate_iteration(minimizer.mesh)
        if not bool(changed):
            break
        minimizer.mesh = new_mesh
        minimizer.enforce_constraints_after_mesh_ops(minimizer.mesh)
        minimizer.mesh.project_tilts_to_tangent()
        minimizer.mesh.increment_version()
        changed_any = True
        aspect_p = _triangle_aspect_percentile(minimizer, percentile=perc)
        if not np.isfinite(aspect_p) or aspect_p <= threshold:
            break
    if changed_any:
        reset = getattr(minimizer.stepper, "reset", None)
        if callable(reset):
            reset()
    return changed_any
