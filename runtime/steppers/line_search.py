import logging
from typing import Callable, Dict

import numpy as np

from geometry.entities import Mesh
from runtime.topology import check_max_normal_change, get_min_edge_length

logger = logging.getLogger("membrane_solver")


def backtracking_line_search(
    mesh: Mesh,
    direction: Dict[int, np.ndarray],
    gradient: Dict[int, np.ndarray],
    step_size: float,
    energy_fn: Callable[[], float],
    max_iter: int = 10,
    beta: float = 0.7,
    c: float = 1e-4,
    gamma: float = 1.5,
    alpha_max_factor: float = 10.0,
    constraint_enforcer: Callable[[Mesh], None] | None = None,
) -> tuple[bool, float, float]:
    """Armijo backtracking line search with optional volume guard.

    Parameters
    ----------
    mesh : Mesh
        Mesh being optimized.
    direction : Dict[int, np.ndarray]
        Descent direction for each vertex index.
    gradient : Dict[int, np.ndarray]
        Current gradient at each vertex index.
    step_size : float
        Initial step size to try.
    energy_fn : Callable[[], float]
        Function returning current energy of the mesh.
    max_iter : int, optional
        Maximum number of backtracking iterations, by default ``10``.
    beta : float, optional
        Step size reduction factor, by default ``0.5``.
    c : float, optional
        Armijo condition parameter, by default ``1e-4``.
    gamma : float, optional
        Step size growth factor on success, by default ``1.2``.
    alpha_max_factor : float, optional
        Maximum allowed multiplier for ``step_size`` on success, by default ``10.0``.

    Returns
    -------
    tuple[bool, float]
        Whether the step succeeded and the updated step size.
    """
    original_positions = {vidx: v.position.copy() for vidx, v in mesh.vertices.items()}
    energy0 = energy_fn()
    reduced_tilts = bool(getattr(mesh, "_line_search_reduced_energy", False))
    accept_rule = (
        str(getattr(mesh, "_line_search_reduced_accept_rule", "armijo") or "armijo")
        .strip()
        .lower()
    )
    # If reduced-energy is enabled, energy_fn may relax tilts as part of the
    # baseline evaluation. Snapshot tilts *after* energy0 so all subsequent
    # rejects restore the same state used for Armijo comparisons.
    original_tilts = mesh.tilts_view().copy(order="F")
    original_tilts_in = mesh.tilts_in_view().copy(order="F")
    original_tilts_out = mesh.tilts_out_view().copy(order="F")
    _ = (reduced_tilts, accept_rule)

    # Pre-compute stability threshold
    min_edge_len = get_min_edge_length(mesh)
    safe_step_limit = 0.3 * min_edge_len if min_edge_len > 0 else float("inf")

    # Calculate max possible displacement vector magnitude (unscaled)
    max_dir_norm = 0.0
    if direction:
        # Optimization: convert direction dict values to array once if needed,
        # but max loop is fast enough for now
        max_dir_norm = max(np.linalg.norm(d) for d in direction.values())

    g_dot_d = sum(np.dot(gradient[vidx], direction[vidx]) for vidx in direction)
    if logger.isEnabledFor(logging.DEBUG):
        gnorm = float(
            np.sqrt(sum(np.dot(gradient[vidx], gradient[vidx]) for vidx in direction))
        )
        dnorm = float(
            np.sqrt(sum(np.dot(direction[vidx], direction[vidx]) for vidx in direction))
        )
        logger.debug(
            "Line search stats: g_dot_d=%.6e g_norm=%.6e d_norm=%.6e",
            g_dot_d,
            gnorm,
            dnorm,
        )

    if reduced_tilts and accept_rule not in ("armijo", "decrease_only"):
        raise ValueError(f"Unknown reduced-energy accept rule: {accept_rule!r}")

    if (not reduced_tilts) or accept_rule == "armijo":
        if g_dot_d >= 0:
            logger.debug("Non-descent direction provided; skipping step.")
            return False, step_size, float(energy0)

    alpha = step_size
    alpha_max = alpha_max_factor * step_size

    backtracks = 0
    for _ in range(max_iter):
        # Heuristic: Check if step is small enough to be unconditionally safe
        max_disp = alpha * max_dir_norm
        is_safe_small_step = max_disp < safe_step_limit

        # Trial step from original positions.
        for vidx, vertex in mesh.vertices.items():
            if getattr(vertex, "fixed", False):
                continue
            disp = alpha * direction.get(vidx, np.zeros(3))
            vertex.position[:] = original_positions[vidx] + disp
        mesh.increment_version()

        # Stability Check: Only run expensive check if step is large
        if not is_safe_small_step:
            if not check_max_normal_change(mesh, original_positions):
                # Treat as failure -> backtrack
                for vidx, vertex in mesh.vertices.items():
                    vertex.position[:] = original_positions[vidx]
                mesh.set_tilts_from_array(original_tilts)
                mesh.set_tilts_in_from_array(original_tilts_in)
                mesh.set_tilts_out_from_array(original_tilts_out)
                mesh.increment_version()

                alpha *= beta
                backtracks += 1
                if alpha < 1e-8:
                    break
                continue

        # Enforce constraints (e.g. volume) BEFORE checking energy
        if constraint_enforcer is not None:
            constraint_enforcer(mesh)
            mesh.increment_version()

        trial_energy = energy_fn()
        if reduced_tilts and accept_rule == "decrease_only":
            accept = trial_energy <= energy0
        else:
            accept = trial_energy <= energy0 + c * alpha * g_dot_d

        if accept:
            logger.debug(
                "Line search success: alpha=%.3e, backtracks=%d, "
                "E0=%.6f, Etrial=%.6f, armijo=%s",
                alpha,
                backtracks,
                energy0,
                trial_energy,
                accept,
            )
            new_step = min(alpha * gamma, alpha_max)
            return True, new_step, float(trial_energy)

        # Reject this scale: restore and try a smaller one.
        for vidx, vertex in mesh.vertices.items():
            vertex.position[:] = original_positions[vidx]
        mesh.set_tilts_from_array(original_tilts)
        mesh.set_tilts_in_from_array(original_tilts_in)
        mesh.set_tilts_out_from_array(original_tilts_out)
        mesh.increment_version()

        alpha *= beta
        backtracks += 1

        # If the trial step size becomes too small, further reductions are
        # unlikely to produce meaningful changes in geometry, so bail out.
        if alpha < 1e-8:
            break

    logger.debug(
        "Line search failed after %d backtracks; reverting positions and shrinking step size.",
        backtracks,
    )
    for vidx, vertex in mesh.vertices.items():
        vertex.position[:] = original_positions[vidx]
    mesh.set_tilts_from_array(original_tilts)
    mesh.set_tilts_in_from_array(original_tilts_in)
    mesh.set_tilts_out_from_array(original_tilts_out)
    mesh.increment_version()

    logger.debug(
        "Zero-step detected: no trial step reduced energy (alpha reached %.2e).",
        alpha,
    )
    reduced_step = max(alpha * beta, 0.0)
    return False, max(reduced_step, step_size * beta), float(energy0)


def backtracking_line_search_array(
    mesh: Mesh,
    direction: np.ndarray,
    gradient: np.ndarray,
    step_size: float,
    energy_fn: Callable[[], float],
    vertex_ids,
    max_iter: int = 10,
    beta: float = 0.7,
    c: float = 1e-4,
    gamma: float = 1.5,
    alpha_max_factor: float = 10.0,
    constraint_enforcer: Callable[[Mesh], None] | None = None,
) -> tuple[bool, float, float]:
    """Armijo backtracking line search for dense array gradients."""
    if direction.shape != gradient.shape:
        raise ValueError("direction and gradient must have matching shapes")

    movable_rows = []
    original_positions = {}
    for row, vidx in enumerate(vertex_ids):
        vertex = mesh.vertices[vidx]
        if not getattr(vertex, "fixed", False):
            movable_rows.append(row)
        original_positions[vidx] = vertex.position.copy()
    energy0 = energy_fn()
    reduced_tilts = bool(getattr(mesh, "_line_search_reduced_energy", False))
    accept_rule = (
        str(getattr(mesh, "_line_search_reduced_accept_rule", "armijo") or "armijo")
        .strip()
        .lower()
    )
    original_tilts = mesh.tilts_view().copy(order="F")
    original_tilts_in = mesh.tilts_in_view().copy(order="F")
    original_tilts_out = mesh.tilts_out_view().copy(order="F")
    _ = (reduced_tilts, accept_rule)

    min_edge_len = get_min_edge_length(mesh)
    safe_step_limit = 0.3 * min_edge_len if min_edge_len > 0 else float("inf")

    max_dir_norm = 0.0
    if movable_rows:
        max_dir_norm = float(np.max(np.linalg.norm(direction[movable_rows], axis=1)))

    g_dot_d = float(np.sum(gradient * direction))
    if reduced_tilts and accept_rule not in ("armijo", "decrease_only"):
        raise ValueError(f"Unknown reduced-energy accept rule: {accept_rule!r}")

    if (not reduced_tilts) or accept_rule == "armijo":
        if g_dot_d >= 0.0:
            logger.debug("Non-descent direction provided; skipping step.")
            return False, step_size, float(energy0)

    alpha = step_size
    alpha_max = alpha_max_factor * step_size
    backtracks = 0

    for _ in range(max_iter):
        max_disp = alpha * max_dir_norm
        is_safe_small_step = max_disp < safe_step_limit

        for row in movable_rows:
            vidx = vertex_ids[row]
            vertex = mesh.vertices[vidx]
            disp = alpha * direction[row]
            vertex.position[:] = original_positions[vidx] + disp
        mesh.increment_version()

        if not is_safe_small_step:
            if not check_max_normal_change(mesh, original_positions):
                for vidx, pos in original_positions.items():
                    mesh.vertices[vidx].position[:] = pos
                mesh.set_tilts_from_array(original_tilts)
                mesh.set_tilts_in_from_array(original_tilts_in)
                mesh.set_tilts_out_from_array(original_tilts_out)
                mesh.increment_version()
                alpha *= beta
                backtracks += 1
                if alpha < 1e-8:
                    break
                continue

        if constraint_enforcer is not None:
            constraint_enforcer(mesh)
            mesh.increment_version()

        trial_energy = energy_fn()
        if reduced_tilts and accept_rule == "decrease_only":
            accept = trial_energy <= energy0
        else:
            accept = trial_energy <= energy0 + c * alpha * g_dot_d

        if accept:
            logger.debug(
                "Line search success: alpha=%.3e, backtracks=%d, "
                "E0=%.6f, Etrial=%.6f, armijo=%s",
                alpha,
                backtracks,
                energy0,
                trial_energy,
                accept,
            )
            new_step = min(alpha * gamma, alpha_max)
            return True, new_step, float(trial_energy)

        for vidx, pos in original_positions.items():
            mesh.vertices[vidx].position[:] = pos
        mesh.set_tilts_from_array(original_tilts)
        mesh.set_tilts_in_from_array(original_tilts_in)
        mesh.set_tilts_out_from_array(original_tilts_out)
        mesh.increment_version()

        alpha *= beta
        backtracks += 1
        if alpha < 1e-8:
            break

    logger.debug(
        "Line search failed after %d backtracks; reverting positions and shrinking step size.",
        backtracks,
    )
    for vidx, pos in original_positions.items():
        mesh.vertices[vidx].position[:] = pos
    mesh.set_tilts_from_array(original_tilts)
    mesh.set_tilts_in_from_array(original_tilts_in)
    mesh.set_tilts_out_from_array(original_tilts_out)
    mesh.increment_version()

    logger.debug(
        "Zero-step detected: no trial step reduced energy (alpha reached %.2e).",
        alpha,
    )
    reduced_step = max(alpha * beta, 0.0)
    return False, max(reduced_step, step_size * beta), float(energy0)
