import logging
from typing import Callable, Dict

import numpy as np

from geometry.entities import Mesh
from runtime.topology import check_max_normal_change, get_min_edge_length

logger = logging.getLogger('membrane_solver')


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
) -> tuple[bool, float]:
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
    original_positions = {
        vidx: v.position.copy()
        for vidx, v in mesh.vertices.items()
        if not getattr(v, "fixed", False)
    }

    energy0 = energy_fn()

    # Pre-compute stability threshold
    min_edge_len = get_min_edge_length(mesh)
    safe_step_limit = 0.3 * min_edge_len if min_edge_len > 0 else float('inf')

    # Calculate max possible displacement vector magnitude (unscaled)
    max_dir_norm = 0.0
    if direction:
        # Optimization: convert direction dict values to array once if needed,
        # but max loop is fast enough for now
        max_dir_norm = max(np.linalg.norm(d) for d in direction.values())

    def constraint_violation(m: Mesh) -> float:
        """Return a max relative violation over area/volume constraints."""
        max_violation = 0.0
        if m.bodies:
            for body in m.bodies.values():
                # Area constraint
                if body.options.get("target_area") is not None:
                    target = body.options["target_area"]
                    area = body.compute_surface_area(m)
                    denom = max(abs(target), 1.0)
                    max_violation = max(max_violation, abs(area - target) / denom)
                # Volume constraint
                tgt_vol = body.target_volume
                if tgt_vol is None:
                    tgt_vol = body.options.get("target_volume")
                if tgt_vol is not None:
                    vol = body.compute_volume(m)
                    denom = max(abs(tgt_vol), 1.0)
                    max_violation = max(max_violation, abs(vol - tgt_vol) / denom)
        return max_violation

    base_violation = constraint_violation(mesh)
    g_dot_d = sum(np.dot(gradient[vidx], direction[vidx]) for vidx in direction)

    if g_dot_d >= 0:
        logger.debug("Non-descent direction provided; skipping step.")
        return False, step_size

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
            if hasattr(vertex, "constraint"):
                vertex.position[:] = vertex.constraint.project_position(
                    vertex.position
                )

        # Stability Check: Only run expensive check if step is large
        if not is_safe_small_step:
            if not check_max_normal_change(mesh, original_positions):
                # Treat as failure -> backtrack
                for vidx, vertex in mesh.vertices.items():
                    if getattr(vertex, "fixed", False):
                        continue
                    vertex.position[:] = original_positions[vidx]

                alpha *= beta
                backtracks += 1
                if alpha < 1e-8:
                    break
                continue

        trial_energy = energy_fn()
        if trial_energy <= energy0 + c * alpha * g_dot_d:
            armijo_pass = True
        else:
            armijo_pass = False

        if constraint_enforcer is not None:
            constraint_enforcer(mesh)
            trial_energy_after_constraint = energy_fn()
            vio_after = constraint_violation(mesh)
        else:
            trial_energy_after_constraint = trial_energy
            vio_after = base_violation

        constraint_improved = vio_after < base_violation * (1.0 - 1e-6)

        # Accept if Armijo passes, or if constraints improved while energy did not blow up.
        energy_guard = trial_energy_after_constraint <= energy0 * 1.05 + 1e-8

        if armijo_pass or (constraint_improved and energy_guard):
            logger.debug(
                "Line search success: alpha=%.3e, backtracks=%d, "
                "E0=%.6f, Etrial=%.6f (post-constraint=%.6f), "
                "violation %.3e -> %.3e (improved=%s, armijo=%s)",
                alpha,
                backtracks,
                energy0,
                trial_energy,
                trial_energy_after_constraint,
                base_violation,
                vio_after,
                constraint_improved,
                armijo_pass,
            )
            new_step = min(alpha * gamma, alpha_max)
            return True, new_step

        # Reject this scale: restore and try a smaller one.
        for vidx, vertex in mesh.vertices.items():
            if getattr(vertex, "fixed", False):
                continue
            vertex.position[:] = original_positions[vidx]

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
        if getattr(vertex, "fixed", False):
            continue
        vertex.position[:] = original_positions[vidx]

    logger.debug(
        "Zero-step detected: no trial step reduced energy (alpha reached %.2e).",
        alpha,
    )
    reduced_step = max(alpha * beta, 0.0)
    return False, max(reduced_step, step_size * beta)
