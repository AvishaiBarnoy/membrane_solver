from typing import Dict, Callable
import numpy as np
import logging
from geometry.entities import Mesh

logger = logging.getLogger('membrane_solver')


def backtracking_line_search(
    mesh: Mesh,
    direction: Dict[int, np.ndarray],
    gradient: Dict[int, np.ndarray],
    step_size: float,
    energy_fn: Callable[[], float],
    max_iter: int = 10,
    beta: float = 0.2,
    c: float = 1e-4,
    gamma: float = 1.2,
    alpha_max_factor: float = 10.0,
    constraint_enforcer: Callable[[Mesh], None] | None = None,
) -> tuple[bool, float]:
    """Perform Armijo backtracking line search along ``direction``.

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
    g_dot_d = sum(np.dot(gradient[vidx], direction[vidx]) for vidx in direction)

    if g_dot_d >= 0:
        logger.debug("Non-descent direction provided; skipping step.")
        return False, step_size

    alpha = step_size
    alpha_max = alpha_max_factor * step_size

    for _ in range(max_iter):
        for vidx, vertex in mesh.vertices.items():
            if getattr(vertex, "fixed", False):
                continue
            disp = alpha * direction.get(vidx, np.zeros(3))
            vertex.position[:] = original_positions[vidx] + disp
            if hasattr(vertex, "constraint"):
                vertex.position[:] = vertex.constraint.project_position(
                    vertex.position
                )

        if constraint_enforcer is not None:
            constraint_enforcer(mesh)

        trial_energy = energy_fn()
        if trial_energy <= energy0 + c * alpha * g_dot_d:
            logger.debug(f"Line search accepted step size {alpha:.3e}")
            new_step = min(alpha * gamma, alpha_max)
            return True, new_step

        alpha *= beta

    logger.debug(
        f"Line search failed to satisfy Armijo after {max_iter} iterations. Reverting."
    )
    for vidx, vertex in mesh.vertices.items():
        if getattr(vertex, "fixed", False):
            continue
        vertex.position[:] = original_positions[vidx]

    logger.info("Zero-step detected: no trial step reduced energy.")
    logger.info(f"Current step_size = {step_size:.2e}")
    return False, step_size
