from typing import Dict, Callable, Optional, Tuple
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


def strong_wolfe_line_search(
    mesh: Mesh,
    direction: Dict[int, np.ndarray],
    gradient: Dict[int, np.ndarray],
    step_size: float,
    energy_fn: Callable[[], float],
    gradient_fn: Optional[Callable[[], Dict[int, np.ndarray]]] = None,
    c1: float = 1e-4,
    c2: float = 0.9,
    max_iter: int = 20,
    interpolation: bool = True,
    adaptive_c2: bool = True
) -> tuple[bool, float]:
    """
    Strong Wolfe line search with interpolation and adaptive parameters.
    
    This is a more sophisticated line search that satisfies both Armijo condition
    and curvature condition (Strong Wolfe conditions), providing better convergence
    properties than simple backtracking.
    
    Parameters
    ----------
    mesh : Mesh
        Mesh being optimized.
    direction : Dict[int, np.ndarray]
        Search direction for each vertex.
    gradient : Dict[int, np.ndarray]
        Current gradient at each vertex.
    step_size : float
        Initial step size.
    energy_fn : Callable[[], float]
        Function returning current energy.
    gradient_fn : Callable, optional
        Function returning current gradient. If None, finite differences used.
    c1 : float, optional
        Armijo condition parameter (0 < c1 < c2 < 1).
    c2 : float, optional
        Curvature condition parameter.
    max_iter : int, optional
        Maximum number of iterations.
    interpolation : bool, optional
        Whether to use interpolation for step size reduction.
    adaptive_c2 : bool, optional
        Whether to adapt c2 based on search direction properties.
        
    Returns
    -------
    tuple[bool, float]
        Success status and final step size.
    """
    original_positions = _save_positions(mesh)
    
    # Compute initial values
    phi_0 = energy_fn()
    dphi_0 = sum(np.dot(gradient[vidx], direction[vidx]) for vidx in direction)
    
    if dphi_0 >= 0:
        logger.debug("Non-descent direction provided; skipping step.")
        return False, step_size
    
    # Adaptive c2 based on problem characteristics
    if adaptive_c2:
        direction_norm = np.sqrt(sum(np.sum(d**2) for d in direction.values()))
        gradient_norm = np.sqrt(sum(np.sum(g**2) for g in gradient.values()))
        if gradient_norm > 0:
            curvature_estimate = abs(dphi_0) / (direction_norm * gradient_norm)
            c2 = min(0.9, max(0.1, 0.5 + 0.4 * curvature_estimate))
    
    # More-Thuente algorithm with bracketing and zoom phases
    alpha_max = min(10.0 * step_size, 1.0)  # Reasonable upper bound
    
    try:
        # Phase 1: Bracketing
        alpha_lo, alpha_hi, phi_lo, phi_hi, dphi_lo = _bracketing_phase(
            mesh, direction, energy_fn, gradient_fn, original_positions,
            phi_0, dphi_0, step_size, alpha_max, c1, max_iter // 2
        )
        
        if alpha_lo is None or alpha_hi is None or phi_lo is None or phi_hi is None or dphi_lo is None:
            logger.debug("Bracketing phase failed, falling back to backtracking")
            return _improved_backtracking(
                mesh, direction, gradient, step_size, energy_fn, 
                original_positions, phi_0, dphi_0, c1, interpolation, max_iter
            )
        
        # Phase 2: Zoom phase
        alpha_star = _zoom_phase(
            mesh, direction, energy_fn, gradient_fn, original_positions,
            alpha_lo, alpha_hi, phi_lo, phi_hi, dphi_lo, phi_0, dphi_0,
            c1, c2, max_iter // 2, interpolation
        )
        
        if alpha_star is not None and alpha_star > 1e-12:
            # Apply final step
            _apply_step(mesh, direction, original_positions, alpha_star)
            logger.debug(f"Strong Wolfe line search succeeded with α = {alpha_star:.3e}")
            return True, min(alpha_star * 1.2, 2.0 * step_size)  # Modest growth
        
    except Exception as e:
        logger.debug(f"Strong Wolfe search failed with error: {e}")
    
    # Fallback to improved backtracking
    _restore_positions(mesh, original_positions)
    return _improved_backtracking(
        mesh, direction, gradient, step_size, energy_fn, 
        original_positions, phi_0, dphi_0, c1, interpolation, max_iter
    )


def _save_positions(mesh: Mesh) -> Dict[int, np.ndarray]:
    """Save current mesh positions."""
    return {
        vidx: v.position.copy()
        for vidx, v in mesh.vertices.items()
        if not getattr(v, "fixed", False)
    }


def _restore_positions(mesh: Mesh, positions: Dict[int, np.ndarray]) -> None:
    """Restore mesh positions."""
    for vidx, vertex in mesh.vertices.items():
        if getattr(vertex, "fixed", False):
            continue
        vertex.position[:] = positions[vidx]


def _apply_step(mesh: Mesh, direction: Dict[int, np.ndarray], 
                original_positions: Dict[int, np.ndarray], alpha: float) -> None:
    """Apply step with given step size."""
    for vidx, vertex in mesh.vertices.items():
        if getattr(vertex, "fixed", False):
            continue
        disp = alpha * direction.get(vidx, np.zeros(3))
        vertex.position[:] = original_positions[vidx] + disp
        if hasattr(vertex, "constraint") and getattr(vertex, "constraint", None) is not None:
            constraint = getattr(vertex, "constraint")
            vertex.position[:] = constraint.project_position(vertex.position)


def _compute_directional_derivative(
    mesh: Mesh, direction: Dict[int, np.ndarray], 
    gradient_fn: Optional[Callable], gradient: Dict[int, np.ndarray]
) -> float:
    """Compute directional derivative."""
    if gradient_fn is not None:
        current_grad = gradient_fn()
        return sum(np.dot(current_grad[vidx], direction[vidx]) for vidx in direction)
    else:
        # Use provided gradient (may be stale but better than nothing)
        return sum(np.dot(gradient[vidx], direction[vidx]) for vidx in direction)


def _bracketing_phase(
    mesh: Mesh, direction: Dict[int, np.ndarray], energy_fn: Callable,
    gradient_fn: Optional[Callable], original_positions: Dict[int, np.ndarray],
    phi_0: float, dphi_0: float, alpha_init: float, alpha_max: float, 
    c1: float, max_iter: int
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Bracketing phase of More-Thuente algorithm.
    
    Returns
    -------
    alpha_lo, alpha_hi, phi_lo, phi_hi, dphi_lo : tuple of floats or None
        Bracketing interval endpoints and function values, or None if failed.
    """
    alpha_prev = 0.0
    alpha_curr = alpha_init
    phi_prev = phi_0
    dphi_prev = dphi_0
    
    for i in range(max_iter):
        # Evaluate at current point
        _apply_step(mesh, direction, original_positions, alpha_curr)
        phi_curr = energy_fn()
        
        # Check Armijo condition and detect bracket
        if (phi_curr > phi_0 + c1 * alpha_curr * dphi_0) or \
           (i > 0 and phi_curr >= phi_prev):
            # Found bracket
            return alpha_prev, alpha_curr, phi_prev, phi_curr, dphi_prev
            
        # Compute directional derivative
        dphi_curr = _compute_directional_derivative(mesh, direction, gradient_fn, {})
        
        # Check if we satisfy strong Wolfe (unlikely in bracketing but possible)
        if abs(dphi_curr) <= -c1 * dphi_0:  # Using c1 as a rough approximation
            return alpha_curr, alpha_curr, phi_curr, phi_curr, dphi_curr
            
        # Check if derivative became positive (found bracket)
        if dphi_curr >= 0:
            return alpha_curr, alpha_prev, phi_curr, phi_prev, dphi_curr
            
        # Expand search
        alpha_prev = alpha_curr
        phi_prev = phi_curr
        dphi_prev = dphi_curr
        alpha_curr = min(2 * alpha_curr, alpha_max)
        
        if alpha_curr >= alpha_max:
            break
    
    return None, None, None, None, None


def _zoom_phase(
    mesh: Mesh, direction: Dict[int, np.ndarray], energy_fn: Callable,
    gradient_fn: Optional[Callable], original_positions: Dict[int, np.ndarray],
    alpha_lo: float, alpha_hi: float, phi_lo: float, phi_hi: float, 
    dphi_lo: float, phi_0: float, dphi_0: float, c1: float, c2: float, 
    max_iter: int, interpolation: bool
) -> Optional[float]:
    """
    Zoom phase of More-Thuente algorithm.
    
    Returns
    -------
    alpha_star : float or None
        Optimal step size satisfying Strong Wolfe conditions, or None if failed.
    """
    for _ in range(max_iter):
        # Choose trial point using interpolation
        if interpolation and abs(alpha_hi - alpha_lo) > 1e-12:
            alpha_trial = _interpolate_step_size(
                alpha_lo, alpha_hi, phi_lo, phi_hi, dphi_lo, phi_0, dphi_0
            )
        else:
            alpha_trial = 0.5 * (alpha_lo + alpha_hi)
        
        # Safeguard against too small steps
        if abs(alpha_trial - alpha_lo) < 1e-12 or abs(alpha_trial - alpha_hi) < 1e-12:
            alpha_trial = 0.5 * (alpha_lo + alpha_hi)
        
        # Evaluate at trial point
        _apply_step(mesh, direction, original_positions, alpha_trial)
        phi_trial = energy_fn()
        
        # Check Armijo condition
        if (phi_trial > phi_0 + c1 * alpha_trial * dphi_0) or (phi_trial >= phi_lo):
            # Trial point too high, replace upper bound
            alpha_hi = alpha_trial
            phi_hi = phi_trial
        else:
            # Trial point satisfies Armijo, check curvature
            dphi_trial = _compute_directional_derivative(mesh, direction, gradient_fn, {})
            
            # Check Strong Wolfe conditions
            if abs(dphi_trial) <= c2 * abs(dphi_0):
                return alpha_trial  # Success!
            
            # Update bounds based on derivative sign
            if dphi_trial * (alpha_hi - alpha_lo) >= 0:
                alpha_hi = alpha_lo
                phi_hi = phi_lo
            
            alpha_lo = alpha_trial
            phi_lo = phi_trial
            dphi_lo = dphi_trial
        
        # Check convergence
        if abs(alpha_hi - alpha_lo) < 1e-12:
            break
    
    return alpha_lo if alpha_lo > 1e-12 else None


def _interpolate_step_size(
    alpha_lo: float, alpha_hi: float, phi_lo: float, phi_hi: float,
    dphi_lo: float, phi_0: float, dphi_0: float
) -> float:
    """
    Interpolate step size using cubic or quadratic interpolation.
    
    Returns
    -------
    alpha_new : float
        Interpolated step size.
    """
    # Try cubic interpolation first
    try:
        d1 = dphi_lo + dphi_0 - 3 * (phi_lo - phi_0) / alpha_lo
        d2_sq = d1**2 - dphi_lo * dphi_0
        
        if d2_sq >= 0:
            d2 = np.sqrt(d2_sq)
            alpha_new = alpha_lo - alpha_lo * (dphi_lo + d2 - d1) / (dphi_lo - dphi_0 + 2 * d2)
            
            # Check if interpolation is reasonable
            if alpha_lo < alpha_new < alpha_hi:
                return alpha_new
    except (ZeroDivisionError, ValueError, OverflowError):
        pass
    
    # Fallback to quadratic interpolation
    try:
        alpha_new = -dphi_0 * alpha_lo**2 / (2 * (phi_lo - phi_0 - dphi_0 * alpha_lo))
        if alpha_lo < alpha_new < alpha_hi:
            return alpha_new
    except (ZeroDivisionError, ValueError, OverflowError):
        pass
    
    # Final fallback: bisection
    return 0.5 * (alpha_lo + alpha_hi)


def _improved_backtracking(
    mesh: Mesh, direction: Dict[int, np.ndarray], gradient: Dict[int, np.ndarray],
    step_size: float, energy_fn: Callable, original_positions: Dict[int, np.ndarray],
    phi_0: float, dphi_0: float, c1: float, interpolation: bool, max_iter: int
) -> tuple[bool, float]:
    """
    Improved backtracking with interpolation and adaptive reduction factors.
    """
    alpha = step_size
    phi_prev = phi_0
    alpha_prev = 0.0
    
    for i in range(max_iter):
        _apply_step(mesh, direction, original_positions, alpha)
        phi_alpha = energy_fn()
        
        # Check Armijo condition
        if phi_alpha <= phi_0 + c1 * alpha * dphi_0:
            logger.debug(f"Improved backtracking succeeded with α = {alpha:.3e}")
            return True, min(alpha * 1.1, 2.0 * step_size)
        
        # Use interpolation for next step size
        if interpolation and i > 0:
            # Quadratic interpolation
            try:
                alpha_new = -dphi_0 * alpha**2 / (2 * (phi_alpha - phi_0 - dphi_0 * alpha))
                # Safeguard: ensure reasonable reduction
                alpha_new = max(0.1 * alpha, min(0.5 * alpha, alpha_new))
            except (ZeroDivisionError, ValueError, OverflowError):
                alpha_new = 0.5 * alpha
        else:
            # Use adaptive reduction factor
            reduction_factor = 0.5 if i == 0 else max(0.1, 0.5 * (1 - i / max_iter))
            alpha_new = alpha * reduction_factor
        
        alpha_prev, phi_prev = alpha, phi_alpha
        alpha = alpha_new
        
        if alpha < 1e-12:
            break
    
    # Restore original positions
    _restore_positions(mesh, original_positions)
    logger.debug("Improved backtracking failed; no suitable step found")
    return False, step_size


# Enhanced line search with automatic algorithm selection
def adaptive_line_search(
    mesh: Mesh,
    direction: Dict[int, np.ndarray],
    gradient: Dict[int, np.ndarray],
    step_size: float,
    energy_fn: Callable[[], float],
    gradient_fn: Optional[Callable[[], Dict[int, np.ndarray]]] = None,
    algorithm: str = "auto"
) -> tuple[bool, float]:
    """
    Adaptive line search that automatically selects the best algorithm.
    
    Parameters
    ----------
    algorithm : str
        Line search algorithm: "auto", "strong_wolfe", "backtrack", or "simple"
    """
    if algorithm == "auto":
        # Choose algorithm based on problem characteristics
        direction_norm = np.sqrt(sum(np.sum(d**2) for d in direction.values()))
        gradient_norm = np.sqrt(sum(np.sum(g**2) for g in gradient.values()))
        
        if gradient_fn is not None and gradient_norm > 1e-6:
            algorithm = "strong_wolfe"
        else:
            algorithm = "backtrack"
    
    if algorithm == "strong_wolfe":
        return strong_wolfe_line_search(
            mesh, direction, gradient, step_size, energy_fn, gradient_fn
        )
    elif algorithm == "backtrack":
        return _improved_backtracking(
            mesh, direction, gradient, step_size, energy_fn,
            _save_positions(mesh), energy_fn(), 
            sum(np.dot(gradient[vidx], direction[vidx]) for vidx in direction),
            1e-4, True, 15
        )
    else:  # "simple" or fallback
        return backtracking_line_search(
            mesh, direction, gradient, step_size, energy_fn
        )
