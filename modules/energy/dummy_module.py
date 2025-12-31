import logging

logger = logging.getLogger("membrane_solver")


def compute_energy_and_gradient(
    _mesh=None,
    _global_params=None,
    _param_resolver=None,
    *,
    compute_gradient: bool = True,
):
    logger.info("THIS IS A DUMMY MODULE FOR TESTING")
    if not compute_gradient:
        return 0.0, {}
    return 0.0, {}


def compute_energy_and_gradient_array(
    _mesh,
    _global_params,
    _param_resolver,
    *,
    positions,
    index_map,
    grad_arr,
):
    """Dense-array dummy energy (no-op)."""
    return 0.0


__all__ = ["compute_energy_and_gradient", "compute_energy_and_gradient_array"]
