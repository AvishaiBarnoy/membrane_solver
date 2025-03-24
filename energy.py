# energy.py
import importlib
import logging

logger = logging.getLogger("MembraneSolver")

def load_energy_modules(modules_list):
    """
    Dynamically loads energy modules from the 'modules' package.

    Parameters:
        modules_list (list of str): Names of the energy modules to load.

    Returns:
        dict: {module_name: module_object}
    """
    loaded_modules = {}
    for module_name in modules_list:
        try:
            module = importlib.import_module(f"modules.{module_name}")
            loaded_modules[module_name] = module
            logger.info(f"Successfully loaded module: {module_name}")
        except ImportError as e:
            logger.error(f"Failed to import module '{module_name}': {e}")
    return loaded_modules

def total_energy(facets, body, global_params, module_dict):
    """
    Calculates total energy based on loaded modules and geometry.

    Parameters:
    - facets: list of Facet objects
    - body: dictionary with body properties (e.g. target_volume, list of facet indices)
    - global_params: dict of global physical parameters (e.g. surface tension)
    - modules_dict: dict of loaded modules from load_energy_modules()

    Returns:
    - float: total energy
    """
    energy_sum = 0.0

    for name, module in modules_dict.items():
        try:
            energy = module.energy(facets, body, global_params)
            logger.debug(f"Module '{name}' contributed energy: {energy}")
            energy_sum += energy
        except Exception as e:
            logger.warning(f"Module '{name}' failed to compute energy: {e}")

    logger.info(f"Total energy from all modules: {energy_sum}")
    return energy_sum

if __name__ == "__main__":
    logger.info("This script is intended to be imported, not run directly.")
