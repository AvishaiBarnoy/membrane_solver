import importlib
import logging

logger = logging.getLogger("membrane_solver")

class EnergyModuleManager:
    # TODO: add documentation
    def __init__(self, module_names):
        self.modules = {}
        for name in module_names:
            try:
                self.modules[name] = importlib.import_module(f"modules.{name}")
                logger.info(f"Loaded energy module: {name}")
            except ImportError as e:
                logger.error(f"Could not load module '{name}': {e}")
                raise

    def get_energy_function(self, name, object_type):
        """
        Given a module name and object type ('facet', 'body'), return an appropriate energy function.
        Prioritized lookup:
            1. 'calculate_energy' (default handler for all objects)
            2. type-specific, e.g., 'calculate_surface_energy' or 'calculate_volume_energy'
        """
        mod = self.modules[name]

        if hasattr(mod, "calculate_energy"):
            return getattr(mod, "calculate_energy")

        fn_map = {
            "facet": "calculate_surface_energy",
            "body": "calculate_volume_energy"
        }

        fn_name = fn_map.get(object_type)
        if hasattr(mod, fn_name):
            return getattr(mod, fn_name)

        raise AttributeError(
            f"Module '{name}' must define either 'calculate_energy' or '{fn_name}' for type '{object_type}'"
        )
