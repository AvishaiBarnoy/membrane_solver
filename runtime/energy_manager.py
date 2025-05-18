import importlib
import logging

logger = logging.getLogger("membrane_solver")

class EnergyModuleManager:
    # TODO: add documentation
    def __init__(self, module_names):
        self.modules = {}
        for name in module_names:
            try:
                self.modules[name] = importlib.import_module(f"modules.energy.{name}")
                logger.info(f"Loaded energy module: {name}")
            except ImportError as e:
                logger.error(f"Could not load module '{name}': {e}")
                raise

    def get_module(self, mod):
        """
        Retrieve a loaded energy module by name.
        """
        if mod in self.modules.keys():
            return self.modules[mod]
        raise KeyError(f"Energy module '{mod}' not found.")

    def get_energy_function(self, name, object_type):
        # TODO: is this needed? mostly moved to compute_energy_and_gradient() formulation
        """
        Given a module name and object type ('facet', 'body'), return an appropriate energy function.
        Prioritized lookup:
            1. 'calculate_energy' (default handler for all objects)
            2. type-specific, e.g., 'calculate_surface_energy' or 'calculate_volume_energy'
        """
        mod = self.modules[name]

        if hasattr(mod, "calculate_energy"):
            return getattr(mod, "calculate_energy")
        elif hasattr(mod, "compute_energy_and_gradient"):
            return getattr(mod, "compute_energy_and_gradient")

        fn_map = {
            "surface": "calculate_surface_energy",
            "volume": "calculate_volume_energy"
        }

        fn_name = fn_map.get(object_type)
        if hasattr(mod, fn_name):
            return getattr(mod, fn_name)

        raise AttributeError(
            f"Module '{name}' must define either 'calculate_energy' or '{fn_name}' for type '{object_type}'"
        )
