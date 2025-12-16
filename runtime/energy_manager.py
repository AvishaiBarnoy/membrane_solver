# runtime/energy_manager.py

import importlib
import logging
from collections import Counter

logger = logging.getLogger("membrane_solver")

class EnergyModuleManager:
    def __init__(self, module_names):
        self.modules = {}
        counted = Counter(module_names)
        for name, count in counted.items():
            if count > 1:
                logger.warning(f"Energy module '{name}' specified {count} times; using only one instance.")

            try:
                self.modules[name] = importlib.import_module(f"modules.energy.{name}")
                logger.info(f"Loaded energy module: {name}")
            except ImportError as e:
                logger.error(f"Could not load energy module '{name}': {e}")
                raise

    def get_module(self, mod):
        """
        Retrieve a loaded energy module by name.
        """
        if mod in self.modules.keys():
            return self.modules[mod]
        raise KeyError(f"Energy module '{mod}' not found.")
