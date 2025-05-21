# runtime/constraint_manager.py

import importlib
import logging

logger = logging.getLogger("ConstraintManager")

class ConstraintModuleManager:
    def __init__(self, module_names):
        self.modules = {}
        for name in module_names:
            try:
                self.modules[name] = importlib.import_module(f"modules.constraints.{name}")
                logger.info(f"Loaded energy module: {name}")
            except ImportError as e:
                logger.error(f"Could not load constraint module '{name}': {e}")
                raise
        #self.modules = self._load_modules(module_names)

    def get_module(self, mod):
        """
        Retrieve a loaded constraint module by name.
        """
        if mod in self.modules.keys():
            return self.modules[mod]
        raise KeyError(f"Constraint module '{mod}' not found.")

    def _load_modules(self, names):
        loaded = {}
        for name in names:
            try:
                module = importlib.import_module(f"modules.constraints.{name}")
                if hasattr(module, "enforce_constraint"):
                    loaded[name] = module
                else:
                    logger.warning(f"Constraint module '{name}' lacks 'enforce_constraint' function.")
            except ImportError as e:
                logger.warning(f"Could not load constraint module '{name}': {e}")
        return loaded

    def enforce_all(self, mesh, **kwargs):
        for name, module in self.modules.items():
            logger.debug(f"Enforcing constraint: {name}")
            module.enforce_constraint(mesh, **kwargs)

    def __contains__(self, name):
        return name in self.modules

    def __getitem__(self, name):
        return self.modules[name]

