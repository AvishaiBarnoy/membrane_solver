# energy.py
import importlib
import logging

logger = logging.getLogger("MembraneSolver")

def load_energy_modules(modules_list):
    loaded_modules = {}
    for module_name in modules_list:
        try:
            module = importlib.import_module(f"modules.{module_name}")
            loaded_modules[module_name] = module
