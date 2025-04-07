import argparse
from energy import total_energy
from geometry_io import load_geometry
from energy_manager import EnergyModuleManager
from logging_config import setup_logging
import logging
logger = logging.getLogger('membrane_solver')

import sys

if __name__ == "__main__":
    logger = setup_logging()


    # TODO: add argparse to get input file
    filename = "meshes/sample_geometry.json"
    # Loading geometry
    vertices, facets, body, global_params = parse_inputfile(filename)

    # Loading energy modules
    used_modules = set()
    for facet in facets:
        used_modules.add(facet.options.get("energy", "surface"))

    for body in bodies:
        used_modules.add(body.options.get("energy", "volume"))
    manager = EnergyModuleManager(used_modules)

    initial_total_energy = total_energy(vertices, facets, global_params, body, modules)
    print(f"Initial total energy: {initial_total_energy}")
    print("print here initial values")

