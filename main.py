import argparse
from energy import total_energy
from geometry_io import load_geometry
from logging_config import setup_logging
import logging
logger = logging.getLogger('membrane_solver')

import sys

if __name__ == "__main__":
    logger = setup_logging()

    logger.critical("""Critical error! Cannot proceed further. main.py script not
                    implemented yet""")
    sys.exit(1)

    logger.info('Starting geometry calculation.')
    # TODO: add argparse to get input file
    filename = "meshes/sample_geometry.json"
    vertices, facets, global_params, body, modules = load_geometry(filename)
    total = total_energy(vertices, facets, global_params, body, modules)

