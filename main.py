from logging_config import setup_logging
import argparse
from energy import total_energy

if __name__ == "__main__":
    logger = setup_logging()

    logger.info('Starting geometry calculation.')
    # TODO: add argparse to get input file
    filename = "meshes/sample_geometry.json"
    vertices, facets, global_params, body, modules = load_geometry(filename)
    total = total_energy(vertices, facets, global_params, body, modules)

