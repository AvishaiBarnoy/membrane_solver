import argparse
from energy import total_energy
from geometry.geometry_io import load_data, parse_geometry, parse_inputfile
from runtime.energy_manager import EnergyModuleManager
from runtime.refinement import refine_polygonal_facets, refine_triangle_mesh
from logging_config import setup_logging
import logging

logger = logging.getLogger('membrane_solver')

import sys

if __name__ == "__main__":
    logger = setup_logging()


    # TODO: add argparse to get input file
    filename = "meshes/sample_geometry.json"
    # Loading geometry
    data = load_data(filename)
    vertices, edges, facets, bodies, global_params = parse_geometry(data=data)
    vertices, edges, tri_facets, bodies = refine_polygonal_facets(vertices,
                                                                  edges,
                                                                  facets,
                                                                  bodies,
                                                                  global_params)
    logger.info("\nAfter initial triangulation:")
    logger.info(f"Number of vertices: {len(vertices)}")
    for v in vertices: logger.debug(v.position)
    logger.info(f"Number of facets: {len(tri_facets)}")
    logger.debug("Triangulated facets:")
    for facet in tri_facets: logger.debug(f"{facet} {facet.options}")

    vertices, edges, tri_facets, bodies = refine_triangle_mesh(vertices, edges, tri_facets, bodies)

    min_instructions = data.get("instructions", [])
    print(min_instructions)
    sys.exit(1)

    # 1. Find all energy module names used 
    used_modules = set()
    for facet in facets:
        used_modules.add(facet.options.get("energy", "surface"))

    sys.exit(1)
    for body in bodies:
        used_modules.add(body.options.get("energy", "volume"))

    # 2. Load them
    manager = EnergyModuleManager(used_modules)

    # 3. Calculate initial energy
    initial_total_energy = total_energy(vertices, facets, bodies, global_params, modules)
    print(f"Initial total energy: {initial_total_energy}")
    print("print here initial values")

