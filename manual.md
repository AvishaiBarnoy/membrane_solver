Here I will take notes for the manual...

1. energy modules are loaded once during input parsing
2. modules have a strict naming and should end with a funcion "compute_energy_and_gradient()"
    which takes these inputs: mesh, global_params, param_resolver 
    mesh: geometric structure and data on the system
    global_params: global parameters can be given as mesh.global_parameters,
                    will maybe change later
    param_resolver: a resolver function in case some instances have parameters
                    different from those in global_parameters, e.g., some
                    facets have different surface_tension 
3. if no energy is given to an element then default energy will be added:
    facet - surface_tension
    body - volume_stiffness which is only relevant if there is target_volume

4. Inheritance rules:
    4.1 child facets inherit all energy and constraints of parent facet
    4.2 split edges inherit constraints of parent edge
    4.3 middle edges generated in the parent facet (between new vertices) only
        inherit constraints defined at the facet level, since they are brand
        new (also applies to facets generated during polygonal refinement)
    4.4 new midpoint vertices will inherit constraints (including fixed) from
        their parent edge or facet. So fixed on edges and facets define the
        fixed attribute of their child vertices. 

5. options structure for input file, python dictionary:
    {"refine": true/false}
    {"constraints": ["constraint1", "constraint2"]} or {"constraints": "constraint1"}
    {"energy": ["energy1", "energy2"]} or {"energy": "energy"}
    Parameters can also be adjusted, e.g., {"surface_tension": 5} will override
        the default surface tension value of 1.0.

