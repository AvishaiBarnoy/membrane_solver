Here will be notes for the manual...

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
