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

4. Inheritance rules:
    4.1 child facets inherit all energy and constraints of parent facet
    4.2 split edges inherit constraints of parent edge
    4.3 middle edges generated in the parent facet (between new vertices) only
        inherit constraints defined at the facet level, since they are brand
        new (also applies to facets generated during polygonal refinement)
    4.4 new midpoint vertices will inherit constraints (including fixed) if and only if
        both parent vertices have the constraint and the edge has the fixed
        flag. During polygonal refinement the middle vertex will only inherit
        constraints (including fixed) defined at the facet level.
