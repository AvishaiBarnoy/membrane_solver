# Membrane Solver

Membrane Solver is a simulation platform inspired by Surface Evolver, designed to model and minimize the energy of geometric structures such as membranes and surfaces. It supports volume and surface area constraints, dynamic refinement of meshes, and customizable energy modules for physical effects like surface tension and curvature. The project aims to provide a flexible and extensible framework for exploring complex geometries and their energy-driven evolution.

## Interactive mode

Run `main.py` with the `-I/--interactive` flag to enter a simple command prompt
after any initial instructions execute. Commands such as `g5` perform five
minimization steps while `r` refines the mesh. Type `quit` or `exit` to stop the
loop and save the final mesh.

## Geometry loading

`parse_geometry` automatically triangulates any facet with more than three edges
using `refine_polygonal_facets`. Triangular facets remain unchanged. The
returned mesh is therefore ready for optimization without further refinement.


## TODO:
1. equiangulation
2. vertex averaging
3. add fixed option 
4. add tests:
    3.1 test fixed
5. implement no_refine
6. optimize optimization computations 
7. how to calculate mean curvature from facets?
8. surface area constraints:
    8.1. local - specific faces
    8.2. global - total area conservation -> maybe replace volume constraint?
9. different library for visualization, for example:
    12.1 https://docs.pyvista.org/
    12.2 https://vedo.embl.es/
10. add option for "comment" attribute in the json file, maybe change to
    another format that supports comments.
11. when running add -v/--verbose options to print out log messages or just
    put them in the log file
12. add "quiet" mode for minimization
13. initial energy calculation after loading -> test modules
14. add interactive mode for refining, minimization, etc.
15. visualization script - exporting to obj, vtk, etc.
16. remove "options" and have specific categories and attributes.
        i.e., a constraints category, energy category, etc.
17. look at the Evolver manual pp.235 at the 16.8 Iteration chapter
18. automatic time-step as 5% of the system size or maybe other small value
19. add "gyration radius"

## Design thoughts 
1. Should I keep track of indices after refining facets and/or edges?
    1.1 Yes: will allow better backtracking and debugging
    1.2 No: will allow better tracking of current state and number of object

## roadmap
1. cube that minimizes into a sphere
    1.1 soft constraint - volume energy - DONE
    1.2 hard constraint - lagrange multiplier
2. square that minimizes into circle
    2.1 soft constraint - surface area energy
    2.2 hard constraint - lagrange multiplier
3. capillary bridge (catenoid) - two circles at fixed distance, tests surface - tension and volume constraint
4. Pinned hemisphere under tension to yield spherical caps - boundary constraints 
        and mean curvature-pressure balance
5. Pure Gaussian curvature - check no change under constant topology, and no edge
6. Tilt source decay - Should form a “dimple” or invagination
7. Dimpled sphere with one embedded caveolin disk - 
8. box that minimizes into a sphere with a dent - fixed/no_refine
9. a plane with a disk and outer perimeter
10. after implementing mean curvature energy -> membrane between two fixed parallel circels
    results in a catenoid
    10.1 check how cat.fe treats the fixed surface area of the soapfilm
11. a flat sheet that folds to its spontaneous curvature
12. final(?) - a single caveolin with outside membrane decays
13. automatic minimization, user defines target refinement and the program
    tries to minimize
