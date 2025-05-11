# My try on a SurfaceEvolver-like software

## TODO:
1. add fixed option to vertices 
3. how to calculate mean curvature from facets?
4. energy minimization algorithms
6. surface area constraints:
    1. local - specific faces
    2. global - total area conservation -> maybe replace volume constraint?
7. rigid facets -> all edges move together?
8. fixed option -> vertix: don't move, edges: don't change length, facets: does anything?
    how is energy evaluated over fixed components? in SE integrals are not
    evaluated over fixed edges, what about vertices (is energy even evaluted
    over individual vetices???) and facets?
12. different library for visualization, for example:
    12.1 https://docs.pyvista.org/
    12.2 https://vedo.embl.es/
14. add option for "comment" attribute in the json file, maybe change to
    another format that supports comments.
15. when running add -v/--verbose options to print out log messages or just
    put them in the log file
16. initial energy calculation after loading -> test modules
18. add interactive mode for refining, minimization, etc.
19. visualization script - exporting to obj, vtk, etc.

## Design thoughts 
1. Should I keep track of indices after refining facets and/or edges?
    1.1 Yes: will allow better backtracking and debugging
    1.2 No: will allow better tracking of current state and number of object

## roadmap
1. cube that minimizes into a sphere - volume conservation
2. square that minimizes into circle - area conservation
3. capillary bridge (catenoid) - two circles at fixed distance, tests surface
        tension and volume constraint
4. Pinned hemisphere under tension to yield spherical caps - boundary constraints 
        and mean curvature-pressure balance
5. Pure Gaussian curvature - check no change under constant topology
6. Tilt source decay - Should form a “dimple” or invagination
7. Dimpled sphere with one embedded caveolin disk - 
8. box that minimizes into a sphere with a dent - fixed/no_refine
9. a plane with a disk and outer perimeter
10. after implementing mean curvature energy -> membrane between two fixed parallel circels
    results in a catenoid
    10.1 check how cat.fe treats the fixed surface area of the soapfilm
11. a flat sheet that folds to its spontaneous curvature
