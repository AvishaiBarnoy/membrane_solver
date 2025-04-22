# My try on a SurfaceEvolver-like software

## TODO:
1. make sure that refining creates new objects/instances and not just lists
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
11. add tests
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
20. create class Mesh/PolygonalDomain that wraps general functions and keeps
    track of all lists (vertices, edges, facets, bodies).

## Design thoughts 
1. Should I keep track of indices after refining facets and/or edges?
    1.1 Yes: will allow better backtracking and debugging
    1.2 No: will allow better tracking of current state and number of object

## roadmap
1. box that minimizes into a sphere - volume conservation
2. square that minimizes into circle - area conservation
3. box that minimizes into a sphere with a dent - fixed/no_refine
4. a plane with a disk and outer perimeter
5. after implementing mean curvature energy -> membrane between two fixed parallel circels
    results in a catenoid
    6.1 check how cat.fe treats the fixed surface area of the soapfilm
