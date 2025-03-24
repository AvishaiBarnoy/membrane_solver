# My try on a SurfaceEvolver-like software

## TODO:
1. add fixed option to vertices 
2. define volume constraints -> leads to pressure
    if volume!=target_volume then pressure arises
    adding "negative surface tension" proportional to the deviation from the
    target volume
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
13. move scripts to a different folder and leave only main and deploy outside.
14. add option for "comment" attribute in the json file, maybe change to
    another format that supports comments.
15. when running add -v/--verbose options to print out log messages or just
    put them in the log file
16. initial energy calculation after loading -> test modules
18. add interactive mode for refining, minimization, etc.

## Current limitations
1. only one object with volume
2.

## roadmap
1. box that minimizes into a sphere
2. box that minimizes into a sphere with a dent
3. a square that minimizes into a disk -> what polygon would be considered
   a disk?
4. a plane with a disk and outer perimeter
5. after implementing mean curvature energy -> membrane between two fixed parallel circels
   results in a catenoid 
