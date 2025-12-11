# Membrane Solver

Membrane Solver is a simulation platform inspired by Surface Evolver, designed to model and minimize the energy of geometric structures such as membranes and surfaces. It supports volume and surface area constraints, dynamic refinement of meshes, and customizable energy modules for physical effects like surface tension and curvature. The project aims to provide a flexible and extensible framework for exploring complex geometries and their energy-driven evolution.

## Interactive mode

`main.py` starts in interactive mode by default, presenting a simple command
prompt after any initial instructions execute. Use `--non-interactive` to skip
the prompt. Commands such as `g5` perform five minimization steps while `r`
refines the mesh. Type `quit` or `exit` to stop the loop and save the final
mesh.

If no input file is specified on the command line you will be prompted for the
path. File names may be given with or without the `.json` suffix.

## Geometry loading

`parse_geometry` automatically triangulates any facet with more than three edges
using `refine_polygonal_facets`. Triangular facets remain unchanged. The
returned mesh is therefore ready for optimization without further refinement.


## TODO:
1. ~~equiangulation~~ ✅ COMPLETED - Available via 'u' command
2. use https://fdmatoz.github.io/PyMembrane/pythonapi/bending.html for
   reference
3. add fixed option 
4. add tests:
    4.1 test fixed
5. implement no_refine
6. optimize optimization computations 
7. how to calculate mean curvature from facets?
8. surface area constraints:
    8.1. local - specific faces
    8.2. global - total area conservation -> maybe replace volume constraint?
9. different library for visualization, that allows minimization while updating the figure on display
10. add option for "comment" attribute in the json file, maybe change to
    another format that supports comments.
11. when running add -v/--verbose options to print out log messages or just
    put them in the log file
12. add "quiet" mode for minimization
13. initial energy calculation after loading -> test modules
14. interactive mode while showing updating visualization
        idea: https://matplotlib.org/stable/gallery/animation/double_pendulum.html#sphx-glr-gallery-animation-double-pendulum-py
15. visualization script - exporting to obj, vtk, etc.
16. remove "options" and have specific categories and attributes.
        i.e., a constraints category, energy category, etc.
17. look at the Evolver manual pp.235 at the 16.8 Iteration chapter
18. automatic time-step as 10% of the system size or maybe other small value
19. add "gyration radius" - in dev tools

## Design thoughts 
1. Should I keep track of indices after refining facets and/or edges?
    1.1 Yes: will allow better backtracking and debugging
    1.2 No: will allow better tracking of current state and number of object

## Roadmap

The detailed development roadmap has been moved to `docs/ROADMAP.md`. In brief,
near‑term goals include:

- Stabilizing and benchmarking baseline shape problems (cube→sphere,
  square→circle, capillary bridge).
- Adding curvature energies (mean and Gaussian) and validating against classic
  examples such as catenoids and pinned caps.
- Implementing tilt fields and caveolin‑like inclusions as a 3D extension of
  the model in `docs/caveolin_generate_curvature.pdf`.

## Performance benchmarks

- `python benchmarks/benchmark_cube_good.py` runs the full `cube_good_min_routine`
  recipe (minimization, refinement, equiangulation, vertex averaging, etc.) and
  reports the average wall-clock time over three runs. This is the main
  regression benchmark we use when tuning steppers, constraints, or geometry
  kernels. It runs entirely in-place and requires no writable temporary
  directories.
- `python benchmarks/benchmark_surface_energy.py` compares the old vs. new
  surface energy implementations by generating temporary cube variants with
  different facet `energy` modules. It prints average runtimes for both
  implementations so you can evaluate surface-kernel optimizations in
  isolation.
