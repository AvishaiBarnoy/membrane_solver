# Membrane Solver

Membrane Solver is a simulation platform inspired by Surface Evolver, designed to model and minimize the energy of geometric structures such as membranes and surfaces. It supports volume and surface area constraints, dynamic refinement of meshes, and customizable energy modules for physical effects like surface tension, line tension, and curvature. The project aims to provide a flexible and extensible framework for exploring complex geometries and their energy-driven evolution.

## Interactive mode

`main.py` starts in interactive mode by default, presenting a simple command
prompt after any initial instructions execute. Use `--non-interactive` to skip
the prompt. Commands such as `g5` perform five minimization steps while `r`
refines the mesh (`rN` repeats refinement N times). Use `p` (or the new alias
`i`) to print physical properties at any time. Type `quit` or `exit` to stop
the loop and save the final
mesh.

If no input file is specified on the command line you will be prompted for the
path. File names may be given with or without the `.json` suffix.

## Geometry loading

`parse_geometry` automatically triangulates any facet with more than three edges
using `refine_polygonal_facets`. Triangular facets remain unchanged. The
returned mesh is therefore ready for optimization without further refinement.

For edge‑only or “wire‑frame” geometries, the `faces` section in the JSON is
optional: meshes with just `vertices` and `edges` can be loaded and visualized
without defining facets.

## Visualization

Use the visualization script to inspect geometries:

```bash
python visualize_geometry.py meshes/cube.json
```

The script now delegates to a small CLI under `visualization/cli.py` and
accepts several flags:

- `python visualize_geometry.py meshes/cube.json --transparent`
  Draw facets semi‑transparent.

- `python visualize_geometry.py meshes/cube.json --no-edges`
  Hide edges and show only filled facets (useful for solid views).

- `python visualize_geometry.py meshes/simple_line.json --no-facets --scatter`
  Visualize line‑only meshes: edges only, plus vertex scatter points.

- `python visualize_geometry.py meshes/cube.json --no-axes --save outputs/cube.png`
  Remove axes and save the figure to an image file instead of only showing it.

All interactive visualizations are based on the shared `visualization.plotting.plot_geometry`
helper, which is also used in the test suite.

## Constraint modules

Membrane Solver mirrors Evolver’s mix of penalties and hard constraints. Surface
area (body/facet/global) and volume constraints are configured via
`constraint_modules` plus the relevant `target_*` values on entities or in
`global_parameters`. Perimeter conservation uses a loop of signed edge indices:

```json
"constraint_modules": ["perimeter"],
"global_parameters": {
  "perimeter_constraints": [
    {
      "edges": [1, 2, 3, 4],
      "target_perimeter": 4.0
    }
  ]
}
```

The new regression tests in `tests/test_perimeter_minimization.py` load the same
square loop (see `tests/sample_meshes.square_perimeter_input`) to verify that
energy drops while the loop returns to the requested perimeter, even after mesh
refinement and equiangulation.


## Development notes

- The evolving backlog/TODO list now lives in `docs/ROADMAP.md` so changes are
  tracked alongside the code. Refer to that document for design sketches,
  medium-term research targets, and open questions.

## Linting

This repo uses Ruff for linting (and can adopt Ruff formatting later).

```bash
pip install ruff
ruff check .
```

To enforce linting locally before commits, install and enable `pre-commit`:

```bash
pip install pre-commit
pre-commit install
pre-commit run -a
```

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

- `python benchmarks/suite.py` is the main entry point for performance testing. It runs a set of standard scenarios (including `cube_good` and `square_to_circle`), tracks execution time history in `benchmarks/results.json`, and highlights regressions or improvements.
- `python benchmarks/benchmark_cube_good.py` runs the full `cube_good_min_routine` recipe (minimization, refinement, equiangulation, vertex averaging, etc.) and reports the average wall-clock time.
- `python benchmarks/benchmark_square_to_circle.py` runs the `square_to_circle` scenario (square sheet relaxing to a circle with line tension), serving as a stress test for mesh maintenance operations.
