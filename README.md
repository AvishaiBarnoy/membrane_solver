# Membrane Solver

Membrane Solver is a simulation platform inspired by Surface Evolver, designed to model and minimize the energy of geometric structures such as membranes and surfaces. It supports volume and surface area constraints, dynamic refinement of meshes, and customizable energy modules for physical effects like surface tension, line tension, and curvature. The project aims to provide a flexible and extensible framework for exploring complex geometries and their energy-driven evolution.

## Interactive mode

`main.py` starts in interactive mode by default, presenting a simple command
prompt after any initial instructions execute. Use `--non-interactive` to skip
the prompt.

Commands:
- `g5`: Perform five minimization steps.
- `r` / `rN`: Refine the mesh (N times).
- `u`: Equiangulate the mesh.
- `V`: Vertex average.
- `p` / `i`: Print physical properties (area, volume, surface radius of gyration, target volume).
- `print [entity] [filter]`: Query geometry (e.g., `print vertex 0`, `print edges len > 0.5`).
- `print energy breakdown`: Show per-module energy contributions.
- `print macros`: List available macros.
- `set [param/entity] [value]`: Set properties (e.g., `set surface_tension 1.5`, `set vertex 0 fixed true`).
- `lv` / `live_vis`: Toggle live 3D visualization during minimization.
- `quit` / `exit`: Stop the loop and save the final mesh.

The default minimization stepper is Gradient Descent; switch to Conjugate Gradient
with the `cg` command if needed.

If no input file is specified on the command line you will be prompted for the
path. File names may be given with or without the `.json` suffix.

## Macros

Input files may define named macros (similar to Evolver macros). Each macro is a
sequence of existing interactive commands. In interactive mode you can invoke a
macro by typing its name directly.

Example:

```yaml
macros:
  gogo: "g 5; r; g 5; u; g 5"
instructions:
  - gogo
```

On load, the CLI prints any available macros for quick discovery.

## Explicit IDs (Optional)

For readability you can provide explicit IDs for `vertices`, `edges`, `faces`,
and `bodies` using a mapping form instead of lists. This avoids “counting
lines” when referring to faces/bodies and is closer to Evolver’s explicit IDs.

Example:

```yaml
vertices:
  10: [0, 0, 0]
  20: [1, 0, 0]
edges:
  1: [10, 20]
faces:
  100: [1, r2, 3]
bodies:
  0:
    faces: [100]
    target_volume: 1.0
```

## Expression-based energy/constraints

You can attach expression-based energies or hard constraints to entities using
safe arithmetic expressions over `x`, `y`, `z` and global parameters.

Example:

```yaml
vertices:
  0: [1, 2, 3, {expression: "x + y + z"}]
edges:
  1: [0, 1, {expression: "x", expression_measure: "length"}]
```

Hard constraint example:

```yaml
vertices:
  0: [0, 0, 0, {constraint_expression: "x", constraint_target: 1.0}]
```

Define reusable symbols for expressions with a top-level `defines` block:

```yaml
global_parameters:
  angle: 60.0
defines:
  WALLT: "-cos(angle*pi/180)"
edges:
  1: [0, 1, {expression: "-(WALLT*y)", expression_measure: "length"}]
```

Fixed edges imply fixed endpoints: marking an edge as `fixed` also freezes its
two vertices and preserves that behavior through refinement.

## Profiling

To profile each benchmark case and generate per-case `.pstats` outputs:

```bash
./profile.sh
```

This runs `benchmarks/suite.py --profile` and writes results under
`benchmarks/outputs/profiles` by default.

## Gauss-Bonnet diagnostics

For meshes with boundary loops (e.g., disk rims modeled as holes), enable
Gauss-Bonnet drift checks by setting `gauss_bonnet_monitor=true` in
`global_parameters`. Debug logs report the total invariant, the interior
angle-defect sum, and per-loop boundary geodesic sums so drift can be
localized after refinement or remeshing. To exclude facets from the
diagnostic (treating them as “inactive”), set
`gauss_bonnet_exclude: true` in the facet options.

For `gaussian_curvature` energy, boundary loops are supported by default:
the module evaluates the Gauss–Bonnet sum (interior defects + boundary
turning) and multiplies by `gaussian_modulus`. To exclude facets from this
sum, set `gauss_bonnet_exclude: true` in facet options. Enable
`gaussian_curvature_strict_topology=true` to raise on non-manifold edges or
invalid boundary loops (tolerance via `gaussian_curvature_defect_tol`).

## Geometry loading and Input Formats

`parse_geometry` supports both JSON (`.json`) and YAML (`.yaml`, `.yml`) formats.
YAML is recommended for adding comments and using anchors/aliases.

The loader automatically triangulates any facet with more than three edges
using `refine_polygonal_facets`. Triangular facets remain unchanged. The
returned mesh is therefore ready for optimization without further refinement.

For edge‑only or “wire‑frame” geometries, the `faces` section is optional.

## Visualization

Use the visualization module to inspect geometries:

```bash
python -m visualization.cli meshes/cube.json
```

You can also visualize any input mesh directly via `main.py`:

```bash
python main.py -i meshes/cube.json --viz
python main.py -i meshes/cube.json --viz-save outputs/cube.png --viz-no-axes
```

**Live Visualization**: Inside the interactive console, type `lv` (or `live_vis`) to open a real-time plotting window that updates with every minimization step.

The CLI accepts several flags:

- `python -m visualization.cli meshes/cube.json --transparent`
  Draw facets semi‑transparent (now with corrected alpha rendering).

- `python -m visualization.cli meshes/cube.json --no-edges`
  Hide edges and show only filled facets (useful for solid views).

- `python -m visualization.cli meshes/simple_line.json --no-facets --scatter`
  Visualize line‑only meshes: edges only, plus vertex scatter points.

- `python -m visualization.cli meshes/cube.json --no-axes --save outputs/cube.png`
  Remove axes and save the figure to an image file instead of only showing it.

All interactive visualizations are based on the shared `visualization.plotting.plot_geometry`
helper, which ensures equal aspect ratios to prevent distortion.

## Constraint modules

Membrane Solver mirrors Evolver’s mix of penalties and hard constraints. Surface
area (body/facet/global) and volume constraints are configured via
`constraint_modules` plus the relevant `target_*` values on entities or in
`global_parameters`. Perimeter conservation uses a loop of signed edge indices:

Constraint modules that do not provide gradients are enforced geometrically
during minimization and emit a warning at runtime.

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

`pin_to_circle` supports a `fit` mode for moving circular rims. Set
`pin_to_circle_mode: "fit"` (plus `pin_to_circle_group` on the tagged entities)
to keep the rim circular while letting the circle translate/rotate with the mesh.

The new regression tests in `tests/test_perimeter_minimization.py` load the same
square loop (see `tests/sample_meshes.square_perimeter_input`) to verify that
energy drops while the loop returns to the requested perimeter, even after mesh
refinement and equiangulation.

## Bending energy (Willmore/Helfrich)

Enable curvature/bending energy with `bending_modulus` and the `bending` energy
module. Use:
- `bending_energy_model`: `helfrich` (default) or `willmore`
- `spontaneous_curvature` (alias: `intrinsic_curvature`) for Helfrich
- `bending_gradient_mode`: `approx` (fast), `analytic` (validated), or
  `finite_difference` (slow, for verification)


## Development notes

- The evolving backlog/TODO list now lives in `docs/ROADMAP.md` so changes are
  tracked alongside the code. Refer to that document for design sketches,
  medium-term research targets, and open questions.

## Testing & quality checks

- Install deps with `pip install -r requirements.txt` (adds pytest, pytest-cov, Ruff, etc.).
- Run `pytest -q` before and after significant edits. Recent suites add coverage for the volume penalty path (`tests/test_volume_energy.py`) and low-level error handling (`tests/test_exceptions.py`).
- Lint via `ruff check .` (or `pre-commit run -a`) to match CI.
- Coverage hotspots are tracked via `pytest --cov=. --cov-report=term-missing`; focus on geometry entities, module managers, and CLI commands.

## Logging

By default, runs do not write a log file. Use `--log [PATH]` to write logs to a
file; if `PATH` is omitted, the log is written next to the input mesh (same
basename, `.log` suffix).

## Architecture overview

- A living dependency diagram is stored at `docs/mermaid_diagram.txt` (render with any Mermaid-compatible viewer). It ties CLI entry points, geometry entities, runtime managers, and modules together.
- Shared exception types (e.g., `InvalidEdgeIndexError`) live in `exceptions.py` so both geometry and runtime code can surface actionable error messages.
- Runtime components (`runtime/topology.py`, `runtime/refinement.py`, etc.) now have targeted regression tests to keep mesh maintenance stable as new energy/constraint modules land.

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

- `python benchmarks/suite.py` is the main entry point for performance testing. It runs a set of standard scenarios (`cube_good`, `square_to_circle`, `catenoid`, `spherical_cap`, `dented_cube`, `two_disks_sphere`), tracks execution time history in `benchmarks/results.json`, and highlights regressions or improvements.
- `python benchmarks/benchmark_cube_good.py` runs the full `cube_good_min_routine` recipe (minimization, refinement, equiangulation, vertex averaging, etc.) and reports the average wall-clock time.
- `python benchmarks/benchmark_square_to_circle.py` runs the `square_to_circle` scenario (square sheet relaxing to a circle with line tension), serving as a stress test for mesh maintenance operations.
- `python benchmarks/benchmark_catenoid.py` runs the `catenoid` scenario (surface tension minimization between two fixed rings), validating `pin_to_circle` constraints and surface minimization.
- `python benchmarks/benchmark_cap.py` validates the spherical cap scenario, checking apex height, radius, and spherical fit quality against theoretical predictions. It can also be used as a standalone analysis tool: `python benchmarks/benchmark_cap.py outputs/result.json`.
- `python benchmarks/benchmark_dented_cube.py` runs a cube→(nearly) sphere benchmark while keeping one face tagged as a planar/circular dent scaffold.
- `python benchmarks/benchmark_two_disks_sphere.py` runs a sphere scaffold with two small flat disk patches, exercising circle/plane constraints on a closed surface.
