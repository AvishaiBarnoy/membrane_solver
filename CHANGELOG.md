# Changelog

All notable changes to this project are documented here. Dates use YYYY-MM-DD.

## [Unreleased]
### Added
- Stability improvements:
  - Implemented a "Safe Step Heuristic" in `backtracking_line_search` to prevent triangle flips (inverted normals) during minimization. Expensive geometric checks are only run for large steps (>30% of min edge length), preserving performance.
  - Added `runtime/topology.py` with `detect_vertex_edge_collisions` and `check_max_normal_change`.
  - `main.py` now runs collision detection after every minimization sequence and logs warnings if the mesh self-intersects.
- Performance optimizations:
  - Replaced `numpy.cross` with a specialized `_fast_cross` helper in `geometry/entities.py`, reducing overhead for small arrays.
  - Vectorized volume gradient computations in `modules/energy/volume.py`.
  - Optimized memory allocation in hot loops (using `np.empty` instead of list comprehensions).
  - Total simulation runtime reduced by ~64% (from ~15.5s to ~5.6s on `cube_good_min_routine`).
- New benchmark suite: `benchmarks/suite.py` runs and compares multiple scenarios (`cube_good`, `square_to_circle`) and tracks performance history.
- Automatic target-area detection on bodies/facets and regression tests (square with area constraint, tetra with volume constraint).
- Benchmarks now run in read-only sandboxes (no temp files); README/manual updated with benchmark usage.
- Integration tests covering the cube penalty scenario (energy decrease, volume preservation, refine+equiangulate validity) and parsing tests for the interactive `rN` command.
- Perimeter constraint coverage: `tests/test_perimeter_minimization.py` now drives a constrained square loop through minimization, refinement, and equiangulation, checking that perimeter returns to its target while area stays near 1 even in the presence of small discretisation errors.
- A dedicated `visualization` package with a reusable `plot_geometry` helper and a CLI (`visualize_geometry.py`) supporting toggling facets/edges, transparency, axes, scatter, and saving figures.
- Command-line line-tension controls: `--line-tension` and `--line-tension-edges` on `main.py` allow tagging edges with `line_tension` energy without editing JSON.

### Changed
- Volume handling:
  - `lagrange` mode auto-loads the `volume` constraint and uses gradient projection.
  - `penalty` mode auto-loads the `volume` energy module instead, restoring quadratic penalty behaviour.
- `main.py` now imports `visualize_geometry` lazily so headless runs/benchmarks don’t crash when Matplotlib can’t create cache dirs.
- `benchmark_cube_good.py` runs directly on `cube_good_min_routine.json` without writing outputs.
- Interactive mode: `rN` now repeat-refines without scripting loops, `i` is a shorthand for the properties command, and README/manual document the updated syntax. The old README TODO list was moved to `docs/ROADMAP.md`.
- Visualization: `visualize_geometry.py` is now a thin wrapper around the `visualization` CLI, and facet rendering skips <3-edge facets so line-only meshes can be displayed cleanly.

### Fixed
- Fixed `square_to_circle.json`: Added missing instruction list and updated it to interleave minimization with mesh maintenance (`r`, `g10`, `u`, `V`) to prevent mesh tangling/overlap during large deformations.
- `load_data` accepts `Path` objects (needed for tests writing tmp JSONs).
- `cube_good_min_routine` converges again (energy ≈ 4.85) when run under penalty mode.

## [0.3.0] - 2025-12-05
### Added
- Body/facet surface-area constraints with Lagrange-style enforcement.
- Regression tests for area constraints.

### Changed
- Default volume mode switched to `lagrange` hard constraint; penalty mode opt-in via `global_parameters` or `--volume-mode`.
- Constraint manager enforces fixed volume after each accepted step.

## [0.2.0] - 2025-07-07
### Added
- Conjugate Gradient stepper with Armijo line search.
- Equiangulation (`u` command) and refinement improvements.
- Initial cube benchmark script.

### Changed
- CLI defaults to interactive mode; logging/quiet flags improved.
- Refinement respects `no_refine` and preserves normals.

## [0.1.0] - 2025-05-15
Initial release with core mesh entities, JSON I/O, surface/volume energies, constraint manager, gradient-descent stepper, vertex averaging, refinement, and test suite.
