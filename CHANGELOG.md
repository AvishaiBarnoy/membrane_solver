# Changelog

All notable changes to this project are documented here. Dates use YYYY-MM-DD.

## [Unreleased]
### Added
- Automatic target-area detection on bodies/facets and regression tests (square with area constraint, tetra with volume constraint).
- Benchmarks now run in read-only sandboxes (no temp files); README/manual updated with benchmark usage.
- Integration tests covering the cube penalty scenario (energy decrease, volume preservation, refine+equiangulate validity) and parsing tests for the interactive `rN` command.

### Changed
- Volume handling:
  - `lagrange` mode auto-loads the `volume` constraint and uses gradient projection.
  - `penalty` mode auto-loads the `volume` energy module instead, restoring quadratic penalty behaviour.
- `main.py` now imports `visualize_geometry` lazily so headless runs/benchmarks don’t crash when Matplotlib can’t create cache dirs.
- `benchmark_cube_good.py` runs directly on `cube_good_min_routine.json` without writing outputs.
- Interactive mode: `rN` now repeat-refines without scripting loops, `i` is a shorthand for the properties command, and README/manual document the updated syntax. The old README TODO list was moved to `docs/ROADMAP.md`.

### Fixed
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
