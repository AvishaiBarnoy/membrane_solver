# Changelog

All notable changes to this project will be documented in this file.

The format is inspired by [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
but dates and versions are intentionally kept light for a research‑oriented codebase.

## Unreleased

### Added

- `CHANGELOG.md` to track high‑level changes over time.
- Logging configuration updates:
  - Default log level set to `INFO` for file logging.
  - New `--debug` CLI flag in `main.py` to enable verbose `DEBUG` logging.
  - Switched to `logging.FileHandler(mode="w")` so each run overwrites the previous log file instead of appending.

### Existing (backfilled summary)

- Core mesh and geometry representation (`Vertex`, `Edge`, `Facet`, `Body`, `Mesh`) with:
  - Area and area‑gradient for facets.
  - Volume and volume‑gradient for bodies.
  - Connectivity maps (vertex↔edges↔facets) and basic validation.
- JSON geometry I/O:
  - Input format with per‑entity `options` for `energy`, `constraints`, and local parameters.
  - `global_parameters` block with defaults for surface tension, volume stiffness, etc.
  - Automatic triangulation of polygonal facets via centroid‑based refinement.
- Energy modules:
  - Surface tension energy and gradient (`modules/energy/surface.py`).
  - Volume penalty energy and gradient (`modules/energy/volume.py`).
- Constraint handling:
  - `fixed` vertices and constraint projection for gradients/positions.
  - Infrastructure for constraint modules (`modules/constraints`, `ConstraintModuleManager`).
- Runtime / optimization:
  - `Minimizer` coordinating energy, constraints, and steppers.
  - Gradient Descent and Conjugate Gradient steppers with Armijo backtracking line search.
  - Mesh refinement, equiangulation, and vertex averaging utilities.
- CLI and interactive driver (`main.py`):
  - Instruction‑based and interactive control (`gN`, `gd`/`cg`, `r`, `V`, `u`, `visualize`, `save`, etc.).
  - Basic visualization script for inspecting meshes.
- Test suite covering core geometry, I/O, energy modules, refinement, and steppers.

