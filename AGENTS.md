# AGENTS Development Guide

## Repository Map
- `geometry/`: Core entities (`Mesh`, `Vertex`) and vectorized geometric helpers.
- `modules/`: Physics logic. `energy/` for potentials, `constraints/` for manifolds.
- `runtime/`: Engine orchestration (`Minimizer`, `EnergyManager`, `ConstraintManager`).
- `runtime/projections/`: Geometric projection helpers (tilt, curved disk).
- `runtime/diagnostics/`: Auditing and logging helpers.
- `commands/`: CLI command implementation (Command pattern).
- `fortran_kernels/`: Performance-critical math (Bending/Tilt).
- `tests/fixtures/`: Scientific reference data. **Do not delete.**

## Fragile & High-Risk Files
Avoid reading these in full unless structural changes are required. Use `grep_search` and targeted `read_file`.
- `runtime/minimizer.py`: Partially decomposed monolithic class. New helpers reside in:
  - `runtime/preconditioners.py`
  - `runtime/diagnostics/audit.py`
  - `runtime/projections/tilt.py`
  - `runtime/projections/curved_disk.py`
- `geometry/entities.py`: Complex SoA/AoS synchronization and version-based caching. Highly fragile.
- `modules/energy/bending_tilt_leaflet.py`: High-density physics with sensitive theory-parity branches.

## Agent Guardrails
1. **Behavior Preservation**: No behavior changes allowed unless explicitly directed. For bug fixes, reproduce with a test case first.
2. **Theory/Benchmark Lanes**: Do not remove logic branches gated by `theory_parity_lane`, `shared_rim_staggered_v1`, or similar. These are required for scientific validation and have dedicated regression tests.
3. **Ignore List**: Do not inspect `evolver/`, `docs/tex/`, or `benchmarks/outputs/` unless specifically asked.
4. **No Hacks**: Do not use "fudge factors" or tuning knobs to force agreement with theory.

## Standard Agent Output
When reporting work or investigative findings, use this format:
- **Files Changed/Inspected**: List absolute paths.
- **Functions Moved/Fixed**: Brief technical summary.
- **Invariants Preserved**: Explicitly state what was *not* changed (e.g., call order, mutation patterns).
- **Validation**: Specific tests run and result (e.g., `pytest -q tests/test_foo.py` passed).

## Technical Standards
- **Hybrid SoA Pattern**: Numerical optimization MUST use dense NumPy arrays (SoA). Mesh topology remains object-oriented (AoS).
- **Caching**: Always check `mesh._version` or `mesh._topology_version` before recalculating.
- **Hot-Loop Rule**: Avoid Python loops inside energy/gradient calculations. Use vectorized NumPy operations.
- **Testing**: Every change must be verified. Run `pytest -q` on relevant files before and after edits.
