# AGENTS Development Guide

## Repository Map
- `geometry/`: Core entities (Mesh, Vertex) and vectorized geometric helpers.
- `modules/`: Physics logic. `energy/` for potentials, `constraints/` for manifolds.
- `runtime/`: Engine orchestration (Minimizer, EnergyManager, ConstraintManager).
- `commands/`: CLI command implementation (Command pattern).
- `fortran_kernels/`: Performance-critical math (Bending/Tilt).
- `tests/fixtures/`: Scientific reference data. **Do not delete.**

## Fragile & High-Context Files
Avoid reading these in full unless the task explicitly requires structural changes. Use `grep_search` and targeted `read_file` instead.
- `runtime/minimizer.py` (~3200 lines): Core engine. Contains complex state capture and benchmark-specific continuation logic.
- `geometry/entities.py` (~2300 lines): SoA/AoS synchronization and caching. Fragile.
- `modules/energy/bending_tilt_leaflet.py` (~2400 lines): High-density physics with legacy/theory-parity branches.

## Agent Guardrails
1. **Theory Parity**: Do not remove logic branches gated by `theory_parity_lane` or `global_params.get("inner_coupled_update_mode")`. These are required for scientific validation.
2. **Behavior Preservation**: No behavior changes allowed unless explicitly in the directive. For bug fixes, reproduce with a test case first.
3. **Ignore List**: Do not inspect `evolver/`, `docs/tex/`, or `benchmarks/outputs/` unless specifically asked.
4. **No Hacks**: Do not use "fudge factors" or tuning knobs to force agreement with theory.

## Standard Agent Output
When reporting work or investigative findings, use this format:
- **Files Changed/Inspected**: List absolute paths.
- **Variants Migrated/Fixed**: Brief technical summary.
- **Invariants Preserved**: Explicitly state what was *not* changed.
- **Validation**: Specific tests run and result (e.g., `pytest -q tests/test_foo.py` passed).

## Technical Standards
- **Hybrid SoA Pattern**: Numerical optimization MUST use dense NumPy arrays (SoA). Mesh topology remains object-oriented (AoS).
- **Caching**: Always check `mesh._version` or `mesh._topology_version` before recalculating.
- **Hot-Loop Rule**: Avoid Python loops inside energy/gradient calculations. Use vectorized NumPy operations.
- **Testing**: Every change must be verified. Run `pytest -q` on relevant files before and after edits.
