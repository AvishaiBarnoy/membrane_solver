# Comprehensive Refactoring Plan: Bending, Minimizer, and Constraints

## Background & Motivation
Following the successful decomposition of `bending_tilt_leaflet.py`, several other core modules remain monolithic, hindering readability, maintainability, and future performance optimizations (such as Fortran/C offloading). `bending.py`, `minimizer.py`, and `rim_slope_match_out.py` are currently the largest and most complex files in the codebase.

## Scope & Impact
This plan covers the structural decomposition of three major components:
1.  **Energy Module:** `modules/energy/bending.py` (~950 lines)
2.  **Runtime Orchestration:** `runtime/minimizer.py` (~2800 lines)
3.  **Constraint Module:** `modules/constraints/rim_slope_match_out.py` (~2000 lines)

The impact will be purely structural. No numerical or physical behavior will be altered.

## Proposed Solution
We will systematically decompose these files using the same established patterns from the `bending_tilt_leaflet` refactor:
- Extract parameter resolution and static data gathering into payload modules.
- Extract complex mathematical operations (e.g., analytic gradient backpropagation) into dedicated gradient modules.
- Extract isolated algorithmic loops (e.g., tilt relaxation) into dedicated stepper/manager modules.

## Alternatives Considered
-   **Inline Optimization:** Attempting to optimize the monolithic files directly via Numba or Cython. *Rejected* because the files are too tangled; separating data gathering from computation is a prerequisite for clean offloading.
-   **Rewrite in C++:** *Rejected* due to scope. Python remains the high-level orchestrator.

## Implementation Plan

### Phase 1: Decompose `modules/energy/bending.py`
*   **Goal:** Apply the `bending_tilt` decomposition pattern to the base bending energy.
*   **Tasks:**
    *   Extract cotangent gradient and area variation math into `modules/energy/bending_gradient.py`.
    *   Extract effective area computation and parameter resolution into `modules/energy/bending_utils.py` or `bending_payload.py`.
    *   Leave `bending.py` as a lean orchestrator for the `compute_energy_and_gradient` loop.

### Phase 2: Extract Tilt Relaxation from `runtime/minimizer.py`
*   **Goal:** Simplify the `Minimizer` class by moving tilt-specific inner loops to a dedicated manager.
*   **Tasks:**
    *   Create `runtime/steppers/tilt_relaxation.py` (or similar).
    *   Move `_relax_tilts`, `_relax_leaflet_tilts`, and associated tilt-gradient accumulation loops out of the main minimizer.
    *   Update `minimizer.py` to delegate to this new manager when tilt relaxation is required.

### Phase 3: Decompose `modules/constraints/rim_slope_match_out.py`
*   **Goal:** Break down the massive 2000-line constraint module.
*   **Tasks:**
    *   Extract geometric fitting (e.g., `_fit_plane_normal`, `_orthonormal_basis`) into `geometry/fitting.py` (or a dedicated utilities file).
    *   Extract arc-length parameterization and interpolation logic.
    *   Extract the dense joint-gradient assembly logic into a dedicated submodule.
    *   Extract diagnostic reporters (e.g., `matching_ring_diagnostics`, `coarse_rim_family_diagnostics`) into `tools/diagnostics/` or a reporting submodule.

## Verification
*   After each phase, run the full test suite (`pytest -q`) to ensure exact numerical parity.
*   Specifically run the `regression` and `e2e` test marks.
*   Ensure that no new Python loops are introduced in hot paths; verify that array-based broadcasting is maintained.

## Migration & Rollback
*   Each phase will be performed on a separate feature branch.
*   If any phase causes insurmountable CI failures, the branch will be abandoned and the previous state remains intact on `main`. No data migration is necessary.
