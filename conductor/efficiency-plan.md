# Efficiency Improvements Plan

## Objective
Implement a series of performance optimization strategies identified during codebase analysis to improve the simulation speed of `membrane_solver`.

## Key Files & Context
- **Global Parameters**: `core/parameters/global_parameters.py`
- **Tilt Relaxation**: `runtime/steppers/tilt_relaxation.py`
- **Minimizer**: `runtime/minimizer.py`
- **Build Configuration**: `setup.py`, CI configuration
- **Documentation**: `README.md`, `manual.md`, `opt_ideas.md`

## Implementation Steps

### 1. Solver Choice
- Update the default tilt solver from Gradient Descent (`gd`) to Conjugate Gradient (`cg`).
- Modify `core/parameters/global_parameters.py` (or the default dictionary logic) to change the fallback value for `tilt_solver` to `"cg"`.
- Update relevant documentation (`README.md`, `manual.md`) to reflect the new default.

### 2. SoA (Structure of Arrays) Enforcement
- Ensure that the "Scatter-Gather" SoA architecture is enforced project-wide.
- Audit active energy and constraint modules to guarantee they implement the `compute_energy_and_gradient_array` API to avoid the slow dictionary fallback.
- Update documentation and `opt_ideas.md` to reflect the strict SoA enforcement.

### 3. Preconditioning
- Enable Jacobi preconditioning by default for the Conjugate Gradient tilt solver.
- Modify `core/parameters/global_parameters.py` to ensure `tilt_cg_preconditioner` defaults to `"jacobi"`.

### 4. Mesh Quality
- Enable automated mesh quality repair to maintain stability and prevent numerical stiffness during minimization.
- Add or update default parameter keys in `core/parameters/global_parameters.py`:
  - `mesh_quality_auto_repair_enabled`: `True`
  - `mesh_quality_auto_repair_every`: (e.g., 10 or 50)
  - `mesh_quality_aspect_threshold`: (e.g., a reasonable aspect ratio threshold)

### 5. Compile Kernels
- Ensure high-performance Fortran kernels are built and utilized.
- Update installation instructions in `README.md`/`manual.md` to emphasize `MEMBRANE_SOLVER_BUILD_EXT=1 pip install -e .`
- Optionally add a warning log in the Python code if the kernels fail to load, urging the user to build the extensions for performance.

## Verification & Testing
- Run the full test suite (`pytest -q`) to ensure no regressions occur due to the parameter changes.
- Benchmark typical runs (e.g., `benchmarks/benchmark_tilt_relaxation.py`) before and after the changes to quantify the performance improvements.
- Verify locally that the Fortran kernels load correctly when built.
