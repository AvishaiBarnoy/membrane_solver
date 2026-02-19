# Changelog

All notable changes to this project are documented here. Dates use YYYY-MM-DD.

## [Unreleased]
### Added
- CI now runs categorized test subsets via pytest markers: `unit`, `regression`, and `e2e`.
- Flat one-leaflet disk benchmark reproduction tooling:
  - Theory utility: `tools/diagnostics/flat_disk_one_leaflet_theory.py` (exact formulas from `docs/tex/1_disk_flat.tex`).
  - Reproduction CLI: `tools/reproduce_flat_disk_one_leaflet.py` with `--outer-mode disabled|free`, refinement and theta-scan controls, and YAML reporting.
  - Free-mode report now includes deterministic outer-leaflet perturbation-decay probe metrics (`outer_decay_probe_max_before/after`) to confirm undriven outer-tilt relaxation.
  - Baseline acceptance fixtures and CLI parity test for both modes (`tests/fixtures/flat_disk_one_leaflet_disabled_baseline.yaml`, `tests/fixtures/flat_disk_one_leaflet_free_baseline.yaml`, `tests/test_reproduce_flat_disk_one_leaflet_acceptance.py`).
  - Acceptance/unit/regression tests for TeX parity, planarity, one-leaflet decay profile, and free-outer-leaflet consistency (`tests/test_flat_disk_one_leaflet_theory_unit.py`, `tests/test_flat_disk_one_leaflet_benchmark_e2e.py`).
  - Added inner-leaflet KH-style split smoothness energy `tilt_splay_twist_in` with explicit splay/twist moduli (`tilt_splay_modulus_in`, `tilt_twist_modulus_in`) and default zero twist modulus.
  - Flat disk reproduction CLI now supports `--smoothness-model dirichlet|splay_twist` (default `dirichlet`); `splay_twist` uses `tilt_splay_twist_in` in the benchmark harness.
  - Flat disk reproduction CLI now supports `--theta-mode scan|optimize` (default `scan`) so `theta_B` can be either scanned or optimized as a scalar DOF from one command.
  - Flat disk reproduction CLI now supports `--theta-mode optimize_full`, which performs optimize + local reduced-energy polish and reports both raw and polished `theta_B` values for deterministic “full optimization” reproducibility.
  - Flat disk optimize presets now include `full_accuracy_r3` for refine-3 full-optimization runs, and reports now include `parity.recommended_mode_for_theory` to indicate whether `optimize` or `optimize_full` best matches theory by a balanced parity score.
  - Flat disk benchmark now accepts `--splay-modulus-scale-in` (default `1.0`) to scale inner `tilt_splay_modulus_in` in `splay_twist` mode for theory-parity tuning workflows.
  - Flat disk theory utility now includes physical-to-dimensionless conversion (`physical_to_dimensionless_theory_params`) so benchmark runs can be specified in physical units and converted with explicit energy/length scales.
  - Flat disk reproducer now supports `--parameterization legacy|kh_physical` plus physical scaling CLI controls (`--kappa-physical`, `--kappa-t-physical`, `--length-scale-nm`, `--radius-nm`, `--drive-physical`) and reports these unit metadata fields in YAML output.
  - Flat disk baseline acceptance now includes a dedicated `kh_physical` lane fixture (`tests/fixtures/flat_disk_one_leaflet_kh_physical_disabled_baseline.yaml`) in addition to the legacy disabled/free baselines, with lane-specific expectations for parity pass/fail.
  - Added KH physical-lane per-theta term audit utility (`tools/diagnostics/flat_disk_kh_term_audit.py`) to report mesh/theory split terms (internal/contact/total) at user-selected `theta_B` values.
  - Flat disk reproduction CLI default refinement is now `--refine-level 2` (was `1`) to keep default parity in the `<2x` acceptance range with full rim continuity enforced.
  - Flat disk optimize-mode defaults are now lighter and parity-stable at the default refinement (`theta_optimize_steps=20`, `theta_optimize_inner_steps=20`, `theta_optimize_delta=2e-4`).
  - Flat disk reproduction CLI now supports `--optimize-preset fast_r3` for faster `refine_level>=3` optimize-mode runs, with report metadata indicating whether the preset was applied.
  - Added benchmark coverage for refine-3 optimize preset runtime-vs-parity tradeoff (`tests/test_flat_disk_one_leaflet_optimize_preset_benchmark.py`).
- Interactive CLI:
  - Tab completion for command/macro names (TTY only).
  - Compound commands via semicolons (e.g. `g50; V3; g10`).
  - `s bilayer` visualization mode (parity with `lv bilayer`).
- Performance tooling: `tools/profile_tilt.py` to profile tilt relaxation hot-loops.
- Performance tooling: `tools/profile_macro_hotspots.py` for per-step macro timing plus optional `cProfile` capture of a selected command.
- New benchmark case: `benchmarks/benchmark_tilt_relaxation.py` (tilt relaxation hot-loop timing).
- Docs: clarified tilt controls (`tilt_solve_mode`, `tilt_solver`, inner-step knobs), documented `show_edges`, and clarified torus local-vs-integrated Gaussian curvature interpretation.
- Constraint alias support: `pin_surface_group_to_shape` now maps to `pin_to_plane` (including mode/group/normal/point option aliases).
- Refactor: extracted minimizer diagnostic-state and tilt-fixed-mask helpers into `runtime/minimizer_helpers.py` (behavior-preserving maintainability split).
- Refactor: extracted reduced-energy line-search tilt-relax wrapper logic from `Minimizer` into `runtime/minimizer_helpers.py` (behavior-preserving maintainability split).
- Refactor: extracted reusable triangle-array math helpers from `geometry/entities.py` into `geometry/triangle_ops.py` (behavior-preserving maintainability split).
- Refactor: extracted geometry cache-validity predicates from `geometry/entities.py` into `geometry/cache_checks.py` (behavior-preserving maintainability split).
- Refactor: extracted tilt-projection math helpers from `runtime/minimizer.py` into `runtime/tilt_projection.py` (behavior-preserving maintainability split).
- Refactor: extracted triangle-row cache construction helpers from `geometry/entities.py` into `geometry/triangle_rows.py` (behavior-preserving maintainability split).
- Refactor: extracted optional leaflet axisymmetric projection flow from `runtime/minimizer.py` into `runtime/tilt_projection.py` helper wiring (behavior-preserving maintainability split).
- Refactor: extracted leaflet trial tilt projection/restore helper from `runtime/minimizer.py` into `runtime/tilt_projection.py` to reduce duplicate inner-loop setup code.
- Refactor: extracted geometry cache-write assignment helpers from `geometry/entities.py` into `geometry/cache_writes.py` (behavior-preserving maintainability split).
- Rim source energies now follow fitted pin-to-circle rims (`pin_to_circle_mode: fit`) as they translate in space.
- KH-pure tilt benchmark variants without smoothness regularization (`meshes/tilt_benchmarks/kh_pure_*`).
- Tilt benchmark runner script (`tools/tilt_benchmark_runner.py`) for energy/tilt/divergence summaries.
- Tilt benchmark runner smoke test to ensure tilt meshes load and metrics print (`tests/test_tilt_benchmark_runner.py`).
- KH-pure refinement stability regression coverage (`tests/test_kh_pure_benchmarks.py`).
- Tilt source decay benchmark meshes (`meshes/tilt_benchmarks/tilt_source_rect.yaml`, `meshes/tilt_benchmarks/tilt_source_annulus.yaml`).
- Dual-leaflet tilt plumbing: `tilt_in`/`tilt_out` fields, IO round-trip, refinement inheritance, and live-vis color modes.
- Bilayer tilt energies (`tilt_in`, `tilt_out`, `tilt_smoothness_in/out`) with optional `tilt_coupling`, plus leaflet-aware tilt relaxation in the minimizer.
- Leaflet-specific bending-tilt coupling modules (`bending_tilt_in`, `bending_tilt_out`).
- E2E tilt relaxation tests for leaflet-specific bending-tilt modules.
- Example bilayer tilt decay meshes under `meshes/bilayer_tilt/`.
- `lv bilayer` mode to visualize outer/inner leaflet tilt magnitudes on dual surfaces.
- Bilayer tilt example mesh with opposing leaflet sources (`meshes/bilayer_tilt/tilt_bilayer_rect_opposite.yaml`).
- Kozlov/Barnoy contact-term parameterization for rim sources via `tilt_rim_source_contact_*` (Δε, a, h → γ), used by `tilt_rim_source_in/out/bilayer`.
- Single-leaflet rim-source curvature-induction regression (`tests/test_single_leaflet_curvature_induction.py`) using `meshes/caveolin/kozlov_1disk_3d_tensionless_single_leaflet_soft_source.yaml` as a fast induction case.
- Single-leaflet disk+outer rim example mesh with shape relaxation (`meshes/caveolin/kozlov_1disk_3d_tensionless_single_leaflet_source.yaml`).
- Single-leaflet 1-disk diagnostics + regression coverage (`tools/diagnose_1disk_3d_single_leaflet.py`, `tests/test_kozlov_1disk_3d_single_leaflet_behavior.py`).
- Disk-profile single-leaflet target module and mesh (`modules/energy/tilt_disk_target_in.py`, `meshes/caveolin/kozlov_1disk_3d_tensionless_single_leaflet_profile.yaml`).
- Added outer-leaflet disk target module and bilayer profile mesh (`modules/energy/tilt_disk_target_out.py`, `meshes/caveolin/kozlov_1disk_3d_tensionless_bilayer_profile.yaml`).
- Hard rim-slope matching constraint module with tilt projection (`modules/constraints/rim_slope_match_out.py`) and regression coverage (`tests/test_rim_slope_match_out_constraint.py`).
- `pin_to_plane` now supports `slide`/`fit` modes with group-based plane fitting, plus regression coverage (`tests/test_pin_to_plane_slide.py`).
- Hard per-leaflet full in-plane rim continuity constraint for multi-disk setups (`modules/constraints/tilt_vector_match_rim.py`) and regression coverage (`tests/test_tilt_vector_match_rim_constraint.py`).
- Small-drive 1-disk tensionless regression + macro E2E coverage (`tests/test_kozlov_1disk_3d_tensionless_small_drive_regression.py`, `tests/test_e2e_kozlov_1disk_3d_small_drive_macro.py`) and benchmark-suite entry (`benchmarks/benchmark_kozlov_1disk_3d_tensionless.py`, `tools/suite.py`).
- Added SciPy as a project dependency (`requirements.txt`, `pyproject.toml`).
- Feature-branch guard script and pre-commit hook (`tools/ensure_feature_branch.sh`).
- **Kozlov–Hamm Tilt Coupling**: Implemented coupled bending+tilt energy integral $\frac{1}{2} \int \kappa (2H - c_0 + \text{div}(t))^2 dA$ in `modules/energy/bending_tilt.py`.
- **Tilt Solve Modes**: Added `nested` and `coupled` relaxation modes in `Minimizer` to optimize the tilt field alongside surface geometry.
- **Enhanced Tilt Visualization**:
  - `lv tilt` / `lv div`: Heatmap visualization of tilt magnitude and divergence.
  - `--tilt-arrows`: Vector field overlay showing vertex tilt directions in 3D.
  - Interactive colorbar support for scalar fields in live visualization.
- **Repo Hygiene & Packaging**:
  - Comprehensive `.gitignore` for compiled extensions (`*.so`, `*.dylib`), build outputs, and caches.
  - Standardized packaging with `pyproject.toml` and `setup.py`.
  - Explicit build helper: `python -m membrane_solver.build_ext` for f2py kernels.
  - Hardened CI: Python version matrix (3.11, 3.12) and packaging-based test execution.
- Optional Fortran (f2py) surface-energy kernel (`fortran_kernels/surface_energy.f90`) with automatic runtime fallback to NumPy when not built/available.
- Optional Fortran (f2py) tilt kernels (`fortran_kernels/tilt_kernels.f90`) for P1 divergence and curvature accumulation with runtime fallback.
- Closed-body outwardness validation (signed volume check) with open-body exemption for cases like droplets on hard planes.
- Milestone C: 3D Kozlov annulus mesh with bilayer tilt↔curvature coupling (`meshes/caveolin/kozlov_annulus_milestone_c_soft_source.yaml`) plus E2E regression tests (including leaflet sign flip).
- Analytic regression benchmark for the 1-disk model (tensionless: distal/proximal tilts match; `docs/tex/1_disk_3d.pdf`).
- θ_B bilayer rim source energy module `tilt_rim_source_bilayer` (single source definition; equivalent to `tilt_rim_source_in` + `tilt_rim_source_out` with equal parameters).
- Acceptance E2E parity test for the named free-disk mesh (`tests/test_kozlov_free_disk_mesh_theory_parity_acceptance_e2e.py`) against `docs/tex/1_disk_3d.tex` with 15% tolerance on `theta_B` and energy plus convergence-trend checks.

### Fixed
- Reduced repeated geometry work in `tilt_splay_twist_in` by reusing mesh-cached P1 triangle shape gradients and triangle normals during inner tilt loops, improving flat-disk `refine_level=3` optimize runtime while preserving parity.
- Reduced sparse leaflet-KKT per-iteration overhead in `runtime/constraint_manager.py` by precomputing the active-space projection matrix when the cached solve operator is available, improving refine-3 flat optimize runtime with unchanged parity.
- Reduced `tilt_thetaB_boundary_in` enforcement overhead by applying in-place updates on boundary rows and touching tilt-cache versioning, avoiding full inner-tilt array copy/scatter per relaxation call.
- Reduced sparse-row KKT assembly overhead in `runtime/constraint_manager.py` by accumulating directly into a `(N,3)` reshaped view instead of constructing expanded flat-index buffers per constraint row.
- Cached leaflet interior/base-term masks in `modules/energy/bending_tilt_leaflet.py` to avoid rebuilding boundary/group exclusions across repeated bending-tilt evaluations.
- Cached geometry-derived tilt-constraint payloads for `rim_slope_match_out` and `tilt_thetaB_boundary_in` when geometry cache is active, reducing repeated per-call setup in inner KKT tilt projection loops.
- Reduced per-call allocation overhead in `Mesh.set_tilts_in_from_array` and `Mesh.set_tilts_out_from_array` by storing cache-backed row views instead of per-vertex copy buffers.
- Cached leaflet-absence vertex masks in `modules/energy/leaflet_presence.py` with mesh-version and preset-token invalidation, removing repeated per-call vertex scans from hot loops.
- Added vertex-group row-index caching for `tilt_thetaB_contact_in`, `tilt_thetaB_boundary_in`, and `rim_slope_match_out` to reduce repeated per-vertex scans in inner minimization loops.
- Partitioned `_compute_effective_areas` cache entries by caller token (`modules/energy/bending.py`) so dual-leaflet evaluations no longer evict each other’s cached effective areas.
- Reduced leaflet-tilt synchronization overhead in the minimizer by using cache-backed array updates during thetaB candidate scans and tilt-relaxation constraint enforcement (`runtime/minimizer.py`), while preserving final energy/thetaB results in regression tests.
- Cached `bending_tilt` boundary-group row collection in `modules/energy/bending_tilt_leaflet.py` to avoid repeated per-vertex scans during inner tilt relaxation loops.
- Geometry-freeze caching now reuses curvature/area weights during tilt relaxation when positions are fixed.
- Reduced repeated per-vertex rebinding overhead in `Mesh.set_tilts_in_from_array` and `Mesh.set_tilts_out_from_array` by tracking vertex row-binding version state, while keeping raw vertex tilt attributes synchronized for cache-correct energy parity.
- Optimized sparse leaflet tilt KKT projection in `runtime/constraint_manager.py` by projecting over compressed active DOFs for row-sparse constraints instead of building a full stacked dense constraint matrix each call.
- Cached ordered boundary geometry payloads in `modules/energy/tilt_thetaB_contact_in.py` when geometry cache is active, reducing repeated angle ordering and arc-length setup in inner θ_B contact energy evaluations.
- Cached sparse leaflet tilt KKT projection operators (`active_cols`, compressed `C`, and `A^{-1}`) in `runtime/constraint_manager.py` across geometry-cache-active iterations to avoid rebuilding/factorizing the same constraint system every inner loop call.
- Refined sparse leaflet KKT operator cache invalidation in `runtime/constraint_manager.py` to key on sparse row-payload content (instead of full global-parameter state), improving cache-hit stability while preserving correctness on payload changes.
- Sparse leaflet KKT operator caches now store a Cholesky factorization of the compressed Gram matrix (`A = C C^T`) with direct-solve fallback, avoiding repeated explicit matrix inversion in operator builds.
- `energy` breakdown output now separates internal energy vs external work terms (sources) and supports `energy ref` for reference-state deltas.
- **Tilt Persistence**: Corrected `save_geometry` to persist `tilt` and `tilt_fixed` vertex properties to output JSON files.
- **Live-Vis Warnings**: Silenced `plt.pause` warnings on headless CI backends.
- **Hybrid SoA Architecture**: Refactored the minimization pipeline to use a Structure-of-Arrays pattern. Optimization now runs on dense NumPy arrays, eliminating O(N) Python dictionary overhead and resulting in a **3.5x speedup** for large meshes.
- **Gauss-Bonnet diagnostics**: Added `runtime/diagnostics/gauss_bonnet.py` to monitor Gaussian curvature invariants on open surfaces with boundary loops, including per-loop boundary geodesic sums and drift checks during minimization.
  - Supports excluding facets from the diagnostic via `gauss_bonnet_exclude` in facet options to model effective holes.
- Added `print energy breakdown` to show per-module energy contributions (including Gaussian curvature when enabled).
- CLI now prints defined macros on load and supports `print macros`.
- Added interactive `history` command to list commands entered in the current session.
- Added Gauss-Bonnet sample meshes: `meshes/gauss_bonnet_disk.json`, `meshes/gauss_bonnet_disk_excluded.yaml`, `meshes/gauss_bonnet_torus.yaml`, `meshes/boxy_torus_start.yaml`.
- Added `meshes/hemisphere_start.yaml` (sphere-with-hole starter mesh) for Gauss–Bonnet boundary checks.
- **Bending Energy Module**: Implemented the squared mean curvature integral (Willmore energy) via the discrete Laplace-Beltrami operator.
  - Uses **Cotangent Weights** for geometric accuracy.
  - Implemented **Mixed Voronoi Areas** (Meyer et al. 2003) for dual-area stability on distorted meshes.
  - Robust **bi-Laplacian force** implementation for energy-consistent minimization.
  - **Boundary Filtering**: Curvature is ignored for boundary vertices, ensuring flat planar patches correctly evaluate to zero bending energy.
- **Gaussian curvature (Helfrich \bar{kappa} term)**: Added `modules/energy/gaussian_curvature.py` for closed vesicles with constant `gaussian_modulus` (energy is Gauss–Bonnet topological constant; zero shape gradient).
- **Analytic bending gradient**: Added an analytic backpropagation gradient for the discrete Willmore/Helfrich bending energy and validated it against finite differences (`tests/test_bending_finite_difference.py`).
- **Numerical Consistency Suite**: Added `tests/test_numerical_consistency.py` with automated **Finite Difference** checks for all energy modules (`surface`, `volume`, `line_tension`, `bending`), ensuring that analytical gradients perfectly match energy slopes.
- `print energy`: New CLI command to display the current total energy of the mesh.
- Reusable curvature helper: `geometry/curvature.py` provides vectorized curvature and area data for use by any module.
- Evolver-style input `macros`: define named command sequences and invoke them directly by name in interactive mode or from `instructions`.
- Optional explicit-ID input forms for `vertices`, `edges`, `faces`, and `bodies` (mapping form), improving readability when hand-authoring meshes.
- Surface radius of gyration reporting in `properties` output, plus the `--radius-of-gyration` CLI flag.
- Regression tests:
  - `tests/test_volume_energy.py` covers both penalty/laprange code paths in `modules/energy/volume.py`, including gradient accumulation.
  - `tests/test_exceptions.py` asserts that `InvalidEdgeIndexError` is raised when geometry routines see 0-based or missing edge IDs.
- **CLI Enhancements**:
  - `print [entity] [filter]`: Query geometry (e.g., `print edges len > 0.5`).
  - `set [target] [value]`: Modify global parameters or entity properties (e.g., `set vertex 0 fixed true`).
  - `live_vis` / `lv`: Real-time 3D visualization during minimization steps.
  - `--viz` / `--viz-save`: Visualize an input JSON and exit (no minimization).
  - `--compact-output-json`: Save output JSON in compact (single-line) form.
  - `--debugger`: Drop into `ipdb`/`pdb` post-mortem on uncaught exceptions.
- Debugging docs: added `docs/DEBUGGING.md` and `ipdb` to `requirements.txt`.
- **Spherical Cap Benchmark**: Added `benchmarks/benchmark_cap.py` to verify spherical cap geometry (apex height, radius, RMSE) against theoretical values. This script doubles as a standalone analysis tool.
- **Catenoid Benchmark**: Added `benchmarks/benchmark_catenoid.py` and `meshes/catenoid_good_min.json` to validate surface tension minimization between two fixed rings.
- New benchmarks:
  - `benchmarks/benchmark_dented_cube.py` using `meshes/bench_dented_cube.json`.
  - `benchmarks/benchmark_two_disks_sphere.py` using `meshes/bench_two_disks_sphere.json`.
- **YAML Support**: `load_data` now supports `.yaml` and `.yml` files for mesh definitions, enabling comments and anchors/aliases.
- **JSON Presets**: Added support for a `definitions` block in input files. Entities can use `"preset": "definition_name"` to inherit properties, reducing duplication for constraints and options.
- Stability improvements:
  - Implemented a "Safe Step Heuristic" in `backtracking_line_search` to prevent triangle flips (inverted normals) during minimization. Expensive geometric checks are only run for large steps (>30% of min edge length), preserving performance.
  - Added `runtime/topology.py` with `detect_vertex_edge_collisions` and `check_max_normal_change`.
  - `main.py` now runs collision detection after every minimization sequence and logs warnings if the mesh self-intersects.
- Performance optimizations:
  - Replaced `numpy.cross` with a specialized `_fast_cross` helper in `geometry/entities.py`, reducing overhead for small arrays.
  - Vectorized volume gradient computations in `modules/energy/volume.py`.
  - Optimized memory allocation in hot loops (using `np.empty` instead of list comprehensions).
  - Total simulation runtime reduced by ~64% (from ~15.5s to ~5.6s on `cube_good_min_routine`).
- New benchmark suite: `tools/suite.py` runs and compares multiple scenarios (`cube_good`, `square_to_circle`, `catenoid`, `spherical_cap`) and tracks performance history.
- **CI**: Added benchmark suite execution to the CI workflow.
- **Cleanup**: Removed broken `modules/gaussan_curvature.py` and obsolete methods in `energy_manager.py`.
- Automatic target-area detection on bodies/facets and regression tests (square with area constraint, tetra with volume constraint).
- Cached triangle row indices and position-array reuse to avoid per-step reallocation during surface energy/gradient evaluation.

### Changed
- Default `bending_gradient_mode` is now `"analytic"` (more accurate and typically easier to minimize than `"approx"`).
- Benchmark suite profiling: `python tools/suite.py --profile` emits per-case `.pstats` files (plus optional text summaries) for easier A/B performance analysis.
- Fixed edges now freeze their endpoint vertices (including during refinement), matching Evolver-style behavior.
- Expression-based energy and constraints with safe evaluation and numeric gradients.
- Increased `tilt_thetaB_optimize_delta` to `0.03` in `meshes/caveolin/kozlov_1disk_3d_tensionless_single_leaflet_profile_hard_rim_R12_free_disk.yaml` to reach the theory-scale `theta_B` within the bounded acceptance workflow.
- Expression `defines`: top-level symbols evaluated from expressions (usable in energy/constraint expressions).
- Benchmarks now run in read-only sandboxes (no temp files); README/manual updated with benchmark usage.
- Documentation callouts: README, `manual.md`, and `docs/ROADMAP.md` now highlight testing requirements, shared exceptions, and upcoming placeholder modules for curvature/tilt energies.
- Integration tests covering the cube penalty scenario (energy decrease, volume preservation, refine+equiangulate validity) and parsing tests for the interactive `rN` command.
- Perimeter constraint coverage: `tests/test_perimeter_minimization.py` now drives a constrained square loop through minimization, refinement, and equiangulation, checking that perimeter returns to its target while area stays near 1 even in the presence of small discretisation errors.
- A dedicated `visualization` package with a reusable `plot_geometry` helper and a CLI (`visualize_geometry.py` was removed in favor of `visualization/cli.py` or direct use).
- Command-line line-tension controls: `--line-tension` and `--line-tension-edges` on `main.py` allow tagging edges with `line_tension` energy without editing JSON.
- `pin_to_circle` fit mode: `pin_to_circle_mode: "fit"` keeps a tagged rim circular while allowing the circle to translate/rotate with the mesh; supports `pin_to_circle_group` for multiple rims.
- Added a movable circular-rim demo mesh: `meshes/bench_moving_circle_fit.yaml` (now a real facet with stabilized rim spacing).

### Changed
- `history` command now records expanded macro commands and skips unknown instructions.
- Compiled Fortran kernels are built explicitly (e.g. `python -m membrane_solver.build_ext`) and always fall back to NumPy when unavailable.
- Bending energy now defaults to the Helfrich model (`bending_energy_model="helfrich"`) with zero spontaneous curvature unless overridden.
- `gaussian_curvature` can now enforce strict topology validation via `gaussian_curvature_strict_topology`.
- `gaussian_curvature` now supports boundary loops by default, using Gauss–Bonnet interior+boundary terms (with optional `gauss_bonnet_exclude` facet filtering).
- Line search now evaluates Armijo acceptance on the post-constraint state to keep step acceptance consistent with enforced constraints.
- Single-constraint KKT projection is now supported when a constraint module supplies a gradient.
- Multi-constraint KKT projection now solves a small constraint system when multiple gradients are provided.
- Gradient projection now uses the KKT path exclusively; fixed vertices are zeroed separately and legacy per-entity constraint projection is removed.
- Constraint projection now relies solely on KKT-style gradients; legacy apply-constraint gradient paths have been removed.
- Added a BFGS stepper (`bfgs`/`hessian` command) for quasi-Newton-style steps on moderate-sized problems.
- Added cached triangle-row indices and position-array reuse to reduce repeated mesh-to-array conversions during energy evaluation.
- Interactive: `tilt_stats` now supports `in`/`out` leaflets (`tstat` alias) and no longer collides with the `tX` step-size shortcut.
- Interactive: `set vertex ... x|y|z ...` updates vertex coordinates (not just options), enabling deterministic symmetry breaking.
- Minimization defaults now use Gradient Descent in the CLI; Conjugate Gradient remains available via `cg`.
- Line search acceptance is strict Armijo (no constraint-only acceptance path), improving stability at the cost of being more conservative.
- Body-area constraint gradients now project onto the constraint manifold, and hard constraints are enforced once before minimization starts to align with target values.
- Volume handling:
  - `lagrange` mode auto-loads the `volume` constraint and uses gradient projection.
  - `penalty` mode auto-loads the `volume` energy module instead, restoring quadratic penalty behaviour.
- `main.py` now imports `visualize_geometry` lazily so headless runs/benchmarks don’t crash when Matplotlib can’t create cache dirs.
- `benchmark_cube_good.py` runs directly on `cube_good_min_routine.json` without writing outputs.
- Interactive mode: `rN` now repeat-refines without scripting loops, `i` is a shorthand for the properties command, and README/manual document the updated syntax. The old README TODO list was moved to `docs/ROADMAP.md`.
- Visualization: Facet rendering skips <3-edge facets so line-only meshes can be displayed cleanly. Enforced equal aspect ratio in 3D plots.

### Fixed
- **Bending Energy & Gradients**:
  - Corrected the mean curvature calculation to use standard **Voronoi Areas** instead of redistributed "Effective Areas", resolving physical inaccuracy near boundaries (e.g., spherical cap energy fixed).
  - Fixed a missing factor of **2** in the analytic gradient of the Helfrich energy density.
  - Harmonized the energy and gradient functions to use a "Mixed Area" formulation (Curvature from Voronoi, Weight from Effective), reducing the relative error between Analytic and Finite Difference gradients on open meshes from ~50% to ~1.7e-10.
- **Mesh Validation**: Fixed `validate_edge_indices` in `geometry/entities.py`. It previously assumed contiguous 1-based indices, which caused `equiangulate` (which creates sparse indices) to fail validation and revert topology changes. This resolves "cone-like" artifacts in spherical cap simulations.
- **Visualization**: Fixed transparency bug where alpha channel was ignored.
- Fixed `square_to_circle.json`: Added missing instruction list and updated it to interleave minimization with mesh maintenance (`r`, `g10`, `u`, `V`) to prevent mesh tangling/overlap during large deformations.
- `load_data` accepts `Path` objects (needed for tests writing tmp JSONs).
- `cube_good_min_routine` converges again (energy ≈ 4.85) when run under penalty mode.
- `save_geometry` now reindexes sparse IDs so save→load roundtrips remain valid after refinement/equiangulation.
- `detect_vertex_edge_collisions` now builds the vertex cache when missing.
- Parsing treats `"fixed"` inside an entity `constraints` list as a fixed flag (not a constraint module name).
- Fixed a sign-convention mismatch in the analytic bending gradient (discrete curvature uses `K = -L X`), which caused large deviations from finite differences.

### Maintenance
- **Consolidated Analysis**: Merged `analyze_hemisphere.py` logic into `benchmarks/benchmark_cap.py`.
- **Removed**: `visualize_geometry.py` (redundant wrapper) and `analyze_hemisphere.py`.
- Fixed F821 lint errors for undefined names in steppers and tests.
- Added Ruff + pre-commit configuration for local linting.
- Moved logging setup helper to `runtime/logging_config.py` and updated imports.

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
