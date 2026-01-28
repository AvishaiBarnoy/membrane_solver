# Roadmap

This file tracks medium‑ and long‑term goals for the membrane solver. It is
intended for development and planning; users should consult `README.md` and
`manual.md` for day‑to‑day usage.

## 1. Baseline shape problems

1. Cube that minimizes into a sphere
   - [x] 1.1 Soft constraint – volume energy (`volume_constraint_mode="penalty"`)
   - [x] 1.2 Hard constraint – Lagrange multiplier via volume‑gradient projection
        (`volume_constraint_mode="lagrange"`, default)

2. Square that minimizes into a circle
   - [x] DONE: Using `line_tension` energy (minimizes perimeter) + `body_area`
         constraint (conserves area). Validated in `tests/test_line_tension.py`.
   - [x] 2.1 Soft constraint – surface‑area energy (`modules/energy/body_area_penalty.py`)
   - [x] 2.2 Hard constraint – Lagrange multiplier for fixed area (`modules/constraints/body_area.py`)

3. Capillary bridge (catenoid)
   - [x] Two circles at fixed distance; tests surface tension and surface constraint.
   - [x] Validated in `benchmarks/benchmark_catenoid.py` using `pin_to_circle`.

4. Pinned spherical cap under tension
   - [x] Yield spherical caps using boundary constraints and pressure balance.
   - [x] Validated in `benchmarks/benchmark_cap.py`.

## 2. Curvature‑driven phenomena

5. Mean curvature (bending / Helfrich) implementation
   - [x] Standardize the discrete mean‑curvature definition: Uses **Cotangent Laplace-Beltrami** with **Mixed Voronoi Areas** (`geometry/curvature.py`).
   - [x] Provide a bending‑energy module: `modules/energy/bending.py`.
   - [x] Add unit‑level validation: Verified against analytical sphere energy ($4\pi$) and zero-energy for flat planes. Added strict **Finite Difference** checks in `tests/test_numerical_consistency.py`.
   - [x] Add an analytic bending gradient and validate it against finite differences (`tests/test_bending_finite_difference.py`).
   - [x] Add parameter plumbing: `bending_modulus` integrated via CLI and input files.
   - [x] Default bending model switched to Helfrich (`bending_energy_model="helfrich"`) with zero spontaneous curvature unless overridden.

6. Pure Gaussian curvature
   - Check invariance under topology‑preserving deformations: no net change in
     energy for a closed surface with fixed topology.
   - [x] Implement a Gaussian curvature energy module (`modules/energy/gaussian_curvature.py`)
         for closed surfaces with constant `gaussian_modulus` (Gauss–Bonnet topological constant).
   - [x] Add Gauss-Bonnet drift diagnostics for open surfaces with boundary loops
         (`runtime/diagnostics/gauss_bonnet.py`) and regression tests.

7. Inclusion disk (geometry scaffold)
   - A tagged/fixed inclusion patch on an approximately spherical surface.
     This is a geometry + infrastructure benchmark (explicitly not caveolin physics yet).
   - [x] Benchmark scaffold: `benchmarks/benchmark_two_disks_sphere.py` using
     `meshes/bench_two_disks_sphere.json` (currently modeled as two small disks).
   - [x] `pin_to_circle` supports a fit mode for movable circular rims when needed.
   - Reference (future): `docs/caveolin_generate_curvature.pdf`.

8. Dented sphere with flat circular patch (from cube)
   - [x] Removed: We assume rigid circular inclusion footprints (caveolin disks),
     so boundary circularity is enforced directly (e.g. `pin_to_circle`) rather
     than emerging from line tension + area constraints.
   - The existing scaffold (`benchmarks/benchmark_dented_cube.py` +
     `meshes/bench_dented_cube.json`) remains useful as a geometry harness.

9. Tilt source decay
   - Introduce localized tilt sources that decay away from the source region;
     should form a “dimple” or invagination.
   - [ ] Placeholder module for tilt energy with CLI toggles so users can enable
     the hook even before the math lands.
   - [ ] KH-pure variants: curl-free vs curl-rich tilt fields on flat patches.
   - [x] KH-pure refinement stability regression tests (`tests/test_kh_pure_benchmarks.py`).
   - [x] Benchmark runner for tilt mesh suite (summaries + plots + smoke test).
   - [x] Tilt-source decay benchmark meshes (`meshes/tilt_benchmarks/tilt_source_rect.yaml`,
         `meshes/tilt_benchmarks/tilt_source_annulus.yaml`).
   - [ ] **Bilayer tilt fields (`tilt_in` / `tilt_out`)**: extend the 2D leaflet model
         in `docs/s41467-025-64084-9.pdf` to a 3D surface with per-leaflet tilt vectors.
     - [x] Data model: extend `Vertex` and `Mesh` with `tilt_in`, `tilt_out`,
           `tilt_fixed_in`, `tilt_fixed_out`, plus SoA views + cache invalidation.
     - [x] I/O: parse/save `tilt_in`/`tilt_out` in YAML/JSON; keep backwards-compatible
           `tilt` as a single-field alias (or require explicit selection per benchmark).
     - [x] Refinement: midpoint inheritance per leaflet:
           - midpoint `tilt_*` is always the average of parent tilts
           - midpoint `tilt_fixed_*` is `True` iff both parents are `tilt_fixed_*`
     - [x] Operators/metrics: expose `|t_in|`, `|t_out|`, `div(t_in)`, `div(t_out)`
           and (optionally) `t_out - t_in` for diagnostics/plots.
    - [x] Energies (first pass: fixed geometry):
          - per-leaflet `tilt` magnitude term (`tilt_modulus_in/out`)
          - per-leaflet `tilt_smoothness` term using `bending_modulus` (or `bending_modulus_in/out`)
          - optional inter-leaflet coupling (e.g. penalize `|t_out - t_in|^2` or `|t_out + t_in|^2`)
    - [x] Energies (second pass: shape coupling):
          - add `bending_tilt_in` / `bending_tilt_out` for leaflet-specific coupling
            (document sign conventions / leaflet orientation).
          - add bilayer-thickness parameters if needed for offset-surface corrections.
    - [x] Minimizer: relax both tilt fields (nested/coupled) with independent fixed masks,
          plus a combined mode for coupled leaflet relaxation.
     - [x] Visualization: add `lv tilt_in`, `lv tilt_out`, `lv div_in`, `lv div_out`,
           plus a `--color-by` mode in `tools/tilt_benchmark_runner.py`.
     - [ ] Unit tests:
           - YAML/JSON round-trip preserves both tilt fields + fixed flags
           - `project_tilts_to_tangent` keeps both fields tangent
           - refinement inheritance tests for each leaflet + mixed fixed/non-fixed parents
     - [ ] Regression tests:
           - finite-difference / directional-derivative checks for new energy gradients
           - refinement convergence (energy decreases under refinement for fixed-geometry decay benchmarks)
           - vectorization guardrails for hot-loop energy assembly (no per-vertex Python loops)
           - analytic identity (tensionless): distal/proximal tilts match for the 1-disk outer membrane using the θ_B bilayer rim source (`tests/test_kozlov_1disk_3d_analytic_regression.py`)
           - [x] hard rim-slope matching constraint (per-vertex pairing; tilt+shape projection)
           - [x] small-drive 1-disk tensionless regression (κ=1, k_t≈135 for 1 unit=15nm)
     - [ ] E2E benchmarks (expected behavior):
           - **Independent leaflets** (no inter-leaflet coupling): a source in `tilt_in`
             decays with length scale λ≈sqrt(k_s/k_t) while `tilt_out` remains ~0 if initialized at 0.
           - **Strong inter-leaflet coupling**: `tilt_out` tracks `tilt_in` (or anti-tracks,
             depending on the chosen coupling) and both share a common decay profile.
           - **With shape coupling** (`bending_tilt`): localized leaflet sources induce
             localized curvature; flipping `tilt_in`↔`tilt_out` should flip the sign
             of the preferred curvature if the model is implemented with correct leaflet orientation.
             Single-leaflet rim sources should induce the opposite leaflet only when
             shape relaxation is enabled (`tests/test_single_leaflet_curvature_induction.py`).
           - [x] 1-disk macro smoke test for small-drive physical scaling (`tests/test_e2e_kozlov_1disk_3d_small_drive_macro.py`)

10. Plane with an inner disk and outer perimeter
   - [ ] Test mixed boundary conditions and perimeter constraints.

## 3. Mean curvature examples

11. After implementing mean‑curvature energy (end‑to‑end benchmarks)
   - [ ] Cube relaxes toward a sphere primarily under bending (with fixed area
         and/or volume constraint; no surface tension term).
   - [ ] Membrane between two fixed parallel circles: catenoid has mean curvature
         ~0, so it should be a near‑zero‑bending‑energy reference solution.
   - [ ] Compare against Surface Evolver `cat.fe`, including how Evolver treats
         fixed surface area of the soap film.

12. Flat sheet that folds to its spontaneous curvature
   - Deferred: requires tightly coupled inextensibility constraints (edge
     lengths, facet areas, and corner-angle preservation) to stay stable.

## Tech debt

1. Integrate common geometric constraints into KKT solves
   - Add KKT support for `pin_to_plane`, `pin_to_circle`, and `global_area`
     so hard constraints participate in the same projection step as energies.

## 4. Caveolin and complex inclusions

13. Caveolin / caveolae (Kozlov bilayer tilt → 3D)
   - Goal: implement the two‑monolayer physics needed to reproduce the Kozlov
     single‑caveolin “2D membrane” model, then generalize to 3D caveolae with
     caveolin disks on a sphere.
   - References: `docs/caveolin_generate_curvature.pdf`,
     `docs/SI_caveolin_generate_curvature.pdf`, `docs/s41467-025-64084-9.pdf`.

  - [x] **Milestone A: bilayer tilt physics modules**
    - [x] Per‑leaflet tilt magnitude/smoothness energies operating on
          `tilt_in` / `tilt_out` (vectorized `*_array` API; no per‑vertex loops).
      - Unit tests: closed-form single triangle energies; dict/array parity.
      - Regression: finite-difference / directional-derivative gradients.
    - [x] Inter‑leaflet coupling energy (e.g. `∫|tilt_out - tilt_in|^2 dA` or
          `∫|tilt_out + tilt_in|^2 dA`) with analytic tilt gradients.
      - Unit tests: symmetry (swap leaflets) and limiting cases (coupling→0 / →∞).
    - [x] Minimizer: relax both leaflets with independent fixed masks
          (`tilt_fixed_in/out`) and a combined solve mode.

   - [x] **Milestone B: Kozlov 2D single caveolin on an “endless” flat membrane**
     - [x] Benchmark geometry: planar annulus (inner rim = caveolin footprint,
           outer rim = far field). Outer boundary constrained to remain flat
           and to remove rigid motions; inner rim carries the caveolin source.
     - [x] Caveolin source model (from Kozlov PDF): encode the source as
           leaflet‑specific boundary conditions and/or local parameters on the
           inclusion patch (document sign conventions).
           - Hard source: clamp rim tilt via `tilt_fixed_in` / `tilt_in`.
           - Soft source: drive rim tilt via `tilt_rim_source_in/out` line-energy term.
             Optionally parameterize the line strength via the Kozlov/Barnoy contact mapping
             `tilt_rim_source_contact_*` (Δε, a, h → γ).
     - [x] Expected behavior (E2E):
           - tilt decays away from the inclusion with λ≈sqrt(k_s/k_t) in the
             small-slope / flat‑geometry limit
           - with strong inter‑leaflet coupling, `tilt_in` and `tilt_out` track
             (or anti‑track) according to the coupling definition
           - invariance: rotating the inclusion on a flat far field should not
             change energies (up to discretization error)
     - [x] Tests:
           - [x] E2E: relax tilts (and then shape if enabled) and assert monotone
             decay + far‑field flatness tolerances.
           - [x] Regression: refinement convergence (energy decreases with refinement
             and approaches a stable limit for fixed parameters).
           - (Optional) compare radial profiles against reference curves/values
             digitized from the Kozlov PDFs once parameter sets are pinned.
     - [x] Implementation notes:
           - Milestone B is flat-geometry: `fixed_plane` + `pin_to_circle` keep rims
             circular and the far field flat; tests relax tilts without advancing
             shape steps.
           - A decay-length diagnostic benchmark is available via
             `benchmarks/benchmark_kozlov_annulus_decay_length.py`.

   - [x] **Milestone C: 3D single caveolin on a flat far field**
     - [x] Enable full shape coupling (bilayer tilt ↔ curvature) and validate
           that the far field remains approximately flat while a localized
           invagination forms near the inclusion.
     - [x] Use the soft rim/source driving terms (`tilt_rim_source_in/out`) with
           shape coupling enabled so the boundary tilt is a free variable minimized
           together with shape (and `tilt_in/out`).
     - [x] Sign test: swapping `tilt_rim_source_in`↔`tilt_rim_source_out` flips
           the preferred curvature direction (up/down invagination).

   - [ ] **Milestone E: 1_disk_3d rim matching (disk + outer membrane)**
     - [x] Enable rim-source selection on *internal* rims
           (`tilt_rim_source_edge_mode: all`).
     - [x] Add a rim-matching energy/constraint that enforces the small-slope
           continuity condition at the disk boundary (e.g. proximal tilt equals
           outer slope `φ*` at `r=R`).
    - [x] Disk+outer benchmark mesh with internal rim drive (`θ_B`) and
          explicit rim matching; far-field tilt clamped to zero.
    - [x] Single-leaflet rim-source variant with shape relaxation to induce the
          opposite leaflet only via curvature (`meshes/caveolin/kozlov_1disk_3d_tensionless_single_leaflet_source.yaml`).
    - [x] Diagnostics + regression for single-leaflet 1-disk behavior
          (`tools/diagnose_1disk_3d_single_leaflet.py`,
          `tests/test_kozlov_1disk_3d_single_leaflet_behavior.py`).
    - [x] Disk-profile target modules and regressions for single-leaflet and bilayer
          boundary forcing (`modules/energy/tilt_disk_target_in.py`,
          `modules/energy/tilt_disk_target_out.py`,
          `tests/test_kozlov_1disk_3d_single_leaflet_profile.py`,
          `tests/test_kozlov_1disk_3d_bilayer_profile.py`).
    - [ ] E2E regression: for γ=0, recover the 1_disk_3d predictions
          (`θ^p(r)=θ^d(r)` in the outer region and `φ*≈θ_B/2`).
     - [ ] Future: enforce/validate that interacting tilt sources on a given
           membrane are defined on the same leaflet side (distal/top = `tilt_in`)
           to avoid mixed-side source interference.

   - [ ] **Milestone D: caveolin disks on a sphere (caveolae)**
     - [ ] Use `meshes/bench_two_disks_sphere.json` as scaffold; add a
           single‑disk variant and then multi‑disk configurations.
     - [ ] E2E: sphere remains stable under refinement; inclusion produces a
           localized deformation without introducing self‑intersections.

14. Multi-disk positional constraints
    - Add positional constraints between circular rims (specified by chord
      length or angular separation).
    - Initial test: two disks with a fixed chord length/angle.
    - Extend to multiple disks on a sphere, controlling pairwise angles.

15. Automatic minimization workflow
    - User defines target refinement or mesh quality criteria; the program
      iterates between minimization, refinement, equiangulation and averaging
      to reach a prescribed resolution and energy tolerance.

## 5. Engineering & Infrastructure

16. CLI & Usability
    - [x] Query/Adjustment commands (`print`, `set`).
    - [x] Live Visualization (`lv`) via Matplotlib interactive mode.
    - [x] Surface radius of gyration reporting in CLI properties output.
    - [x] Add `history` command to replay interactive session.

17. Performance & Architecture
    - [x] **Hybrid SoA Architecture**: Implement "Scatter-Gather" pattern where optimization runs on dense arrays (Structure of Arrays) while topology remains object-oriented.
    - [x] **Geometry Freeze Caching**: Reuse curvature/area intermediates during tilt relaxation when positions are fixed.
    - [ ] **Compiled Extensions**: Port the hot-loop `compute_energy_and_gradient` to Fortran (f2py) or Rust (PyO3) for ~100x speedup.
    - [ ] **Parallelism**: Explore OpenMP for energy summation.
    - [ ] Re-evaluate Conjugate Gradient defaults once stability wins are locked in; keep GD as the conservative baseline.

18. Code Quality
    - [ ] Refactor `fixed` constraints to be core entity properties (removing them from the constraint module list).
    - [x] Consolidate constraint gradient handling into the KKT projection path.
    - [x] Expression-based energies/constraints.
        - [x] Add expression `defines` to share symbolic constants across input files.
    - [ ] Add regression testing for performance (CI integration with thresholds).
    - [x] Add targeted unit tests for core helpers (volume penalties, edge-index exceptions).
    - [ ] Continue raising coverage around energy/constraint managers, CLI commands, and visualization glue.

19. Documentation
    - [x] README + manual now describe testing/diagnostics, and reference the Mermaid architecture diagram.
    - [ ] Expand API docs for future curvature/tilt modules once the placeholders evolve into full features.
