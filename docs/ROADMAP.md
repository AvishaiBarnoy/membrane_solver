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

6. Pure Gaussian curvature
   - Check invariance under topology‑preserving deformations: no net change in
     energy for a closed surface with fixed topology.
   - [ ] Document and scaffold a Gaussian curvature module (`modules/energy/gaussian_curvature.py`)
         with baseline tests asserting `NotImplementedError`.

7. Inclusion disk (geometry scaffold)
   - A tagged/fixed inclusion patch on an approximately spherical surface.
     This is a geometry + infrastructure benchmark (explicitly not caveolin physics yet).
   - [x] Benchmark scaffold: `benchmarks/benchmark_two_disks_sphere.py` using
     `meshes/bench_two_disks_sphere.json` (currently modeled as two small disks).
   - Reference (future): `docs/caveolin_generate_curvature.pdf`.

8. Dented sphere with flat circular patch (from cube)
   - Goal: one face stays planar and its boundary relaxes toward a circle,
     while the rest of the surface relaxes toward a (nearly) spherical shape.
   - Start with one planar face, then repeat with two planar faces.
   - Stretch goal: change target volume mid‑simulation and verify the shape
     inflates/deflates correctly under volume constraints.
   - [x] Benchmark scaffold: `benchmarks/benchmark_dented_cube.py` using
     `meshes/bench_dented_cube.json` (currently a coarse placeholder; the
     “planar + circular rim” behavior needs a per‑vertex/region plane constraint).

9. Tilt source decay
   - Introduce localized tilt sources that decay away from the source region;
     should form a “dimple” or invagination.
   - [ ] Placeholder module for tilt energy with CLI toggles so users can enable
         the hook even before the math lands.

10. Plane with an inner disk and outer perimeter
   - [ ] Test mixed boundary conditions and perimeter constraints.

## 3. Mean curvature examples

11. After implementing mean‑curvature energy (end‑to‑end benchmarks)
   - [ ] Sheet relaxes under bending toward a minimal‑mean‑curvature surface
         (with appropriate area/volume constraints to avoid trivial collapse).
   - [ ] Cube relaxes toward a sphere primarily under bending (with fixed area
         and/or volume constraint; no surface tension term).
   - [ ] Membrane between two fixed parallel circles: catenoid has mean curvature
         ~0, so it should be a near‑zero‑bending‑energy reference solution.
   - [ ] Compare against Surface Evolver `cat.fe`, including how Evolver treats
         fixed surface area of the soap film.

12. Flat sheet that folds to its spontaneous curvature
   - Benchmark for bending energy and spontaneous curvature terms.

## 4. Caveolin and complex inclusions

13. Single caveolin with outer membrane decay
    - Full 3D generalization of the caveolin model with a local curvature /
      tilt source and far‑field membrane.

14. Automatic minimization workflow
    - User defines target refinement or mesh quality criteria; the program
      iterates between minimization, refinement, equiangulation and averaging
      to reach a prescribed resolution and energy tolerance.

## 5. Engineering & Infrastructure

15. CLI & Usability
    - [x] Query/Adjustment commands (`print`, `set`).
    - [x] Live Visualization (`lv`) via Matplotlib interactive mode.
    - [x] Surface radius of gyration reporting in CLI properties output.
    - [ ] Add `history` command to replay interactive session.

16. Performance & Architecture
    - [x] **Hybrid SoA Architecture**: Implement "Scatter-Gather" pattern where optimization runs on dense arrays (Structure of Arrays) while topology remains object-oriented.
    - [ ] **Compiled Extensions**: Port the hot-loop `compute_energy_and_gradient` to Fortran (f2py) or Rust (PyO3) for ~100x speedup.
    - [ ] **Parallelism**: Explore OpenMP for energy summation.
    - [ ] Re-evaluate Conjugate Gradient defaults once stability wins are locked in; keep GD as the conservative baseline.

17. Code Quality
    - [ ] Refactor `fixed` constraints to be core entity properties (removing them from the constraint module list).
    - [x] Consolidate constraint gradient handling into the KKT projection path.
    - [x] Expression-based energies/constraints.
        - [x] Add expression `defines` to share symbolic constants across input files.
    - [ ] Add regression testing for performance (CI integration with thresholds).
    - [x] Add targeted unit tests for core helpers (volume penalties, edge-index exceptions).
    - [ ] Continue raising coverage around energy/constraint managers, CLI commands, and visualization glue.

18. Documentation
    - [x] README + manual now describe testing/diagnostics, and reference the Mermaid architecture diagram.
    - [ ] Expand API docs for future curvature/tilt modules once the placeholders evolve into full features.
