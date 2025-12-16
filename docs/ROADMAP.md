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

5. Pure Gaussian curvature
   - Check invariance under topology‑preserving deformations: no net change in
     energy for a closed surface with fixed topology.

6. Tilt source decay
   - Introduce localized tilt sources that decay away from the source region;
     should form a “dimple” or invagination.

7. Dimpled sphere with one embedded caveolin disk
   - First 3D generalization of the 1D caveolin model (see
     `docs/caveolin_generate_curvature.pdf`).

8. Box that minimizes into a sphere with a dent
   - Use fixed / `no_refine` regions to pin parts of the surface while the
     rest relaxes.

9. Plane with an inner disk and outer perimeter
   - Test mixed boundary conditions and perimeter constraints.

## 3. Mean curvature examples

10. After implementing mean‑curvature energy
    - Membrane between two fixed parallel circles forming a catenoid.
    - 10.1 Compare against Surface Evolver `cat.fe`, including how it treats
          fixed surface area of the soap film.

11. Flat sheet that folds to its spontaneous curvature
    - Benchmark for bending energy and spontaneous curvature terms.

## 4. Caveolin and complex inclusions

12. Single caveolin with outer membrane decay
    - Full 3D generalization of the caveolin model with a local curvature /
      tilt source and far‑field membrane.

13. Automatic minimization workflow
    - User defines target refinement or mesh quality criteria; the program
      iterates between minimization, refinement, equiangulation and averaging
      to reach a prescribed resolution and energy tolerance.
