# Tilt Benchmarks for the Kozlov–Hamm (KH) Tilt Model

This document defines a benchmark suite for validating an implementation of **lipid tilt** on a triangulated membrane using the **Kozlov–Hamm** framework.

The benchmarks are designed to:
- isolate individual ingredients (tilt cost, tilt splay/smoothness, KH curvature–tilt coupling),
- provide closed-form or scaling expectations where possible,
- detect discretization / BC / sign / tangency errors,
- remain meaningful on evolving 3D meshes.

## Performance Guardrail Protocol

For performance-related PRs touching tilt paths, run the reproducible hotspot
harness and attach the JSON report:

`python tools/tilt_perf_guardrails.py --pin-threads --warmups 1 --runs 5 --output-json benchmarks/outputs/tilt_perf_baseline.json`

For candidate changes, compare against the saved baseline:

`python tools/tilt_perf_guardrails.py --pin-threads --warmups 1 --runs 5 --baseline-json benchmarks/outputs/tilt_perf_baseline.json --output-json benchmarks/outputs/tilt_perf_candidate.json`

Keep the same case list, warmups, and runs for before/after comparisons.

---

## 1) KH Core Identities (do not violate)

### 1.1 Curvature is not an independent Helfrich mode
In KH, mean curvature does **not** appear as a standalone penalty `(J − J0)^2`.
Curvature enters only via a coupled combination with tilt splay.

### 1.2 The sign is fixed: effective curvature uses a minus sign
Kozlov–Hamm define an **effective curvature tensor** and its trace (effective mean curvature) as:

- `b̃ = b − t'`
- `J̃ = J − ∇_s·t`

The minus sign is not a convention; it is fixed by the kinematics (tilt gradients and bending contribute with opposite sign to the same lateral strain).

### 1.3 Minimal KH monolayer energy density (schematic but faithful)
A minimal KH-like monolayer elastic density is:

`f = (k_t/2)|t|^2 + (k_b/2)(J − ∇_s·t)^2`

Notes:
- `t` is a tangent in-plane vector field (tilt).
- `∇_s·t` is the surface divergence (director splay).
- `J` is mean curvature (your code’s definition must be consistent across all benchmarks).
- There is no free coupling coefficient `α` (i.e. do not use `(J + α ∇·t)^2`).

---

## 2) Practical Conventions for Implementation

### 2.1 Recommended field representation
Store a **3D vector per vertex** `t_i ∈ R^3` and enforce tangency:

- `t_i ← t_i − (t_i·n_i) n_i`

Also project the **tilt gradient** used by your optimizer:

- `g_i^t ← g_i^t − (g_i^t·n_i) n_i`

This avoids local-basis artifacts that appear in 2D-component representations unless you implement proper transport/connection.

### 2.2 Length scale
Whenever the surface satisfies `J ≈ 0` (flat surfaces and, more generally, **minimal surfaces**, e.g. a catenoid), the KH coupling reduces to a pure splay term and a screening length emerges:

If `J=0` then

`f → (k_t/2)|t|^2 + (k_b/2)(∇_s·t)^2`

and the characteristic screening length is

`ℓ = sqrt(k_b / k_t)`.

---

## 3) Benchmark 0 — Sanity / Null Tests

### B0.1 Tangency preservation
**Setup**
- Any curved mesh.
- Initialize random `t_i`.
- Apply tangency projection.

**Expected**
- `max_i |t_i·n_i| < ε` (near machine precision).
- Stays true after optimization steps if you also project gradients/updates.

**Detects**
- Missing projection.
- Optimizer drifting into normal components.

---

### B0.2 Operator consistency on a fixed surface
**Setup**
- Fix geometry.
- Run tilt minimization.

**Expected**
- Energy decreases monotonically (for a stable line search).
- Refinement improves smoothness and reduces spurious oscillations.

**Detects**
- Inconsistent discrete divergence / gradient.
- Instabilities in tilt solver.

---

## 4) Benchmark 1 — Flat Strip (1D): Single Source → Exponential Decay

### Purpose
Validate discrete splay/divergence and the length scale `ℓ`.

### Geometry
- Rectangle `0 ≤ x ≤ L`, `0 ≤ y ≤ W`.
- Pin geometry flat (`z=0`) and freeze vertex positions.

### BC / Source
Either:
1) **Hard Dirichlet**: enforce `t = t0 e_x` on the `x=0` vertex row.
2) **Soft anchoring band** near `x=0`:
   - `E_anchor = (k_a/2) ∫_band |t − t_pref|^2 dA`

Far edge (`x=L`): free/natural or `t=0`.

### Solved variables
- Tilt only.

### Expected behavior
- `|t(x)|` decays away from the source with `ℓ = sqrt(k_b/k_t)`.
- No vertex motion (geometry fixed).
- No transverse (`y`) patterning in the interior.

### Assertions
- Fit `|t(x)| ~ exp(−x/ℓ)` in the interior; recovered `ℓ` matches theory.
- Energy converges under mesh refinement.
- No checkerboard/oscillations in `t`.

---

## 5) Benchmark 2 — Flat Annulus (2D): Radial Rim Source → Axisymmetric Decay

### Purpose
Stress-test isotropy and boundary-loop handling.

### Geometry
- Flat annulus `R ≤ r ≤ Rout` with quasi-uniform triangulation.

### BC / Source
- Inner rim: radial source `t = t0 e_r` (Dirichlet) or anchoring band.
- Outer rim: free/natural or `t=0`.

### Solved variables
- Tilt only.

### Expected behavior
- Near-axisymmetry: tilt largely radial.
- Monotone decay with `r`.
- No azimuthal striping.

### Assertions
- Azimuthal variance of `|t|` at fixed `r` decreases with refinement.
- Profiles collapse when plotted vs `(r−R)/ℓ` across parameter sweeps.

---

## 6) Benchmark 3 — Two Sources, Same Sign (Reinforcement)

### Purpose
Validate superposition/screening with multiple sources.

### Geometry
- Flat strip or disk; geometry fixed.

### BC / Source
- Two tilt sources separated by distance `d`, both prefer `+t0`
  (two Dirichlet rows/bands or two anchoring regions).

### Expected behavior
- If `d << ℓ`: reinforcement → merged high-tilt region between sources.
- If `d >> ℓ`: two independent boundary layers.

### Metrics / Assertions
- Midpoint amplitude `A(d) = |t(mid)|/t0` increases as `d/ℓ` decreases.
- Energy vs `d` exhibits a crossover from “two independent” to “merged.”

---

## 7) Benchmark 4 — Two Sources, Opposite Sign (Cancellation / Dipole)

### Purpose
Highly diagnostic test of sign conventions and screening.

### Geometry
- Flat strip or disk; geometry fixed.

### BC / Source
- Source A prefers `+t0`, source B prefers `−t0`, separated by `d`.

### Expected behavior
- If `d << ℓ`: strong cancellation → `|t|` suppressed between sources.
- If `d >> ℓ`: two boundary layers; interior tends toward ~0.
- No overshoot / oscillations.

### Metrics / Assertions
- `A(d) = |t(d/2)|/t0 → 0` as `d/ℓ → 0`.
- Interaction energy `E_int(d) = E(d) − 2E(∞)` decays with `d/ℓ`.

---

## 8) Benchmark 5 — KH Coupled Shape Response (Curvature Slaved to Splay)

### Purpose
Validate the hallmark KH coupling: curvature appears only through mismatch with splay via `(J − ∇_s·t)^2`.

### Geometry
- Patch (strip or axisymmetric domain).
- Allow geometry to move (remove rigid-body modes: pin a few vertices / constrain COM and rotation).
- Do **not** include standalone Helfrich bending.

### Source
- Impose a boundary tilt source (Dirichlet rim or anchoring band) generating nonzero `∇_s·t`.

### Solved variables
Support both:
1) **Nested / adiabatic**: for each shape, solve tilt equilibrium; then update shape.
2) **Fully coupled**: optimize `(x, t)` simultaneously.

### Expected behavior
- Curvature localizes where splay exists.
- Away from boundaries, mismatch is small: `J ≈ ∇_s·t`.
- No systematic curvature where `∇_s·t ≈ 0`.

### Assertions
- Mismatch functional `M = ∫ (J − ∇_s·t)^2 dA` decreases with solver convergence and refinement.
- Spatial correlation: regions of large `|J|` coincide with large `|∇_s·t|`.
- Symmetry preserved (no spurious azimuthal modes in axisymmetric setups).

---

## 9) Benchmark 6 — Zero-Splay Should Not Drive Curvature

### Purpose
Regression test: divergence-free tilt patterns should not create curvature through KH coupling.

### Geometry
- Start flat; allow geometry DOFs (with rigid modes removed).

### Setup
- Enforce an approximately divergence-free tilt pattern (e.g. vortex-like on an annulus) using prescribed tilt or very stiff constraints.

### Expected behavior
- Geometry remains flat (or extremely close); no systematic `J` emerges.
- KH mismatch stays small since both `J` and `∇·t` are small.

### Assertions
- `max|J|` small and decreases with refinement.
- `∫ (J − ∇·t)^2 dA` remains small.

---

## 10) Benchmark 7 — Emergent Minimal Surface (Catenoid) + Tilt Screening

### Purpose
Use a **nontrivial surface with J≈0** to confirm that the reduction
`(J − ∇·t)^2 → (∇·t)^2` holds on minimal surfaces that emerge dynamically.

### Geometry
- Two parallel circular boundary rings.
- Allow the surface to minimize geometry (without any standalone bending term).

### Stage A (no tilt source)
- Solve for geometry alone (or with tilt unconstrained and driven to 0).

**Expected**
- Surface approaches a catenoid-like minimal surface: `J ≈ 0` away from boundaries.

### Stage B (add rim tilt source)
- Impose a tilt source on one boundary ring (Dirichlet or anchoring band).
- Solve for tilt on the (nearly) minimal surface (geometry fixed or weakly coupled).

**Expected**
- Tilt exhibits screening governed by `ℓ = sqrt(k_b/k_t)` along intrinsic surface directions.
- No curvature is driven beyond what is required by the boundary geometry; in the interior `J` remains ~0.

### Assertions
- Interior `|J|` remains small (minimal surface).
- Tilt decay length extracted on the surface matches `ℓ`.
- In regions where `J≈0`, the KH energy numerically matches the reduced form:
  `E ≈ ∫[(k_t/2)|t|^2 + (k_b/2)(∇·t)^2] dA`.

---

## 11) Recommended Execution Order

1) B0.1 Tangency preservation
2) B0.2 Operator consistency (fixed surface)
3) Benchmark 1 (flat strip decay)
4) Benchmark 2 (annulus radial decay)
5) Benchmark 3 (two sources, same sign)
6) Benchmark 4 (two sources, opposite sign)
7) Benchmark 6 (zero-splay → no curvature)
8) Benchmark 5 (fully coupled KH shape response)
9) Benchmark 7 (emergent minimal surface + screening)

---

## 12) Summary: What “Passing” Means in KH

A correct KH implementation should demonstrate:

- Tilt stays tangent under optimization.
- Screening length `ℓ = sqrt(k_b/k_t)` appears wherever `J≈0` (flat or minimal surfaces).
- Multiple sources show reinforcement/cancellation with crossover controlled by `d/ℓ`.
- Curvature is never independent: it is slaved to splay through `(J − ∇·t)^2`.
- In coupled runs, curvature appears only where splay exists, and mismatch decreases with refinement.

---

## 13) KH-Pure Variants (No Smoothness Regularizer)

These variants use **only** `bending_tilt` with `tilt_fixed` patterns on a flat
surface, omitting `tilt` and `tilt_smoothness`. They are intended to isolate
the KH coupling term.

- Curl-free field (nonzero divergence): should register **nonzero** energy.
- Curl-rich field (near-zero divergence): should register **near-zero** energy.

See `meshes/tilt_benchmarks/kh_pure_curl_free.yaml` and
`meshes/tilt_benchmarks/kh_pure_curl_rich.yaml`.
