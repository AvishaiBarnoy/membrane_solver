# Tilt Relaxation Algorithmic Options

This note records the current performance conclusions for the exact nested
leaflet tilt-relaxation path, so future optimization work starts from measured
facts instead of repeating rejected micro-optimizations.

## Scope

The reference workload is the exact milestone-C nested leaflet relaxation path:

- benchmark: `python benchmarks/benchmark_tilt_relaxation.py`
- mesh: `meshes/caveolin/kozlov_annulus_milestone_c_soft_source.yaml`
- mode: nested leaflet tilt relaxation

At the time of writing, this path runs at roughly `0.021s` to `0.022s` average
wall time on the current optimized branch state.

## Main Findings

Deeper profiling shows that the remaining cost is no longer dominated by one
obvious cache miss or one obviously wasteful Python loop. The hot path is now
spread across several small, math-heavy vectorized kernels:

- `runtime/minimizer.py:_compute_energy_and_leaflet_tilt_gradients_array`
- `runtime/minimizer.py:_compute_tilt_dependent_energy_with_leaflet_tilts`
- `modules/energy/bending_tilt_leaflet.py:compute_energy_and_gradient_array_leaflet`
- `modules/energy/tilt_smoothness.py:_compute_smoothness_energy_and_gradient`
- `modules/energy/tilt_rim_source_in.py:compute_energy_and_gradient_array`
- `geometry/tilt_operators.py:compute_divergence_from_basis`

On the exact milestone-C path, direct module timings are all in the same rough
range, so there is no longer a single cheap hotspot left to squeeze:

- `bending_tilt_in`: about `3.4e-05 s`
- `bending_tilt_out`: about `3.4e-05 s`
- `tilt_smoothness_in`: about `2.9e-05 s`
- `tilt_smoothness_out`: about `3.0e-05 s`
- `tilt_rim_source_in` energy-only: about `3.0e-05 s`

This means that remaining wins are more likely to come from reducing the number
of expensive subproblem evaluations, not from small local rewrites of a single
NumPy expression.

## Micro-Optimizations Already Rejected

These were tested on the exact path and did not clear the performance bar:

- additional `bending_tilt_leaflet` helper rewrite around in-place `dE/d(div)`
  assembly: slower by about `+15.7%`
- fit-frame cache for `tilt_rim_source_in` in `pin_to_circle_mode: fit`:
  slower by about `+0.95%`
- scratch-buffer path for `geometry/tilt_operators.py`
  `p1_triangle_divergence_from_shape_gradients`: slower by about `+7.0%`

These regressions matter because the active masked triangle count on the exact
milestone-C path is small. Extra scratch management, helper indirection, and
compiled-call overhead often cost more than the saved allocations.

## Compiled-Kernel Feasibility

The repo already has an optional compiled Fortran tilt-divergence kernel.
However, for the exact hot inputs used by the milestone-C nested leaflet path,
the existing compiled path is not the right fast path:

- active masked triangle count is only about `32`
- cached-basis NumPy divergence path: about `6.8e-06 s`
- existing compiled full divergence kernel: about `3.1e-05 s`

The compiled kernel is slower here because it recomputes the basis and pays
higher call/setup overhead than the cached-basis NumPy path. A new compiled
basis-only kernel is therefore not obviously justified for this workload.

## Ranked Algorithmic Options

### 1. Adaptive Nested Leaflet Relaxation

Current nested relaxation in `runtime/minimizer.py:_relax_leaflet_tilts`
evaluates the tilt subproblem many times per outer shape step using a fixed
inner iteration budget.

Most promising variants:

- adaptive `tilt_inner_steps` instead of fixed count
- stop on relative energy decrease, not just gradient norm
- skip or thin inner relaxation on outer steps with very small geometry change
- relax leaflet tilts every `k` outer steps in low-change regimes

Expected payoff:

- high

Risk:

- medium to high, because convergence behavior changes

Why this is the leading option:

- it reduces the number of expensive module evaluations directly
- it attacks the dominant cost center in the current algorithm

### 2. Stronger Tilt Subproblem Solver

The current leaflet tilt solve still relies on repeated first-order evaluations.
A more aggressive subproblem solver may reduce the number of evaluations needed
per outer step.

Candidates:

- stronger preconditioned CG for leaflet tilts
- quasi-Newton or limited-memory updates for the tilt-only subproblem
- block solve using the dominant quadratic tilt terms
- reuse a local linearization for several inner iterations

Expected payoff:

- medium to high

Risk:

- high, because this changes solver behavior more deeply

### 3. Reduced-Model or Surrogate Tilt Inner Steps

Instead of reevaluating all tilt-dependent modules directly on every inner step,
build a temporary reduced model for a few inner iterations, then refresh from
the full energy.

Candidates:

- local quadratic surrogate for the tilt-only objective
- partially frozen curvature/divergence terms for several inner steps
- staggered exact refreshes after a few cheap surrogate steps

Expected payoff:

- medium

Risk:

- high, because surrogate fidelity has to be controlled carefully

## Recommendation

The best next experiment is **adaptive nested leaflet relaxation**.

This is the most plausible remaining path to a material speedup on the exact
milestone-C benchmark because:

- the remaining per-call math is already fairly tight
- recent local micro-optimizations have mostly regressed
- the dominant cost is now the repeated solution of the inner leaflet tilt
  subproblem itself

## What Not To Do Next

Avoid spending more time first on:

- small cache additions around `bending_tilt_leaflet`
- more scratch-array rewrites inside `geometry/tilt_operators.py`
- a new tiny compiled divergence kernel without first proving that the call
  size is large enough to amortize the overhead

Those avenues were either already tested and rejected, or are weak bets for the
current exact workload.
