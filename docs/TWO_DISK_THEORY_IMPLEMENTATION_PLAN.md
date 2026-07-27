# Two-Disc Theory Reproduction Plan

## Objective

Reproduce the membrane-mediated interaction of two embedded discs from the
discrete form of the continuum bending/tilt energy, without fitting the solver
to the desired interaction curve.

The first production observable is

\[
E_{\mathrm{int}}(d)=F_2(d)-2F_1,
\]

where the two-disc and one-disc calculations use matched material parameters,
outer-boundary conditions, local resolution, and minimization tolerances.

## Current Baseline

- The one-disc analytic reference is `docs/1_disk_3d.tex`.
- The active one-disc comparison lane matches the optimal scalar boundary tilt
  and total energy closely, but does not yet match the free-side leaflet fields
  or height profile.
- The field-linear scaffold activates the correct local half split, but its
  fixed-state stationarity residual is nonzero and its global magnitude is too
  large.
- `meshes/bench_two_disks_sphere.json` is geometry infrastructure only.
- `meshes/kozlov_two_holes.yaml` is an unvalidated coarse prototype. Its two
  disconnected rims currently share one center, group, and scalar boundary
  value.
- `modules/constraints/inclusion_components.py` now provides the first
  behavior-preserving multi-inclusion primitive: stable, cached discovery of
  disconnected tagged rims.

## Rules for This Work

1. Do not alter `legacy_coarse`, `physical_edge_default`,
   `shared_rim_staggered_v1`, or existing acceptance baselines while developing
   the new formulation.
2. Every operator change starts with a failing unit or characterization test.
3. Do not introduce fitted coefficients, shell weights selected by target
   agreement, or persistent symmetry-breaking forces.
4. Compare fields and variational residuals before comparing optimized scalar
   energies.
5. Do not promote a two-disc result until its one-disc limit passes Gate 0.

## Gate 0: One-Disc Field Parity

### Required observables

- Measured disk boundary value `theta_B`.
- One-sided `t_in(R+)`, `t_out(R+)`, and `phi(R+)`.
- Inner `I1` and outer `K1` profile errors.
- Height/slope profile errors.
- Energy breakdown and total energy.
- Directional derivative of the reduced energy along the measured boundary
  mode.
- Sensitivity to mesh refinement, outer-domain radius, and initial
  perturbation.

### Acceptance

- `theta_B`, inner elastic energy, outer elastic energy, contact work, and total
  energy each agree with the analytic reference within 5%.
- The normalized interface residuals and normalized fixed-state stationarity
  residual are below `1e-3`.
- Inner and outer radial profile relative errors are below 10% on their
  resolved comparison bands.
- Two nonzero perturbations differing by at least `1e3` converge to the same
  branch within 2%; the final solution does not retain a persistent forcing
  term.
- One additional local refinement changes the reported observables by less
  than 5%.

### Work package 0A: isolate the residual

- Extend `tools/diagnostics/scaffold_energy_imbalance_audit.py` with
  component-wise directional derivatives for:
  - disk bulk,
  - disk boundary,
  - trace shell,
  - outer support shells,
  - outer bulk.
- Add finite-difference agreement tests for every active energy module along
  the same boundary mode.
- Determine whether the residual is produced by energy assembly, the hard
  projector, or alternating shape/tilt minimization.

### Work package 0B: make the boundary value a field degree of freedom

- Keep the field-linear contact work and its exact gradient.
- Remove scalar `tilt_thetaB_value` feedback from the developmental lane.
- Represent the boundary amplitude as either:
  - ordinary boundary-node field values, or
  - one explicit reduced boundary mode whose gradient is assembled from the
    same continuum work.
- Apply interface continuity through the constraint/KKT formulation rather
  than by repeatedly overwriting a nonstationary state.

### Work package 0C: convergence

- Generate a radial family with controlled spacing relative to
  `ell = sqrt(kappa / kappa_t)`.
- Separate local-resolution error from finite-domain error.
- Freeze the first field-parity baseline only after all Gate 0 criteria pass.

## Gate 1: Multi-Inclusion Boundary Model

### Configuration contract

Introduce an additive inclusion list; do not overload one global center:

```yaml
inclusions:
  - id: left
    interface_group: disk_left
    center: [x1, y1, z1]
    normal: [0, 0, 1]
    contact_drive: 4.286
  - id: right
    interface_group: disk_right
    center: [x2, y2, z2]
    normal: [0, 0, 1]
    contact_drive: 4.286
```

Each inclusion must resolve to one edge-connected interface component. Each
component owns its local center, normal, radial frame, arc-length weights,
disk-side rows, and free-side trace/support rows.

### Required implementation

- Promote `collect_group_components` into a shared boundary descriptor builder.
- Split contact work into a sum over inclusions:

  \[
  F_{\mathrm{cont}}=-\sum_i 2\pi R_i g_i\theta_{B,i}.
  \]

- Assemble the exact field gradient independently on every component.
- Build rim matching and local-shell pairing independently per component.
- Report per-disc boundary values and residuals; never hide them behind a
  global mean.
- Retain the existing single-group path unchanged until the new path has
  equivalent one-disc tests.

### Acceptance

- A one-item `inclusions` list is numerically equivalent to the validated
  one-disc lane.
- Reversing inclusion declaration order does not change energy or fields.
- Translating or rigidly rotating a flat two-disc mesh does not change energy.
- Equal same-side discs produce equal per-disc observables by symmetry.
- Different contact drives remain independent and are reported separately.

## Gate 2: Independent Two-Centre Reference

Implement a reference solver derived directly from the linear continuum
equations, separate from the production mesh operators.

Preferred sequence:

1. Fixed flat geometry and the tilt sector only.
2. Same-sign and opposite-sign circular boundary sources.
3. Coupled shape response.
4. Finite tension.

Use a Fourier--Bessel/multipole or boundary-integral formulation for two
circles. Verify it against the analytic one-disc limit and a high-resolution
finite-domain solve.

The reference must provide fields, boundary tractions, total energy, and
`E_int(d)`, not only a fitted interaction curve.

## Gate 3: Matched Planar Separation Sweep

Use `R = 7/15` and `ell = 1/15` for the first dimensionless benchmark.

- Resolve the near-interface region with target spacing no larger than
  `ell/4`.
- Sweep edge gaps `s/ell` over approximately `0.5, 1, 2, 4, 8`.
- Set center separation to `d = 2R + s`.
- Keep the outer boundary at least several screening lengths beyond both
  inclusions and repeat with a larger domain.
- Generate a matched one-disc control using the same outer domain and local
  discretization policy.

### Acceptance

- `E_int(d)` approaches zero at large separation.
- Same-sign and opposite-sign sources satisfy the reference sign/symmetry
  behavior.
- Swapping disc labels leaves all global observables unchanged.
- Interaction curves converge under local refinement and domain expansion.
- The interaction signal is larger than the estimated subtraction error.

## Gate 4: Curved/Spherical Geometry

Only after Gates 0--3:

- Add a one-disc spherical control derived from
  `meshes/bench_two_disks_sphere.json`.
- Add chord-length/angular-separation constraints.
- Add per-inclusion tangent frames on the sphere.
- Check sphere stability, refinement convergence, and self-intersections.
- Compare locally flat, weak-curvature results with the planar interaction
  curve before studying strong curvature.

## Immediate Next Slice

1. [x] Add a read-only inclusion-boundary audit using
   `collect_group_components`.
2. [x] Make legacy rim/contact operators warn when a configured group contains
   multiple disconnected components, because their single-center result is not
   physically interpretable.
3. [x] Extend the energy-imbalance audit with per-region boundary-mode slopes.
4. [x] Add the failing Gate 0 stationarity test for the developmental scaffold.
5. [x] Use the retraction test to localize and fix the flat-reference curvature
   derivative and missing ambient P1-divergence shape derivative.
6. [ ] Correct the remaining outer bending-tilt shape pullback at the scaffold
   transition before integrating multi-disc solver behavior.

### First region-resolved finding

After the quick `g1` scaffold protocol, the fixed-state radial derivatives show:

- disk-boundary `tilt_in` slope:
  - `bending_tilt_in ≈ +1.94`
  - `tilt_in ≈ +4.34`
  - contact work `≈ -12.57`
  - net `≈ -6.29`
- trace-shell `tilt_out` net slope `≈ +35.29`
- disk-boundary `tilt_out` net slope `≈ -32.28`
- support-shell-1 `tilt_out` net slope `≈ -2.89`

The three outer-leaflet values nearly cancel as a coupled interface motion.
Therefore the large raw trace-shell derivative is not evidence for changing an
individual module coefficient. The next Gate 0 test must probe the
constraint-compatible coupled boundary mode (or its KKT-projected gradient),
then compare that mode with the reduced-energy derivative. Fixing isolated row
gradients would risk breaking a cancellation imposed by interface continuity.

### Gate 0 derivative comparison

The boundary-mode audit now compares:

- raw finite differences,
- the raw analytic gradient,
- tilt-only KKT projection,
- joint shape/tilt KKT projection,
- finite differences after hard constraint enforcement,
- finite differences after enforcement plus tilt relaxation.

At the quick `g1` state:

- raw finite difference and analytic gradient agree at `-6.286739` to numerical
  precision;
- tilt-only KKT gives `-2.183502`;
- joint KKT leaves the raw coordinate derivative at `-6.286739`;
- hard enforcement gives `+11.485305`;
- enforcement plus tilt relaxation gives `-0.829850`.

At the end of `LONG_INTERFACE_PROTOCOL`:

- raw fixed-state derivative is `-3.873821`;
- tilt-only KKT gives `-1.324158`;
- hard enforcement gives `-1.638381`;
- enforcement plus tilt relaxation gives `-0.032263`.

The original-coordinate raw derivative is internally consistent. A subsequent
test of the actual hard-enforcement retraction showed that this was not enough:
the derivative of the retracted energy differed from the full analytic
shape/tilt gradient.

`LONG_INTERFACE_PROTOCOL` ends in repeated `V` commands, which are vertex
averaging operations, not tilt-relaxation passes. It consequently does not end
on a fully minimized state. A final `g50` lowers the total energy but increases
the boundary magnitude and worsens the TeX magnitude ratios; it does not remove
the underlying global-magnitude mismatch. This means the current obstacle is
not explained solely by insufficient final minimization.

### Retraction derivative finding

Numerically differentiating the complete hard-enforcement map established
three facts:

- the retracted plus/minus states satisfy the inner and outer matching
  residuals to machine precision;
- the raw and joint-KKT gradients give the same value on the numerical
  retraction tangent, so `continuity_v2` and the KKT tangent space are
  consistent for this mode;
- the energy finite difference originally differed from that gradient by
  about `0.893`, entirely in the bending-tilt shape-gradient blocks.

The flat-reference lane set the base curvature value to zero but its analytic
shape gradient still differentiated that curvature as geometry-dependent. It
also omitted the position derivative of the ambient P1 tilt divergence.
Suppressing the inactive curvature derivative and adding the vectorized
reverse derivative of the P1 divergence reduced the full retraction mismatch
from about `0.893` to `0.0126`.

The remaining mismatch is localized to the outer bending-tilt shape pullback
at the scaffold transition. A triangle-corner effective-area hypothesis made
the contract worse and was reverted, so the exact missing term is still open.
A strict expected-failure test records that gap. Gate 0 remains closed until
the pullback is exact and the correction survives refinement and long-protocol
validation.

## Validation Matrix

| Change | Minimum validation |
|---|---|
| Component discovery | `tests/test_inclusion_components.py` |
| Boundary descriptor | component tests plus rigid-transform invariance |
| Contact work | analytic gradient and per-component additivity |
| Rim matching | one-disc equivalence plus two-disc order invariance |
| One-disc field operator | Gate 0 acceptance and exact reproducer timing |
| Two-disc planar solve | Gate 1--3 tests and refinement/domain sweeps |
| Spherical solve | prior gates plus topology and self-intersection checks |
