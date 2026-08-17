# Tilt Solve-Mode Matrix

Status: **active — characterization contract**
Owner: `runtime/steppers/tilt_relaxation.py`, with projection helpers in
`runtime/projections/tilt.py`

This matrix records current behavior; it does not redefine solver policy.
Any extraction from the tilt-relaxation coordinator must retain the applicable
row exactly.

| Scenario | Configuration / inputs | Required outcome | Primary coverage |
|---|---|---|---|
| Fixed mode | `tilt_solve_mode=fixed` | No relaxation; statistics report `mode_fixed` | `tests/test_tilt_solve_modes.py`, `tests/test_tilt_leaflet_solve_modes.py` |
| Unknown mode | unrecognized `tilt_solve_mode` | Warn and behave as fixed; no implicit solver selection | `runtime/steppers/tilt_relaxation.py` mode guard |
| Nested single field | free `tilt` rows on fixed geometry | Energy can decrease; zero iteration limit is a no-op | `tests/test_tilt_solve_modes.py` |
| Coupled single field | `tilt_solve_mode=coupled` | Tilt field makes progress; missing coupled-step setting uses the established inner-step fallback | `tests/test_tilt_solve_modes.py`, `tests/test_tilt_solve_mode_coupled_fallback.py` |
| Leaflet fields | `tilt_in` and `tilt_out` | Each leaflet has an independent fixed mask; fixed values are restored after trial/projection | `tests/test_tilt_leaflet_solve_modes.py` |
| Leaflet absence | absent outer vertices | Outer geometry/area contribution uses the absence mask; no absent-row contribution is fabricated | `runtime/steppers/tilt_relaxation.py`, `tests/test_tilt_leaflet_pure.py` |
| Fixed masks | vertex / in / out mask callbacks | Fixed gradient rows are zeroed and fixed field values survive projection and accepted updates | `tests/test_tilt_solve_modes.py`, `tests/test_tilt_leaflet_solve_modes.py` |
| Tangency | stored or trial 3-D tilts | Projection is tangent to current normals; fixed values override projected trial values | `tests/test_tilt_tangent_projection.py`, `tests/test_tilt_validation.py` |
| Projection cadence | `per_step` or `per_pass`, positive interval | Constraint refresh and projection happen at the configured cadence; stats expose cadence and application count | `runtime/steppers/tilt_relaxation.py` |
| CG rejection | `tilt_cg_rejection_fallback=off` or `gd` | Only these modes are accepted; fallback statistics retain attempted/accepted counts and update norms | `tests/test_tilt_leaflet_solve_modes.py` |
| Energy guard | trial relaxation raises or increases guarded energy | Field state rolls back rather than leaking a rejected trial | `tests/test_tilt_relax_energy_guard.py` |
| Theta-B boundary | coupled solve with theta-B constraint | Boundary is enforced before gradient evaluation | `tests/test_tilt_thetaB_relaxation_enforces_boundary_regression.py` |
| Stage A lane | `theory_parity_lane=stage_a_emergent` | Joint projection eligibility remains restricted to the Stage A lane; rim sources remain localized across refinement | `tests/test_tilt_rim_source_bilayer_stage_a_regression.py` |

## Mutation boundary

Problem assembly, direction/trial mechanics, projection/enforcement, and
statistics must become separate units before any large extraction. Only the
commit path may write mesh tilt fields. Projection preserves fixed values after
each candidate field is tangent-projected; the minimizer retains the
post-shape-step projection cadence.

## Extraction sequence

1. Immutable problem data: masks, normals, absence/row regions, and fixed
   values.
2. Direction and trial-energy mechanics using dense arrays only.
3. Projection/enforcement cadence and CG-rejection fallback policy.
4. Field commit/rollback followed by report construction.

No slice may combine an eligibility-policy change with a numerical-mechanics
change.
