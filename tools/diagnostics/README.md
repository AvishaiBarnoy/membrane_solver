# Diagnostic routing and lifecycle

Use this index before opening a diagnostic. Most solver changes do not require the
large aggregates. Generated reports belong under `benchmarks/outputs/`.

## Shared libraries

Imported by multiple tools/tests, not standalone fixing lanes:
`utils.py`, `free_disk_profile_protocol.py`, `free_disk_profile_fits.py`,
`free_disk_energy_split.py`, `flat_disk_one_leaflet_theory.py`, and
`curved_disk_theory.py`.

## Active focused entry points

- Inventory/boundary: `physics_sweep.py`, `inclusion_boundary_audit.py`.
- Flat curved: `flat_disk_curved_3d_audit.py`,
  `flat_disk_curved_3d_bc_sweep.py`, `flat_disk_curved_3d_ablation_sweep.py`.
- Flat KH: `flat_disk_kh_term_audit.py`, `flat_disk_kh_error_source_audit.py`,
  `flat_disk_kh_outer_vertex_audit.py`, `flat_disk_parity_scoreboard.py`,
  `flat_disk_kh_runtime_probe.py`. Start with the term audit; do not recreate
  retired region, partition, or rim-fidelity wrappers.
- Curved gate: `curved_1disk_theory_benchmark.py`.

## Aggregate investigations

Open only for the full workflow they coordinate:
`thetaB_cadence_relaxation_audit.py`, `scaffold_energy_imbalance_audit.py`,
`parity_broad_diagnostic.py`, `parity_acceptance_triage.py`,
`thetaB_normalization_audit.py`, and `curved_1disk_miss_diagnosis.py`.

## Coarse curved one-disk lanes

These retain exploratory evidence with different acceptance expectations from strict
theory gates; the R+epsilon trace-ring work belongs here. Prefer the aggregate diagnosis
unless the task names the metric:

- `curved_1disk_energy_control_volume_audit.py`
- `curved_1disk_first_two_shell_{diveval,ingredient,magnitude}_audit.py`
- `curved_1disk_{forced_theta_diagnostic,outer_profile_source_audit}.py`
- `curved_1disk_{rim_inner_tilt_profile,shape_direction}_audit.py`
- `curved_1disk_shape_propagation_blocker.py`
- `curved_1disk_shared_rim_phi_target_audit.py`
- `curved_1disk_shell2_tiltout{,_source}_audit.py`
- `curved_1disk_transition_band_ownership_audit.py`
- `curved_1disk_trumpet_descent_audit.py`

Delete a diagnostic only when it has no live workflow, unique calculation, dedicated
fixture/contract, or uncovered behavior. Coarse or historical status alone is not enough.
