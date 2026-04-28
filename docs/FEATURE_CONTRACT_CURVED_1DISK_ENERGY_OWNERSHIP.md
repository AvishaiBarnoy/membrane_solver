# Feature Contract: Curved 1-Disk Energy Ownership Diagnosis/Fix

## Goal
Resolve the curved 1-disk post-target-fix energy attribution miss by proving whether the remaining split error is diagnostic accounting or a real runtime leaflet ownership/sign defect.

## Non-goals
- Do not tune energies to match `docs/1_disk_3d.tex`.
- Do not add calibration factors, hidden weights, or public config knobs.
- Do not change shape constraints, theta scan scheduling, or solver behavior in this stream.
- Do not convert known theta-convergence misses to passing unless a verified runtime defect is fixed.

## User-visible behavior
- Diagnostic reports distinguish runtime module totals from region-attributed disk/outer totals.
- Selected-theta energy attribution reconciles with `compute_energy_breakdown()` before claiming a runtime physics defect.
- If attribution is the defect, only diagnostic split/report code changes.
- If runtime ownership/sign is the defect, implementation stops at failing tests until the minimal runtime edit is explicit.

## Public interfaces to add/change
- Extend `tools.diagnostics.curved_1disk_energy_control_volume_audit` output with:
  - `legacy_numeric_energy_split`;
  - `numeric_energy_split` reconciled to runtime module totals;
  - `runtime_module_totals`;
  - `runtime_energy_reconciliation`.
- No user-facing config or runtime API changes by default.

## Data model / file format changes
- JSON report schema gains diagnostic fields only. Mesh YAML and runtime model formats are unchanged.

## Key invariants
- Runtime physics remains unchanged unless a later approved fix explicitly edits energy modules.
- Reconciled diagnostic split satisfies:
  - `inner_elastic + outer_elastic + contact == runtime total`;
  - `inner_elastic + outer_elastic == runtime elastic module total`;
  - shell-attributed outer elastic has near-zero residual against the chosen outer split.
- The report keeps known benchmark misses visible.

## Failure modes & error handling
- If runtime module totals cannot be reconciled, the diagnostic reports the residual and ranks attribution mismatch.
- If shell attribution is incomplete, the report keeps `unattributed_fraction` high and does not recommend runtime physics edits.
- Invalid numeric modes continue to raise the existing `ValueError`s.

## Test plan
- Unit/fast tests for reconciliation helper behavior and aggregate report wiring.
- Benchmark selected-theta test proving the reconciled split has near-zero outer residual.
- Existing curved 1-disk benchmark tests remain known-miss diagnostics.

## Performance considerations
- No hot-loop runtime path changes are planned for Gate A.
- Benchmark diagnostics remain slow and benchmark-marked; fast tests use direct helper calls or monkeypatching.
