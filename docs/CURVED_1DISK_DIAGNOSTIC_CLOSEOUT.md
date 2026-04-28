# Curved 1-Disk Free-Membrane Diagnostic Closeout

## Summary

The current curved single-caveolin free-membrane lane still misses the
`docs/1_disk_3d.tex` tensionless target.  The miss is now localized enough to
justify a separate behavior-changing fix stream rather than more broad
diagnostics.

Current diagnostic-only benchmark signature after removing the runtime module
edits from this PR:

- selected `thetaB = 0.12` vs TeX target `0.1845693593`
- total energy `+0.4362312696` vs TeX target `-1.1597607985`
- selected-theta outer elastic is about `16.02x` the TeX selected-theta value
- contact work is consistent with the selected thetaB, so contact is not the
  limiting term
- the outer height remains flat through the log-fit window; only the first
  active support shell has nonzero height
- the shell-2 target direction audit reports a nearly inward direction
  (`r_dir cos global radial ~= -0.9957`)

The earlier mixed staged state reported selected `thetaB = 0.06` and about
`34.77x` selected-theta outer elastic.  Those numbers depend on runtime module
edits that are intentionally excluded from this diagnostic-only closeout.

## Diagnostic Calls

The aggregate diagnosis ranks the candidate causes as:

1. **Curvature generation does not propagate.**  The first active support shell
   moves, but free outer shells remain flat in the logarithmic profile window.
2. **Excess shared-rim/local-shell elastic cost.**  The realized outer elastic
   term dominates the selected-theta energy mismatch.
3. **Wrong rim/shell target direction or shell-2 continuation.**  Dedicated
   audits show that the propagated shell-2 target direction can point inward
   even when the secant scalar and phi target are positive.

## Validation Snapshot

Commands that passed:

```bash
python -m tools.diagnostics.curved_1disk_miss_diagnosis --skip-target-audits --skip-control-volume
python -m tools.diagnostics.curved_1disk_shared_rim_phi_target_audit
python -m tools.diagnostics.curved_1disk_shell2_tiltout_audit
pytest -q tests/test_curved_1disk_miss_diagnosis.py
pytest -q -m benchmark tests/test_curved_1disk_theory_benchmark.py tests/test_curved_1disk_shared_rim_phi_target_audit.py tests/test_curved_1disk_shell2_tiltout_audit.py
```

Known current-state failure:

```bash
pytest -q -m "e2e or regression" tests/test_kozlov_free_disk_thetaB_convergence_e2e.py
```

Result: `12 failed, 12 passed`.  These failures are expected evidence for the
next fix stream, not a regression introduced by the aggregate closeout report.

## Next Stream

The next PR should be behavior-changing and must follow
`docs/FEATURE_CONTRACT_CURVED_1DISK_SHARED_RIM_FIX.md`.

Do not solve this miss by adding calibration factors, hidden weights, or
theta-dependent tuning.  The first fix should target shell/target construction
and shape propagation semantics for the curved free-disk lane.
