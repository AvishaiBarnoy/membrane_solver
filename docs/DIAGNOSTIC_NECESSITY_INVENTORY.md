# Diagnostic Necessity and Retirement Inventory

Status: **active — tool audit complete**
Owner: `tools/` and `tools/diagnostics/`

## Decision rule

Retain a script until its consumers, scientific lane, artifact schema, and
overlap are reviewed. Zero Python imports alone is not deletion evidence because
supported CLIs may be invoked externally. Consolidation must reduce the complete
production-and-test cluster.

## Retention summary

| Scope | Decision | Evidence |
|---|---|---|
| Theory-parity drivers and audits | retain | dedicated acceptance coverage and distinct artifacts |
| Flat-disk/KH and curved 1-disk diagnostics | retain | distinct fixtures, regimes, or regression consumers |
| Scaffold and theta-B diagnostics | retain | dedicated CLI/schema consumers |
| Benchmark/profile entry points | retain | documented workflows or focused performance coverage |
| Imported helpers | retain only with multiple consumers or a stable schema boundary | direct caller scan |
| Single-consumer helpers | inline when the parent is next changed and total lines decrease | compaction rule |

## Completed audit decisions

- `tools/check_performance.py` was removed in `ddc9994`. It had no callers or
  documented command and read a stale benchmark path; `tools/suite.py` owns the
  maintained workflow.
- `tools/inspect_kernel.py` was removed in `ddc9994`. It was an ad-hoc docstring
  probe with no output contract or regression coverage.
- `tools/render_macro_snapshot.py` remains: it has no repository callers but is
  a coherent standalone visualization CLI.
- `tools/analyze_folding.py` remains outside the audit because it is untracked
  user-owned work.
- `flat_disk_kh_runtime_probe.py` remains a standalone repeated-run performance
  CLI; its small input-contract test is reviewed separately.

## Rejected consolidation

A broad diagnostic-helper extraction was rejected during Phase 5. Although its
focused tests passed, the complete cluster added 4,132 lines and removed 1,183
(net +2,949). The original diagnostic implementations were restored and the new
helper/test scaffolding was removed. No workflow, fixture, lane, or report
schema was retired.

## Required evidence for future removal

1. Direct and documented external consumers.
2. Fixtures, lane/protocol strings, CLI behavior, and report keys.
3. Unique scientific invariant versus duplicated setup.
4. Replacement path and focused before/after artifact comparison.
5. Net production-and-test delta and a recovery path in Git history.
