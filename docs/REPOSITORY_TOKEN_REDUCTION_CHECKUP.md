# Repository Token-Reduction Checkup

Status: **all six phases complete; reopen only from measured friction**
Date: 2026-08-30

## Objective

Reduce repository context and rediscovery cost while preserving solver behavior
and scientific evidence. Count production code, tests, and governance together;
prefer cohesive, bounded ownership over decomposition by file size alone.

## Closeout baseline

- 556 tracked Python files and 126,920 Python lines.
- 43 Markdown files and 6,432 Markdown lines.
- Largest areas: tests 46,303 lines; tools 38,194; modules 19,389; runtime
  10,314; geometry 5,359; commands 1,634.
- Since the Phase 4 branch point (`caca279`), Python is net 5,056 lines smaller
  (5,991 additions and 11,047 deletions).
- Across the Phase 4R closeout range (`98cd1a1..2b7e284`), Python is net 29
  lines smaller and Markdown is net 121 lines smaller.

## Phase reconciliation

| Phase | Status | Ongoing rule |
|---|---|---|
| 1 — repository plan | Complete through `AGENTS.md`, repository routing, guardrails, and measured baselines. | Do not restore the old long plan. |
| 2 — subsystem plan | Complete and intentionally compact. | Add routing only after repeated search cost is observed. |
| 3 — per-change manifests and low-risk extraction | Complete. Evaluation, projection, IO, topology, minimizer-helper, and diagnostic boundaries are present. | Keep change manifests transient. |
| 4 — subsystem convergence | Complete after measured Phase 4R re-evaluation. | Preserve the accepted ownership boundaries; do not decompose large files solely because they are large. |
| 5 — integration and scientific separation | Complete. Fast, PR, background, exhaustive, and theory lanes are separated. | Reopen only under `SLOW_TEST_DEDUP_PLAN.md` criteria. |
| 6 — retirement and ownership cleanup | Complete. Obsolete tools, wrappers, prototypes, and duplicate disk/rim-source implementations were removed or consolidated. | Preserve retained scientific lanes. |

## Phase 4R closeout

Accepted boundaries:

- polygon triangulation owns geometry policy; refinement retains a compatibility
  import;
- refinement option policy is separate from topology mutation;
- thetaB Markdown serialization and report-wide analysis share one report owner;
- flat-KH numerical metrics have one owner separate from audit orchestration;
- shared flat-disk benchmark measurements no longer import through the large
  reproduction driver.

The final three-slice batch added 38 Python lines for module boundaries while
removing 1,555 lines from the three affected drivers: flat-KH audit (896),
flat-disk reproduction (446), and thetaB cadence audit (213). Tests and
Markdown were unchanged in that batch. Existing compatibility imports preserve
callers without duplicating implementations.

Rejected splits remain rejected: the callback-heavy minimizer fallback owner,
the helper-heavy tilt policy split, and thetaB configuration extraction. Each
would add coupling or a wide import surface without enough working-context gain.

## Reopen criteria

Reopen locality work only when at least one condition is measured:

1. repeated tasks must inspect unrelated owner files to make one bounded change;
2. duplicate implementations reappear across retained tools or runtime modules;
3. a behavior bug exposes an ownership or test-boundary gap;
4. sampled agent work shows context or rediscovery cost has materially worsened.

Do not resume repository-wide test cleanup or continue extracting helpers merely
to reduce the size of a retained driver.
