# Theory-Parity Workflow Contract

Status: **partially requalified — coarse-lane contract approved**
Owner: `tools/reproduce_theory_parity.py`

## Protocol boundary

The CLI loads one fixture, constructs a command context, runs the named
protocol (including parity activation where applicable), collects a YAML
report, and optionally updates expansion state. Command ordering, lane choice,
fixture label, metric keys, and YAML key ordering are compatibility behavior.

## Stable inputs and artifacts

| Element | Current owner | Compatibility requirement |
|---|---|---|
| Mesh fixture | CLI `--mesh` / default parity fixture | Report fixture label and lane selection remain stable |
| Expansion policy/state | YAML policy and state paths | State decisions persist with the existing schema |
| Report | `_collect_report_from_context` | `meta`, `metrics`, reduced terms, theory ratios, and interface diagnostics retain keys/units |
| Serialization | `_save_yaml` / CLI write | YAML uses `sort_keys=False` |
| CLI exit status | `main()` | Library extraction must not alter success/failure behavior |

## Protected lanes

- default, i50, i60, near-edge, physical-edge, and scaffold-gapfill fixtures;
- theory parity targets and baseline tolerances;
- fixed-polish and expansion-stage metadata.

## Validation owners

`tests/test_reproduce_theory_parity_acceptance.py` owns baseline artifacts;
`tests/test_theory_parity_against_tex_acceptance.py` owns TeX comparisons; and
`tests/test_theory_parity_trend.py` owns downstream report consumption.

## Current gate result

The focused context/serialization/trend contracts pass. The physical-edge
baseline acceptance test executes but is a strict expected failure because
signed-curvature and shape-pullback corrections changed the energy landscape.
The coarse analytical-lane qualitative contract is approved. The numeric golden
still requires a deliberate rebaseline decision, so protocol activation, metric
collection, and state-update code remain driver-owned.
The reproducible metric comparison is recorded in
`docs/THEORY_PARITY_REQUALIFICATION_EVIDENCE.md`.
The proposed numeric-golden handoff is recorded in
`docs/THEORY_PARITY_REBASELINE_MANIFEST.md`; it does not authorize a refresh.
The current implementation passes derivative, guardrail, director/profile, and
coarse extrapolated-response checks. Continuum-scale free-side trace
continuation remains the responsibility of the separate explicit-trace lane.

## Driver ownership

The broad pure-helper extraction was rejected during Phase 5 because its full
production-and-test cluster increased repository size. Serialization, protocol
running, metric collection, activation, and persistent state I/O therefore
remain driver-owned. A future extraction requires a net reduction or a concrete
multi-consumer contract.
