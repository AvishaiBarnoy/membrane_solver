# Slow Test and CI Closeout

Status: **complete as of 2026-08-29**

## Result

The default, PR, background, and exhaustive selections are separated. Shared
fixtures and helpers compact repeated setup, obsolete diagnostic wrappers are
retired, and the blocking theory manifest runs as two concurrent shards inside
one stable required check.

Measured GitHub Actions baselines:

- pull-request suite critical path: about 7 minutes 14 seconds;
- blocking theory check: 5 minutes 28 seconds (previously 8 minutes 34 seconds);
- main/background suite: 18 minutes 20 seconds.

The background critical path is intentional. It includes exploratory cases
marked `exhaustive` that PR selection omits. The regional free-disk cases test
distinct optimizer, refinement, exclusion, weighting, and attribution
interventions. The three additional flat-disk cases exercise distinct staged
relaxation, outer-tail matching, and unpinned-preset workflows. No exact
duplicate execution was found in the closeout audit.

## Suite contract

- `pytest -q`: fast default confidence.
- PR CI: representative unit, regression, E2E, and blocking theory coverage;
  excludes exhaustive cases.
- Main/background CI: includes exhaustive regression and E2E evidence.
- Scheduled workflows: broad benchmarks, profiles, and exploratory theory.

Do not shorten the background lane merely because it is slow. Change its
selection only when an exact duplicate, obsolete scientific avenue, or cheaper
equivalent assertion is demonstrated. Do not weaken tolerances or protected
theory branches as a runtime optimization.

## Reopen criteria

Reopen test/CI compaction only when one of these is true:

1. PR critical-path time materially regresses from the baseline above.
2. Collection shows the same expensive protocol invocation in multiple
   blocking PR lanes without distinct assertions.
3. A retained exhaustive case is explicitly retired by its scientific owner.
4. A shared result can remove repeated work without coupling independent lanes
   or adding more governance than it removes.
