# Theory-Parity Merge Policy

## Separate scientific contracts

The coarse analytical lane and the continuum/full-physics lane answer different
questions and must not share an acceptance baseline.

- The coarse lane may record deterministic extrapolated-shell behavior and
  directional improvement.
- The continuum lane retains its stronger free-side trace-continuation target.
  Until that target is recovered, it remains a strict known failure rather than
  being redefined or numerically rebaselined to the current output.

Changing a protected theory golden, tolerance, or acceptance inequality requires
explicit scientific review. A deterministic result alone is not evidence that
the result is theoretically correct.

## Pull-request gate

The `Theory correctness (blocking)` CI job runs every node ID listed in
`tests/manifest/theory_critical_pr.txt` with repository marker filtering
disabled. The job must not use `continue-on-error` or suppress a failing command.

Exploratory and exhaustive lanes may remain non-blocking, but they do not replace
the critical manifest. A strict expected failure in the manifest preserves a
known scientific miss and becomes blocking if it unexpectedly changes behavior.
