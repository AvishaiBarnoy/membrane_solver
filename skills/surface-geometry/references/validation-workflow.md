# Validation Workflow

Use this file before changing discrete geometry operators, curvature fields, or geometry-sensitive energy gradients.

## Acceptance-First

- Add an acceptance or integration test first when the user-visible behavior changes.
- Add a focused unit/regression test for the local operator invariant as well.
- For derivative changes, add or update a finite-difference or directional-derivative test.

## Minimum Geometry Checks

- Normalization check:
  verify whether the code path expects `H`, `2H`, `J`, `|H|`, or `H n`.
- Orientation check:
  confirm whether the quantity should be signed or orientation-independent.
- Integrated vs pointwise check:
  verify whether dual areas have already been divided out.
- Degeneracy check:
  make sure new code behaves safely for near-zero triangle areas or near-singular cotangents.

## Good Regression Targets

- Sphere or near-sphere:
  mean and Gaussian curvature should be stable and approximately uniform away from defects.
- Catenoid or minimal-surface fixtures:
  interior mean curvature should stay small.
- Planar mesh:
  mean curvature and angle defects should be near zero in the interior.
- Cotan derivative checks:
  `grad_cotan` and related area derivatives should match finite differences on random non-degenerate triangles.

## Performance and Cache Checks

- If a hot geometry path changes, run the relevant benchmark or at least the exact affected test family at representative mesh sizes.
- Verify cache invalidation against `mesh._version` and facet/vertex row-version guards.
- Do not add per-entity Python loops inside hot geometry or energy evaluation paths.

## When to Stop and Re-check

- The local file's curvature normalization is unclear.
- The code mixes signed normals with magnitude-only curvature.
- The requested formula assumes a different discretization family from the repo's cotan/mixed-area approach.
- A geometry change actually implies a membrane-model change and should be handled with `membrane-physics`.
