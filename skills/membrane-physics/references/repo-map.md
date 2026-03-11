# Repo Map

Use this file when converting KH continuum statements into this repository's implementation patterns.

## Existing Repo Anchors

- `docs/TILT_BENCHMARKS.md`: current statement of KH identities, benchmark design, and performance guardrails.
- `geometry/tilt_operators.py`: vectorized P1 surface-divergence helpers for ambient 3D vertex tilt fields.
- `geometry/entities.py`: dense SoA views such as `positions_view()`, `triangle_row_cache()`, `tilts_view()`, `tilts_in_view()`, `tilts_out_view()`, plus tangent projection helpers.
- `geometry/curvature.py`: curvature-related discrete operators and kernels.
- `modules/` and `tests/fixtures/`: existing KH-facing energy terms, parameter names, and regression fixtures.

## Representation Rules

- Use dense NumPy arrays for hot-loop energy/gradient work.
- Keep mesh topology/object metadata in the object model, but gather positions, triangles, and tilt arrays into SoA views before numerical work.
- For new tilt-dependent terms, prefer `compute_energy_and_gradient_array` style APIs and direct array accumulation.
- Project both tilt states and tilt gradients back to the tangent plane after updates.

## Practical Mapping

- Continuum `div_s(t)` maps to the discrete surface divergence of the vertex tilt field, typically via the P1 operators in `geometry/tilt_operators.py`.
- Continuum curvature quantities must use the repo's existing curvature-sign conventions. Check adjacent tests before editing formulas.
- Leaflet-specific work should use the dedicated in/out tilt arrays and corresponding parameter names rather than overloading the bilayer tilt field.

## Common Mistakes

- Adding a standalone `(J - J0)^2` term to KH monolayer code without the matching `- div_s(t)` contribution.
- Letting tilt vectors acquire normal components during optimization.
- Implementing tilt work with per-vertex Python loops in hot paths.
- Reusing a boundary/source convention from one leaflet for the other without checking the current fixture baselines.
- Writing theory comments that conflict with `docs/TILT_BENCHMARKS.md`.

## Default Review Checklist

- Is the term actually KH, reduced KH, or pure Helfrich?
- Is the minus sign between curvature and splay preserved?
- Are all tilt vectors and tilt gradients tangent?
- Does the code stay in the repo's SoA/vectorized pattern?
- Do existing fixtures/tests already encode the relevant sign convention?
