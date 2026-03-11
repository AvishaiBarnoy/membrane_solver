---
name: membrane-physics
description: Use for repository tasks that implement, review, debug, benchmark, or explain membrane-physics behavior under the Helfrich model or the Kozlov-Hamm lipid-tilt extension. Trigger on requests about membrane bending energy, spontaneous curvature, lipid/director tilt fields, surface normals, tangent-plane projection, surface divergence of tilt, curvature-tilt coupling, effective curvature `J - div_s(t)`, leaflet-specific tilt terms, KH sign conventions, or KH regression/benchmark design. Do not use for generic geometry, meshing, or optimization work unless those tasks explicitly depend on these membrane-physics conventions.
---

# Membrane Physics

Use this skill to keep membrane-physics work aligned with the Kozlov-Helfrich base model and the Hamm-Kozlov tilt extension already reflected in this repository.

Prefer operational guidance over theory dumps: identify the model term being changed, load only the needed reference file, preserve the KH invariants, and route validation through the existing repo tests and benchmarks.

## Workflow

1. Identify the physics scope before editing.
   Distinguish pure Helfrich bending, KH monolayer tilt coupling, leaflet-specific tilt terms, boundary anchoring, or purely geometric operator work.
2. Load only the relevant reference file.
   Read [references/kh-core.md](references/kh-core.md) for continuum identities and sign rules.
   Read [references/repo-map.md](references/repo-map.md) for how those identities map to this codebase.
   Read [references/validation-workflow.md](references/validation-workflow.md) before changing tests, solvers, or hot loops.
3. Preserve the invariants.
   Do not invent a new sign convention, a standalone KH curvature penalty, or a non-tangent tilt representation unless the user explicitly asks for a different model.
4. Implement the smallest change that satisfies the request.
   Keep SoA hot paths vectorized and keep topology/object-model code separate from dense array computations.
5. Validate at the right level.
   Add or update acceptance/regression tests first for behavior changes.
   Run the repo's tilt benchmark/performance guardrails if a tilt hot path, solver, or operator changed.

## Default Decisions

- Represent tilt as ambient 3D vectors stored per vertex, then project to the tangent plane.
- Treat KH curvature coupling through the effective curvature quantities, not as an independent Helfrich mode.
- Reuse the repo's existing divergence/curvature operators and benchmark language when possible.
- If the target file appears to use a different normal orientation or sign convention, inspect adjacent tests and docs before editing.
- If the requested feature deviates from KH, state the deviation explicitly in the code comments, tests, and PR notes.

## When To Update This Skill

Update this skill whenever the repository changes the preferred or required membrane-physics contract that another Codex instance should follow. In practice, update it when any of the following changes:

- the KH or Helfrich algorithmic form used by default,
- tilt representation or tangent-projection rules,
- curvature-tilt sign conventions or normalization assumptions,
- leaflet-specific modeling conventions,
- required acceptance tests, regression tests, or performance-benchmark workflow.

Do not update the skill for local refactors that preserve the same algorithmic and validation contract.

## Trigger Examples

- "Add a Kozlov-Hamm tilt-splay energy term on a triangulated leaflet."
- "Review this curvature-tilt coupling for sign mistakes."
- "Why is my director field producing normal components during optimization?"
- "Add a benchmark for KH tilt decay on a flat annulus."
- "Map the Hamm-Kozlov paper equations onto this repo's dense-array APIs."

## References

- [references/kh-core.md](references/kh-core.md): KH/Helfrich identities, definitions, and sign constraints from the Hamm-Kozlov paper.
- [references/repo-map.md](references/repo-map.md): Where those concepts live in this repo and which implementation patterns to preserve.
- [references/validation-workflow.md](references/validation-workflow.md): Test and benchmark expectations for KH-related changes.
