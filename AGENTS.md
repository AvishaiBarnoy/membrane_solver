# AGENTS Development Guide

This repository simulates membranes and surfaces using Python. The code is
organized under `geometry/`, `modules/`, `runtime/` and other directories.

---

## Operating Mode (Agentic Development)

### Branch Rule (MANDATORY BEFORE CODING)
Do not implement on the default branch or an unrelated existing branch.
Create and work from a new branch for each approved stream of work.

### Planning Gate (DEFAULT WORKFLOW)
Before starting substantial work, produce a **Plan Mode plan** and wait for user approval.
After the plan is approved, the coding agent may implement the approved substeps without
writing a separate Feature Contract for every small change.

The approved plan should cover:
- Goal
- Scope / non-goals
- Acceptance criteria
- Expected tests and validation
- Planned PR split when the work is large

Within an approved plan, the coding agent may:
- make small local design decisions
- try, compare, and discard implementation approaches
- add instrumentation, diagnostics, and focused tests needed to reach the goal
- make tightly related follow-up fixes discovered during implementation

### Design Gate (REQUIRED FOR HIGH-RISK CHANGES)
Produce a **Feature Contract** and wait for user approval before implementing any of the following:
- new features with meaningful user-visible behavior
- changes to public interfaces (functions, classes, CLI, config)
- data model or file format changes
- changes to `runtime/` entry points or main-flow behavior
- architecture changes affecting performance-sensitive paths, solvers, caching, or data layout
- changes to theory-facing model assumptions or physical interpretation
- any work that materially exceeds the approved plan

**Feature Contract (1 page max)**
- Goal (1 sentence)
- Non-goals (explicit exclusions)
- User-visible behavior (acceptance criteria; bullet list)
- Public interfaces to add/change (functions/classes/CLI/config)
- Data model / file format changes (if any)
- Key invariants (what must always remain true)
- Failure modes & error handling (how errors surface)
- Test plan (unit + integration/e2e; what each test proves)
- Performance considerations (if hot loops touched)

If anything is ambiguous, ask concise clarifying questions before implementing.

### Modeling Guardrails
- Do not introduce tuning knobs, fudge factors, or hidden parameters whose purpose is to force agreement with theoretical results.
- Small perturbations that break symmetry or improve numerical robustness are acceptable only when they are explicit, justified, and limited in scope.
- If a change affects numerical behavior, convergence, stability, or energy outcomes, document the assumption and validate it with tests and, when relevant, benchmarks.

---

## Change Budget (SMALL CHANGES ONLY)
To keep reviewable diffs and maintain control:

- Target: <= 300 lines changed per PR (excluding generated files).
- Target: <= 10 files changed per PR.
- One conceptual change per PR. No drive-by refactors.
- These are review-budget targets, not automatic rejection criteria.
- If you exceed the budget, either:
  - justify why the work is tightly coupled and more reviewable as one PR, or
  - propose a PR split before proceeding.
- Prefer the smallest diff that preserves coherent review. Do not split changes so aggressively that reviewers lose the full behavior in one place.

**Refactors:**
- Refactors must be separate PRs unless strictly required for the feature.
- If a refactor is required, justify it explicitly in the PR description.

---

## Coding Guidelines
- Use Python 3.10+ and follow PEP8 style.
- Employ dataclasses for domain objects and include type hints where practical.
- Document all functions with standard docstrings.
- Prioritize efficient algorithms and data structures.
- Debugging should be through logger.
- If requirements or physics/discretization details are unclear, ask the user concise clarifying questions BEFORE implementing; if proceeding, state the assumptions in the PR description.

---

## Performance & Architecture
- **Hybrid SoA Pattern**: Use the "Scatter-Gather" pattern for numerical optimization. Mesh topology remains object-oriented (AoS) for ease of manipulation, but minimization MUST use dense NumPy arrays (SoA).
- **Hot-Loop Vectorization**:
  - Energy and Constraint modules MUST implement `compute_energy_and_gradient_array` for performance.
  - Avoid Python loops over vertices, edges, or facets inside energy calculations.
  - Use `Mesh.positions_view()` and `Mesh.triangle_row_cache()` to obtain vectorized data.
- **Gradient Accumulation**: Use `np.add.at` or direct array modification instead of creating intermediate dictionaries.
- **Caching**: Always check entity versioning (`mesh._version`) before recalculating expensive geometric properties.
- **Performance Changes REQUIRE Benchmarks**:
  - Any change that may affect performance, numerical stability, convergence behavior, or memory usage
    (e.g. energy terms, gradients, constraints, solvers, vectorization, caching, data layout)
    MUST include a benchmark.
  - The benchmark MUST run the exact affected part of the simulation (not a synthetic proxy),
    using a representative mesh and parameter set.

---

## Testing (Acceptance-first)
- Install dependencies with `pip install -r requirements.txt`.
- Run tests from the repository root using `pytest -q`.
- Ensure all tests pass before committing any changes.

### Acceptance Tests BEFORE Main/Runtime Code
For new features or behavioral changes:
1. Translate acceptance criteria into **acceptance tests** (integration/e2e style) that validate system behavior end-to-end.
2. Add these tests FIRST, and confirm they fail for the right reason.
3. Only then implement the feature until tests pass.

**Hard rule:** Do not modify `runtime/` entry points or “main flow” behavior until acceptance tests exist.

### Unit + Regression Tests
- When adding new functionality, include relevant unit tests.
- When editing existing code, verify that tests cover the change; add tests if missing.
- For performance-sensitive code (energy/constraint hot loops), add regression tests that guard:
  - correctness (finite diff / directional derivative)
  - vectorization expectations (avoid per-entity Python loops in hot paths)

---

## TDD Workflow (Agentic)
- Create feature branch.
- Start with failing tests:
  - acceptance tests (integration/e2e) for behavior
  - unit tests for local invariants
- Make the smallest code change to pass the tests; avoid unrelated refactors.
- Refactor only after tests are green, keeping changes behavior-preserving.
- Prefer targeted test runs while iterating (e.g., `pytest -q tests/test_foo.py -k case_name`), then run the full suite before finalizing.
- Make checkpoint commits when they improve handoff, review, or rollback safety.
- Do not require a separate meaningful commit for every tiny TDD step during local iteration.
- Before requesting review or handing work to another agent/person, ensure the branch is in a reviewable committed state.

---

## Pull Requests (MANDATORY STRUCTURE)
Every PR MUST include the following sections in the PR body:
- All Pull Requests MUST follow the structure defined in `.github/pull_request_template.md`.
  Do not omit sections; write "N/A" if a section is not applicable.

**Hard rule:** PR must be reviewable: small diff, focused scope, and reproducible test steps.

**Hard rule:** Performance-related PRs without benchmarks are incomplete and must not be merged.

**Soft guidance:** Docs-only, test-only, or other low-risk housekeeping PRs may use lighter ceremony, but they must still stay focused and include clear validation steps.

---

## Files to update
- Update relevant .md files when needed: README.md, MANUAL.md, CHANGELOG.md, docs/ROADMAP.md
