# AGENTS Development Guide

This repository simulates membranes and surfaces using Python. The code is
organized under `geometry/`, `modules/`, `runtime/` and other directories.

## Coding Guidelines
- Use Python 3.10+ and follow PEP8 style.
- Employ dataclasses for domain objects and include type hints where practical.
- Document all functions with standard docstrings.
- Prioritize efficient algorithms and data structures.
- Debugging should be through logger.
- If requirements or physics/discretization details are unclear, ask the user concise clarifying questions before implementing; if proceeding, state the assumptions in the PR description.

## Performance & Architecture
- **Hybrid SoA Pattern**: Use the "Scatter-Gather" pattern for numerical optimization. Mesh topology remains object-oriented (AoS) for ease of manipulation, but minimization MUST use dense NumPy arrays (SoA).
- **Hot-Loop Vectorization**:
    - Energy and Constraint modules MUST implement `compute_energy_and_gradient_array` for performance.
    - Avoid Python loops over vertices, edges, or facets inside energy calculations.
    - Use `Mesh.positions_view()` and `Mesh.triangle_row_cache()` to obtain vectorized data.
- **Gradient Accumulation**: Use `np.add.at` or direct array modification instead of creating intermediate dictionaries.
- **Caching**: Always check entity versioning (`mesh._version`) before recalculating expensive geometric properties.

## Testing
- Install dependencies with `pip install -r requirements.txt`.
- Run tests from the repository root using `pytest -q`.
- Ensure all tests pass before committing any changes.
- When adding new functionality, include relevant unit tests.
- When editing existing code, verify that tests cover the change. Add tests if
  they are missing.

## TDD Workflow
- Create feature branch
- Checkout to new branch
- Start with a failing test that captures the intended behavior (bug repro or new feature).
- Make the smallest code change to pass the test; avoid unrelated refactors in the same step.
- Refactor only after tests are green, keeping changes behavior-preserving.
- Prefer targeted test runs while iterating (e.g., `pytest -q tests/test_foo.py -k case_name`), then run the full suite before finalizing.
- For performance-sensitive code (energy/constraint hot loops), add regression tests that guard both correctness (finite diff / directional derivative) and vectorization (avoid per-entity Python loops).
- Update documentation
- Commit changes with a clear message
- Push the branch and open a PR
- End each work chunk on a branch with a commit so it is ready for code review.

## Pull Requests
- Summarize your changes clearly in commit messages and PR descriptions.
- Provide detailed explanations in the PR body so reviewers understand the rationale behind the changes.
- Mention the output of `pytest -q` in the PR body.

## Files to update
- Update relevant .md files: README.md, MANUAL.md, CHANGELOG.md, docs/ROADMAP.md
