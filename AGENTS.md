# AGENTS Development Guide

This repository simulates membranes and surfaces using Python. The code is
organized under `geometry/`, `modules/`, `runtime/` and other directories.

## Coding Guidelines
- Use Python 3.10+ and follow PEP8 style.
- Employ dataclasses for domain objects and include type hints where practical.
- Document public functions with standard docstrings.
- Prioritize efficient algorithms and data structures.

## Testing
- Install dependencies with `pip install -r requirements.txt`.
- Run tests from the repository root using `pytest -q`.
- Ensure all tests pass before committing any changes.
- When adding new functionality, include relevant unit tests.
- When editing existing code, verify that tests cover the change. Add tests if
  they are missing.

## Pull Requests
- Summarize your changes clearly in commit messages and PR descriptions.
- Provide detailed explanations in the PR body so reviewers understand the rationale behind the changes.
- Mention the output of `pytest -q` in the PR body.

## Files to update
- Update relevant .md files: README.md, MANUAL.md, CHANGELOG.md, docs/ROADMAP.md
