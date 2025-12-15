 1. Initial setup (done once)

  - Ensure your name/email are set:

    git config --global user.name "Your Name"
    git config --global user.email "you@example.com"
  - Verify status:

    git status

  2. Start new work (create a branch)
  Pick a descriptive name, e.g. feature/mean-curvature or perf/minimizer-cache.

  # Make sure main is up to date
  git checkout main
  git pull origin main        # if you have a remote

  # Create and switch to feature branch
  git checkout -b feature/mean-curvature

  You’re now on the new branch; any commits stay isolated here.

  3. Edit and test

  - Make your code changes.
  - Run formatting/tests:

    pytest -q
  - Inspect what changed:

    git status
    git diff                  # or git diff path/to/file

  4. Stage and commit
  Stage only the files you intend to commit:

  git add CHANGELOG.md runtime/minimizer.py tests/test_cube_minimization.py

  Commit with a descriptive message:

  git commit -m "Add cube minimization test and optimize minimizer"

  You can repeat steps 3–4 as often as you like (“save points” on this branch).

  5. Push and share (optional)
  If you’re using GitHub or another remote, push the branch:

  git push origin feature/mean-curvature

  This gives you a remote backup and lets you open a PR later. If you’re working locally only, you can skip this.

  6. Create a Pull Request
  Once your branch is pushed, you can create a PR via the browser or the command line.

  **Option A: Browser**
  Navigate to your repository on GitHub (or equivalent) and open a Pull Request (PR).

  **Option B: GitHub CLI**
  If you have `gh` installed:

  gh pr create --title "My Feature" --body "Description of changes"

  Regardless of the method:
  - Provide a clear title and description for your changes.
  - Link any relevant issues or documentation.
  - Request reviews from appropriate team members.

  7. Review and Merge Pull Request
  Once your PR has been approved and all checks pass:

  - Ensure your local main branch is up to date:
    git checkout main
    git pull origin main

  - Merge the PR through the GitHub interface (preferred for better tracking and automation).
  - After merging, you can pull the updated main branch locally:
    git pull origin main

  - Then delete the branch if you like:

  git branch -d feature/mean-curvature              # delete local branch
  git push origin --delete feature/mean-curvature   # delete remote branch

  8. Daily cleanup
  Before starting new work, make sure git status is clean (no leftover changes). If you paused mid-feature, just continue on that branch.
