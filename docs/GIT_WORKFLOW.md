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

  6. Merge back to main when ready
  Once the branch is feature-complete and tested:

  git checkout main
  git pull origin main          # get latest main
  git merge feature/mean-curvature
  pytest -q                     # optional final verification
  git push origin main          # publish the updated main

  Then delete the branch if you like:

  git branch -d feature/mean-curvature              # delete local branch
  git push origin --delete feature/mean-curvature   # delete remote branch

  7. Daily cleanup
  Before starting new work, make sure git status is clean (no leftover changes). If you paused mid-feature, just continue on that branch.
