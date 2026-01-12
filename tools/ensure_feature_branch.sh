#!/usr/bin/env bash
set -euo pipefail

if [[ "${ALLOW_MAIN_BRANCH:-}" == "1" ]]; then
  exit 0
fi

branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
if [[ -z "$branch" ]]; then
  echo "Unable to determine git branch. Create a feature branch or set ALLOW_MAIN_BRANCH=1." >&2
  exit 1
fi

if [[ "$branch" == "HEAD" ]]; then
  echo "Detached HEAD. Create a feature branch or set ALLOW_MAIN_BRANCH=1." >&2
  exit 1
fi

if [[ "$branch" == "main" || "$branch" == "master" ]]; then
  echo "Refusing to run on $branch. Create a feature branch or set ALLOW_MAIN_BRANCH=1." >&2
  exit 1
fi
