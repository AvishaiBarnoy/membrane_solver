#!/bin/bash

# Run cProfile for each benchmark case and store per-case .pstats outputs.
# This script is intended to be runnable from any directory.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PROFILE_DIR="${SCRIPT_DIR}/benchmarks/outputs/profiles"

python benchmarks/suite.py --profile --profile-dir "$PROFILE_DIR" --profile-top 30

echo
read -r -p "Delete profile output directory '$PROFILE_DIR'? [y/N] " REPLY
case "$REPLY" in
    [yY][eE][sS]|[yY])
        rm -rf "$PROFILE_DIR"
        echo "Deleted '$PROFILE_DIR'."
        ;;
    *)
        echo "Kept '$PROFILE_DIR'."
        ;;
esac
