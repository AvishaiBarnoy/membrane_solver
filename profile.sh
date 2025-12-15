#!/bin/bash

# Run cProfile on the cube_good_min_routine setup and print top hotspots.
# This script is intended to be runnable from any directory.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PROFILE_OUT="${SCRIPT_DIR}/profile.out"

python -m cProfile -o "$PROFILE_OUT" main.py \
    -i meshes/cube_good_min.json --non-interactive -q

python -m pstats "$PROFILE_OUT" << EOF
sort cumulative
stats 30
EOF

echo
read -r -p "Delete profile output file '$PROFILE_OUT'? [y/N] " REPLY
case "$REPLY" in
    [yY][eE][sS]|[yY])
        rm -f "$PROFILE_OUT"
        echo "Deleted '$PROFILE_OUT'."
        ;;
    *)
        echo "Kept '$PROFILE_OUT'."
        ;;
esac
