"""Render a mesh before/after running a macro sequence.

This is a lightweight helper for comparing workflows against Surface Evolver
macros. It runs a named macro N times via the normal command executor and saves
a PNG snapshot using the headless Matplotlib backend.

Example:
  python tools/render_macro_snapshot.py \
    --mesh meshes/caveolin/annulus_flat_no_tilt.yaml \
    --macro gogo --repeat 2 \
    --out outputs/annulus_flat_no_tilt_gogo2.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True, help="Path to a YAML/JSON mesh.")
    parser.add_argument("--macro", default="gogo", help="Macro name to execute.")
    parser.add_argument("--repeat", type=int, default=1, help="Number of repeats.")
    parser.add_argument(
        "--out", required=True, help="Output PNG path (parent dirs created)."
    )
    parser.add_argument(
        "--color-by",
        default=None,
        help="Optional visualization mode (e.g. tilt_mag, tilt_in, tilt_bilayer).",
    )
    args = parser.parse_args()

    repo_root = _repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from commands.context import CommandContext
    from commands.executor import execute_command_line
    from geometry.geom_io import load_data, parse_geometry
    from runtime.constraint_manager import ConstraintModuleManager
    from runtime.energy_manager import EnergyModuleManager
    from runtime.minimizer import Minimizer
    from runtime.steppers.gradient_descent import GradientDescent
    from visualization.plotting import plot_geometry

    mesh = parse_geometry(load_data(args.mesh))
    minim = Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
    )
    ctx = CommandContext(mesh, minim, minim.stepper)

    for _ in range(max(0, int(args.repeat))):
        execute_command_line(ctx, str(args.macro))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plot_geometry(ctx.mesh, ax=ax, show=False, draw_edges=True, color_by=args.color_by)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
