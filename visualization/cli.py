import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import logging
import os
from typing import Optional, Sequence

import matplotlib.pyplot as plt

from geometry.geom_io import load_data, parse_geometry
from runtime.logging_config import setup_logging
from visualization.plotting import plot_geometry

logger = logging.getLogger("membrane_solver")


def create_parser() -> argparse.ArgumentParser:
    """
    Create an argument parser for the visualization command‑line interface.
    """
    parser = argparse.ArgumentParser(
        description="Visualize membrane geometries from JSON/YAML files."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="meshes/cube.json",
        help="Path to a geometry JSON/YAML file (default: meshes/cube.json).",
    )
    parser.add_argument(
        "--no-facets",
        action="store_true",
        help="Disable drawing of polygonal facets.",
    )
    parser.add_argument(
        "--no-edges",
        action="store_true",
        help="Disable drawing of edges.",
    )
    parser.add_argument(
        "--scatter",
        action="store_true",
        help="Draw vertices as red scatter points.",
    )
    parser.add_argument(
        "--show-indices",
        action="store_true",
        help="Annotate vertices with their indices.",
    )
    parser.add_argument(
        "--transparent",
        action="store_true",
        help="Render facets semi‑transparent.",
    )
    tilt_group = parser.add_mutually_exclusive_group()
    tilt_group.add_argument(
        "--tilt",
        action="store_true",
        help="Color facets by |t| (tilt magnitude).",
    )
    tilt_group.add_argument(
        "--tilt-div",
        action="store_true",
        help="Color facets by div(t).",
    )
    parser.add_argument(
        "--tilt-arrows",
        action="store_true",
        help="Overlay arrows showing vertex tilt direction.",
    )
    parser.add_argument(
        "--tilt-arrows-max",
        type=int,
        default=2000,
        help="Maximum number of tilt arrows to draw (default: 2000).",
    )
    parser.add_argument(
        "--tilt-arrow-scale",
        type=float,
        default=0.1,
        help="Arrow length as a fraction of the plot span (default: 0.1).",
    )
    parser.add_argument(
        "--tilt-streamlines",
        action="store_true",
        help="Overlay simple mesh-graph streamlines following the tilt field.",
    )
    parser.add_argument(
        "--tilt-streamlines-max",
        type=int,
        default=200,
        help="Maximum number of tilt streamlines to draw (default: 200).",
    )
    parser.add_argument(
        "--tilt-streamlines-steps",
        type=int,
        default=80,
        help="Maximum steps per streamline (default: 80).",
    )
    parser.add_argument(
        "--tilt-streamlines-cos-min",
        type=float,
        default=0.2,
        help="Minimum cosine alignment to continue a streamline (default: 0.2).",
    )
    parser.add_argument(
        "--patch-boundaries",
        action="store_true",
        help="Overlay patch boundaries (edges separating facet patch labels).",
    )
    parser.add_argument(
        "--patch-key",
        default="disk_patch",
        help="Facet option key used for patch labels (default: disk_patch).",
    )
    parser.add_argument(
        "--boundary-loops",
        action="store_true",
        help='Overlay mesh boundary loops ("holes").',
    )
    parser.add_argument(
        "--boundary-geodesic",
        action="store_true",
        help="Annotate boundary loops with geodesic curvature sums (Gauss–Bonnet).",
    )
    parser.add_argument(
        "--save",
        metavar="PATH",
        help="Save the rendered figure to PATH instead of only showing it.",
    )

    parser.add_argument(
        "--log",
        nargs="?",
        const="auto",
        default=None,
        help="Write logs to a file. If PATH is omitted, writes a log file next to the input mesh.",
    )
    parser.add_argument("--no-axes", action="store_true", help="Removes axes from plot")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """
    Entry point for the visualization CLI.

    Parameters
    ----------
    argv :
        Optional sequence of command‑line arguments. When ``None``, the
        arguments are taken from ``sys.argv``.
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    input_path = args.input
    ext = os.path.splitext(input_path)[1].lower()
    if ext not in {".json", ".yaml", ".yml"}:
        raise ValueError("Input file must be a JSON or YAML file (.json/.yaml/.yml).")
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file '{input_path}' not found!")

    log_path = None
    if args.log is not None:
        if args.log == "auto":
            from pathlib import Path

            inp = Path(input_path)
            log_path = str(inp.with_suffix(".log"))
        else:
            log_path = str(args.log)
    setup_logging(log_path)

    data = load_data(input_path)
    mesh = parse_geometry(data=data)

    draw_facets = not args.no_facets
    draw_edges = not args.no_edges
    color_by = None
    if args.tilt_div:
        color_by = "tilt_div"
    elif args.tilt:
        color_by = "tilt_mag"

    # If saving but the user did not request otherwise, avoid blocking
    # interactive display. The figure can still be inspected manually
    # when using an interactive backend.
    show = args.save is None

    plot_geometry(
        mesh,
        show_indices=args.show_indices,
        scatter=args.scatter,
        transparent=args.transparent,
        draw_facets=draw_facets,
        draw_edges=draw_edges,
        facet_color=None,
        edge_color="k",
        facet_colors=None,
        edge_colors=None,
        color_by=color_by,
        show_tilt_arrows=args.tilt_arrows,
        tilt_arrows_max=None if args.tilt_arrows_max <= 0 else args.tilt_arrows_max,
        tilt_arrow_scale=args.tilt_arrow_scale,
        show_tilt_streamlines=args.tilt_streamlines,
        tilt_streamlines_max=args.tilt_streamlines_max,
        tilt_streamlines_steps=args.tilt_streamlines_steps,
        tilt_streamlines_cos_min=args.tilt_streamlines_cos_min,
        show_patch_boundaries=args.patch_boundaries,
        patch_key=args.patch_key,
        show_boundary_loops=args.boundary_loops,
        annotate_boundary_geodesic=args.boundary_geodesic,
        no_axes=args.no_axes,
        show=show,
    )

    if args.save:
        fig = plt.gcf()
        fig.savefig(args.save, bbox_inches="tight")
        logger.info("Saved visualization to %s", args.save)
