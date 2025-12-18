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
        description="Visualize membrane geometries from JSON files."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="meshes/cube.json",
        help="Path to a geometry JSON file (default: meshes/cube.json).",
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
    parser.add_argument(
        "--save",
        metavar="PATH",
        help="Save the rendered figure to PATH instead of only showing it.",
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

    setup_logging("membrane_solver.log")

    input_path = args.input
    if not input_path.endswith(".json"):
        raise ValueError("Input file must be a JSON file (.json).")
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file '{input_path}' not found!")

    data = load_data(input_path)
    mesh = parse_geometry(data=data)

    draw_facets = not args.no_facets
    draw_edges = not args.no_edges

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
        no_axes=args.no_axes,
        show=show,
    )

    if args.save:
        fig = plt.gcf()
        fig.savefig(args.save, bbox_inches="tight")
        logger.info("Saved visualization to %s", args.save)
