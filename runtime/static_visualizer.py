"""Simple PyVista viewer for a static mesh."""

from __future__ import annotations

import argparse

from geometry.geom_io import load_data, parse_geometry
from geometry.entities import Mesh
from visualize_geometry import plot_geometry


def show_mesh(mesh: Mesh) -> None:
    """Open a PyVista window with the given mesh."""
    plot_geometry(mesh)


def main() -> None:
    parser = argparse.ArgumentParser(description="Display a mesh using PyVista")
    parser.add_argument("input", help="Input mesh JSON file")
    args = parser.parse_args()

    data = load_data(args.input)
    mesh = parse_geometry(data)
    show_mesh(mesh)


if __name__ == "__main__":
    main()
