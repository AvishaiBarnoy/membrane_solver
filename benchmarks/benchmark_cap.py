#!/usr/bin/env python3
"""Benchmark for Pinned Spherical Cap (Laplace Law verification).

This script runs ``main.py`` on ``meshes/good_min_cap.json`` and verifies
that the final geometry approximates a spherical cap of radius 1.

It also serves as a general analysis tool for checking if a mesh is a spherical cap.
Usage:
    python benchmark_cap.py             # Runs benchmark
    python benchmark_cap.py <json_file> # Analyzes specific file
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# Ensure the project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import load_data, parse_geometry

BASE_JSON = Path(__file__).resolve().parent.parent / "meshes" / "good_min_cap.json"
OUTPUT_JSON = Path(__file__).resolve().parent.parent / "outputs" / "cap_results.json"
RUNS = 1

def _run_simulation(input_path: Path, output_path: Path) -> float:
    """Execute ``main.py`` and return the elapsed time."""
    start = time.perf_counter()

    main_py_path = Path(__file__).resolve().parent.parent / "main.py"

    subprocess.run(
        [
            sys.executable,
            str(main_py_path),
            "-i",
            str(input_path),
            "-o",
            str(output_path),
            "--non-interactive",
            "-q",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return time.perf_counter() - start

def solve_cap_height(volume, radius):
    """
    Solve for the height h of a spherical cap given Volume V and base radius a.
    Formula: V = (pi * h / 6) * (3a^2 + h^2)
    """
    import math

    a = radius
    V = volume

    # Initial guess
    h = 1.0

    for _ in range(20):
        f = (math.pi/6) * h**3 + (math.pi * a**2 / 2) * h - V
        df = (math.pi/2) * h**2 + (math.pi * a**2 / 2)
        h_new = h - f / df
        if abs(h_new - h) < 1e-6:
            return h_new
        h = h_new
    return h

def analyze_mesh_quality(mesh, expected_volume=None, expected_radius=None):
    """Performs geometric analysis: fit to sphere, check radius, check volume."""
    print("--- Mesh Quality Analysis ---")

    vertices = np.array([v.position for v in mesh.vertices.values()])
    if len(vertices) == 0:
        print("Mesh has no vertices.")
        return

    # 1. Apex Height
    max_z = np.max(vertices[:, 2])
    print(f"Apex Height (Max Z): {max_z:.5f}")

    # 2. Base Radius
    base_mask = np.abs(vertices[:, 2]) < 0.05 * (max_z if max_z > 0 else 1.0)
    base_vertices = vertices[base_mask]

    avg_base_radius = None
    if len(base_vertices) > 0:
        base_radii = np.linalg.norm(base_vertices[:, :2], axis=1)
        avg_base_radius = np.mean(base_radii)
        std_base_radius = np.std(base_radii)
        print(f"Base Radius: {avg_base_radius:.5f} +/- {std_base_radius:.5f}")
    else:
        print("No base vertices found (flat base assumption failed).")

    # 3. Sphere Fit (RMSE)
    # Fit (x-xc)^2 + (y-yc)^2 + (z-zc)^2 = R^2
    A = np.c_[2*vertices[:,0], 2*vertices[:,1], 2*vertices[:,2], np.ones(len(vertices))]
    B = np.sum(vertices**2, axis=1)

    try:
        X, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)
        xc, yc, zc, C = X
        R = np.sqrt(C + xc**2 + yc**2 + zc**2)
        print(f"Best Fit Sphere: Center=({xc:.4f}, {yc:.4f}, {zc:.4f}), Radius={R:.4f}")

        dists = np.linalg.norm(vertices - np.array([xc, yc, zc]), axis=1)
        rmse = np.sqrt(np.mean((dists - R)**2))
        print(f"Spherical RMSE: {rmse:.5f} (Lower is better)")
    except Exception as e:
        print(f"Sphere fit failed: {e}")

    # 4. Volume Verification
    try:
        actual_vol = mesh.compute_total_volume()
        print(f"Computed Mesh Volume: {actual_vol:.5f}")

        if expected_volume and expected_radius:
            theoretical_h = solve_cap_height(expected_volume, expected_radius)
            print(f"Theoretical Height for V={expected_volume:.4f}, R={expected_radius:.4f} is h={theoretical_h:.5f}")

            # Comparison
            if abs(max_z - theoretical_h) > 0.05 * theoretical_h:
                 print("WARNING: Apex height mismatch!")
            else:
                 print("SUCCESS: Apex height matches theory.")

    except Exception as e:
        print(f"Volume computation failed: {e}")

def verify_results(input_path: Path, output_path: Path):
    # Load parameters from input
    with open(input_path, 'r') as f:
        input_data = json.load(f)

    radius = 1.0
    if "definitions" in input_data and "bottom_ring" in input_data["definitions"]:
        radius = input_data["definitions"]["bottom_ring"].get("pin_to_circle_radius", 1.0)

    volume = 0.0
    if "bodies" in input_data and "target_volume" in input_data["bodies"]:
        volume = input_data["bodies"]["target_volume"][0]

    print(f"\nVerifying {output_path} against Input Parameters:")
    print(f"Target Radius={radius}, Target Volume={volume}")

    # Load result mesh
    try:
        data = load_data(str(output_path))
        mesh = parse_geometry(data)
        analyze_mesh_quality(mesh, expected_volume=volume, expected_radius=radius)
    except Exception as e:
        print(f"Failed to load result mesh: {e}")

def benchmark(runs: int = RUNS) -> float:
    """Return the average runtime over ``runs`` executions."""
    times = []
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    print(f"Running benchmark on {BASE_JSON} -> {OUTPUT_JSON}...")
    for i in range(runs):
        elapsed = _run_simulation(BASE_JSON, OUTPUT_JSON)
        times.append(elapsed)

    verify_results(BASE_JSON, OUTPUT_JSON)
    return sum(times) / runs

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Analysis mode
        target_file = Path(sys.argv[1])
        if target_file.exists():
            print(f"Analyzing {target_file}...")
            data = load_data(str(target_file))
            mesh = parse_geometry(data)
            analyze_mesh_quality(mesh)
        else:
            print(f"File not found: {target_file}")
    else:
        # Benchmark mode
        avg = benchmark()
        print(f"Average runtime over {RUNS} runs: {avg:.4f}s")
