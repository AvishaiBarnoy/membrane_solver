#!/usr/bin/env python3
"""Benchmark full minimization using different surface energy modules.

This script runs ``main.py`` on ``meshes/cube.json`` using both the old and the
new surface energy implementations. The simulation is executed multiple times for
each variant and the average run time is reported.
"""

from __future__ import annotations

import json
import subprocess
import sys, os
import time
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

BASE_JSON = Path("../meshes/cube.json")
RUNS = 5


def _prepare_input(module: str, tmpdir: Path) -> Path:
    """Return a temporary cube mesh with ``module`` as the surface energy."""
    with BASE_JSON.open() as fh:
        data = json.load(fh)

    new_faces = []
    for entry in data["faces"]:
        if isinstance(entry[-1], dict):
            opts = dict(entry[-1])
            edges = entry[:-1]
        else:
            opts = {}
            edges = entry
        opts["energy"] = [module]
        new_faces.append([*edges, opts])
    data["faces"] = new_faces

    path = tmpdir / f"cube_{module}.json"
    with path.open("w") as fh:
        json.dump(data, fh)
    return path


def _run_simulation(input_path: Path, output_path: Path) -> float:
    """Execute ``main.py`` and return the elapsed time."""
    start = time.perf_counter()

    main_py_path = Path(__file__).resolve().parent.parent / "main.py"

    subprocess.run(
        [sys.executable, str(main_py_path), "-i", str(input_path), "-o", str(output_path), "-q"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return time.perf_counter() - start


def benchmark(module: str, runs: int = RUNS) -> float:
    """Return the average run time for ``module`` over ``runs`` executions."""
    with TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        inp = _prepare_input(module, tmpdir)
        out = tmpdir / "out.json"
        times = [_run_simulation(inp, out) for _ in range(runs)]
    return sum(times) / runs


if __name__ == "__main__":
    for mod in ("surface_old", "surface"):
        avg = benchmark(mod)
        print(f"{mod} average runtime over {RUNS} runs: {avg:.4f}s")
