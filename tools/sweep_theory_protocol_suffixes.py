#!/usr/bin/env python3
"""Sweep fixed-protocol suffix candidates and rank theory-ratio alignment."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.reproduce_theory_parity import (
    DEFAULT_MESH,
    DEFAULT_PROTOCOL,
    ROOT,
    _collect_report,
)

DEFAULT_OUT = (
    ROOT / "benchmarks" / "outputs" / "diagnostics" / "theory_protocol_sweep.yaml"
)
DEFAULT_CANDIDATES = ("base:", "g1:g1", "t1e-3_g4:t1e-3;g4")
RATIO_KEYS = ("theta_ratio", "elastic_ratio", "contact_ratio", "total_ratio")


def parse_candidate(spec: str) -> tuple[str, tuple[str, ...]]:
    """Parse `label:cmd1;cmd2` format into (label, suffix_commands)."""
    text = str(spec).strip()
    if ":" not in text:
        raise ValueError(f"invalid candidate spec (missing ':'): {spec}")
    label, suffix_raw = text.split(":", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"invalid candidate spec (empty label): {spec}")
    suffix = tuple(cmd.strip() for cmd in suffix_raw.split(";") if cmd.strip())
    return label, suffix


def _finite(report: dict[str, Any]) -> bool:
    reduced = report["metrics"]["reduced_terms"]
    vals = [
        float(report["metrics"]["final_energy"]),
        float(reduced["elastic_measured"]),
        float(reduced["contact_measured"]),
        float(reduced["total_measured"]),
    ]
    vals.extend(float(report["metrics"]["theory"]["ratios"][k]) for k in RATIO_KEYS)
    return bool(np.all(np.isfinite(np.asarray(vals, dtype=float))))


def _sign_ok(report: dict[str, Any]) -> bool:
    reduced = report["metrics"]["reduced_terms"]
    return bool(
        float(reduced["elastic_measured"]) > 0.0
        and float(reduced["contact_measured"]) < 0.0
        and float(reduced["total_measured"]) < 0.0
    )


def _score(ratios: dict[str, float], target: float) -> float:
    return float(sum(abs(float(ratios[k]) - float(target)) for k in RATIO_KEYS))


def rank_candidates(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Stable ranking by validity, score, runtime, then label."""
    return sorted(
        entries,
        key=lambda e: (
            not bool(e["checks"]["valid"]),
            float(e["score"]),
            float(e["runtime_s"]),
            str(e["label"]),
        ),
    )


def run_candidate(
    *,
    mesh: Path,
    base_protocol: tuple[str, ...],
    label: str,
    suffix: tuple[str, ...],
    repeat: int,
    target_ratio: float,
) -> dict[str, Any]:
    runtimes: list[float] = []
    scores: list[float] = []
    ratios_by_key: dict[str, list[float]] = {k: [] for k in RATIO_KEYS}
    finite_all = True
    sign_all = True
    last_report: dict[str, Any] | None = None

    protocol = base_protocol + suffix
    for _ in range(int(repeat)):
        t0 = time.perf_counter()
        report = _collect_report(mesh_path=mesh, protocol=protocol)
        runtimes.append(time.perf_counter() - t0)
        last_report = report
        ratios = {
            k: float(report["metrics"]["theory"]["ratios"][k]) for k in RATIO_KEYS
        }
        for key in RATIO_KEYS:
            ratios_by_key[key].append(float(ratios[key]))
        scores.append(_score(ratios, target_ratio))
        finite_all = finite_all and _finite(report)
        sign_all = sign_all and _sign_ok(report)

    mean_ratios = {k: float(np.mean(v)) for k, v in ratios_by_key.items()}
    entry = {
        "label": label,
        "suffix": list(suffix),
        "protocol": list(protocol),
        "runtime_s": float(np.mean(runtimes)),
        "score": float(np.mean(scores)),
        "ratios": mean_ratios,
        "delta_to_target": {
            k: float(mean_ratios[k] - float(target_ratio)) for k in RATIO_KEYS
        },
        "checks": {
            "finite": bool(finite_all),
            "sign_relations": bool(sign_all),
            "valid": bool(finite_all and sign_all),
        },
        "final_energy": float(last_report["metrics"]["final_energy"])
        if last_report
        else None,
    }
    return entry


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mesh", type=Path, default=DEFAULT_MESH)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--target-ratio", type=float, default=2.0)
    p.add_argument(
        "--candidate",
        action="append",
        default=None,
        help="Candidate spec in format `label:cmd1;cmd2`. Repeat flag to add more.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if int(args.repeat) <= 0:
        raise ValueError("--repeat must be >= 1")
    specs = args.candidate if args.candidate else list(DEFAULT_CANDIDATES)
    candidates = [parse_candidate(s) for s in specs]

    entries: list[dict[str, Any]] = []
    for label, suffix in candidates:
        entries.append(
            run_candidate(
                mesh=Path(args.mesh),
                base_protocol=tuple(DEFAULT_PROTOCOL),
                label=label,
                suffix=suffix,
                repeat=int(args.repeat),
                target_ratio=float(args.target_ratio),
            )
        )
    ranked = rank_candidates(entries)
    best = ranked[0] if ranked else None
    out = {
        "meta": {
            "fixture": str(Path(args.mesh).relative_to(ROOT)),
            "base_protocol": list(DEFAULT_PROTOCOL),
            "format": "yaml",
            "repeat": int(args.repeat),
            "target_ratio": float(args.target_ratio),
        },
        "candidates": ranked,
        "summary": {
            "candidate_count": len(ranked),
            "valid_count": sum(1 for x in ranked if x["checks"]["valid"]),
            "best_candidate": None if best is None else best["label"],
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(out, sort_keys=False), encoding="utf-8")
    print(f"wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
