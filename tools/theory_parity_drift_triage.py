#!/usr/bin/env python3
"""Print top offending theory-parity ratios from a trend YAML artifact."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DEFAULT_TREND = (
    Path("benchmarks") / "outputs" / "diagnostics" / "theory_parity_trend.yaml"
)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"trend file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"invalid trend YAML shape: expected mapping at root: {path}")
    return data


def top_offenders(
    trend: dict[str, Any], *, top: int
) -> list[tuple[str, dict[str, Any]]]:
    """Return sorted ratio rows by abs_delta descending, then name ascending."""
    ratios = trend.get("ratios")
    if not isinstance(ratios, dict) or not ratios:
        raise ValueError("invalid trend data: missing 'ratios' mapping")
    rows: list[tuple[str, dict[str, Any]]] = []
    for name, row in ratios.items():
        if not isinstance(row, dict):
            continue
        if "abs_delta" not in row:
            continue
        rows.append((str(name), row))
    if not rows:
        raise ValueError("invalid trend data: no ratio rows with 'abs_delta'")
    rows.sort(key=lambda x: (-float(x[1]["abs_delta"]), x[0]))
    return rows[: max(0, int(top))]


def format_triage(trend: dict[str, Any], *, top: int) -> str:
    """Format a concise triage summary from the trend report."""
    summary = trend.get("summary", {})
    offenders = top_offenders(trend, top=top)

    lines = []
    lines.append(
        "summary:"
        f" all_within_tolerance={summary.get('all_within_tolerance')}"
        f" within={summary.get('within_tolerance_count')}"
        f" total={summary.get('ratio_count')}"
    )
    lines.append(f"top_offenders(top={top}):")
    for name, row in offenders:
        lines.append(
            f"- {name}:"
            f" abs_delta={float(row['abs_delta']):.6g}"
            f" abs_tol={float(row['abs_tol']):.6g}"
            f" actual={float(row['actual']):.6g}"
            f" expected={float(row['expected']):.6g}"
            f" within_tolerance={bool(row['within_tolerance'])}"
        )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trend", type=Path, default=DEFAULT_TREND)
    parser.add_argument("--top", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        trend = _load_yaml(Path(args.trend))
        output = format_triage(trend, top=int(args.top))
    except (FileNotFoundError, ValueError, KeyError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
