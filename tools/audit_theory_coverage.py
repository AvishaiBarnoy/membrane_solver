#!/usr/bin/env python3
"""Audit required theory-to-code/fixture mappings for parity runs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEX = ROOT / "docs" / "tex" / "1_disk_3d.tex"
DEFAULT_FIXTURE = (
    ROOT / "tests" / "fixtures" / "kozlov_1disk_3d_free_disk_theory_parity.yaml"
)
DEFAULT_MANIFEST = ROOT / "tests" / "fixtures" / "theory_coverage_manifest.yaml"
DEFAULT_OUT = (
    ROOT / "benchmarks" / "outputs" / "diagnostics" / "theory_coverage_audit.yaml"
)


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _resolve_path(data: dict[str, Any], dotted: str) -> tuple[bool, Any]:
    cur: Any = data
    for token in dotted.split("."):
        if not isinstance(cur, dict) or token not in cur:
            return False, None
        cur = cur[token]
    return True, cur


def _check_yaml(geom: dict[str, Any], check: dict[str, Any]) -> bool:
    kind = str(check.get("kind", "")).strip()
    path = str(check.get("path", "")).strip()
    ok, value = _resolve_path(geom, path)
    if kind == "path_exists":
        return bool(ok)
    if kind == "path_equals":
        return bool(ok and value == check.get("expected"))
    if kind == "list_contains":
        return bool(ok and isinstance(value, list) and check.get("expected") in value)
    return False


def _check_code(root: Path, ref: dict[str, Any]) -> bool:
    rel = str(ref.get("path", "")).strip()
    path = root / rel
    if not path.exists():
        return False
    needle = ref.get("contains")
    if needle is None:
        return True
    return str(needle) in path.read_text(encoding="utf-8")


def evaluate_manifest(
    *,
    root: Path,
    tex_text: str,
    geometry: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    items_out: list[dict[str, Any]] = []
    required_count = 0
    required_not_present = 0

    for item in manifest.get("items", []):
        required = bool(item.get("required", False))
        if required:
            required_count += 1

        tex_ok = all(str(m) in tex_text for m in item.get("tex_markers", []))
        code_ok = all(_check_code(root, r) for r in item.get("code_refs", []))
        yaml_ok = all(_check_yaml(geometry, c) for c in item.get("yaml_checks", []))

        status = "present" if (tex_ok and code_ok and yaml_ok) else "missing"
        if required and status != "present":
            required_not_present += 1

        items_out.append(
            {
                "id": str(item.get("id", "")),
                "required": required,
                "status": status,
                "theory_statement": item.get("theory_statement"),
            }
        )

    summary = {
        "item_count": len(items_out),
        "present_count": sum(1 for x in items_out if x["status"] == "present"),
        "missing_count": sum(1 for x in items_out if x["status"] == "missing"),
        "required_count": required_count,
        "required_not_present_count": required_not_present,
    }
    return {"items": items_out, "summary": summary}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tex", type=Path, default=DEFAULT_TEX)
    p.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--fail-on-required-not-present", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    report = evaluate_manifest(
        root=ROOT,
        tex_text=Path(args.tex).read_text(encoding="utf-8"),
        geometry=_load_yaml(Path(args.fixture)),
        manifest=_load_yaml(Path(args.manifest)),
    )
    out = {
        "meta": {
            "tex": str(Path(args.tex).relative_to(ROOT)),
            "fixture": str(Path(args.fixture).relative_to(ROOT)),
            "manifest": str(Path(args.manifest).relative_to(ROOT)),
            "format": "yaml",
        },
        "summary": report["summary"],
        "items": report["items"],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(out, sort_keys=False), encoding="utf-8")
    print(f"wrote: {out_path}")
    if bool(args.fail_on_required_not_present):
        return 1 if int(out["summary"]["required_not_present_count"]) > 0 else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
