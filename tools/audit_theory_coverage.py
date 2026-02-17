#!/usr/bin/env python3
"""Audit theory-to-implementation coverage for the parity fixture."""

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


def _check_yaml(geom: dict[str, Any], check: dict[str, Any]) -> tuple[bool, str]:
    kind = str(check.get("kind", "")).strip()
    path = str(check.get("path", "")).strip()
    ok, value = _resolve_path(geom, path)

    if kind == "path_exists":
        return bool(ok), f"{kind}:{path}"
    if kind == "path_equals":
        expected = check.get("expected")
        return bool(ok and value == expected), f"{kind}:{path}"
    if kind == "list_contains":
        expected = check.get("expected")
        return bool(
            ok and isinstance(value, list) and expected in value
        ), f"{kind}:{path}"

    return False, f"unknown_yaml_check:{kind}:{path}"


def _check_code_ref(root: Path, ref: dict[str, Any]) -> tuple[bool, str]:
    rel = str(ref.get("path", "")).strip()
    contains = ref.get("contains")
    path = root / rel
    if not path.exists():
        return False, f"missing_file:{rel}"
    if contains is None:
        return True, f"file_exists:{rel}"
    text = path.read_text(encoding="utf-8")
    needle = str(contains)
    return needle in text, f"contains:{rel}"


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
        item_id = str(item.get("id", ""))
        required = bool(item.get("required", False))
        if required:
            required_count += 1

        tex_checks = []
        for marker in item.get("tex_markers", []):
            marker_s = str(marker)
            ok = marker_s in tex_text
            tex_checks.append({"marker": marker_s, "ok": bool(ok)})

        code_checks = []
        for ref in item.get("code_refs", []):
            ok, desc = _check_code_ref(root, ref)
            code_checks.append(
                {
                    "path": str(ref.get("path", "")),
                    "contains": ref.get("contains"),
                    "ok": bool(ok),
                    "detail": desc,
                }
            )

        yaml_checks = []
        for chk in item.get("yaml_checks", []):
            ok, desc = _check_yaml(geometry, chk)
            yaml_checks.append(
                {
                    "kind": chk.get("kind"),
                    "path": chk.get("path"),
                    "expected": chk.get("expected"),
                    "ok": bool(ok),
                    "detail": desc,
                }
            )

        buckets = [
            all(c["ok"] for c in tex_checks) if tex_checks else True,
            all(c["ok"] for c in code_checks) if code_checks else True,
            all(c["ok"] for c in yaml_checks) if yaml_checks else True,
        ]
        all_ok = all(buckets)
        any_ok = (
            any(c["ok"] for c in (tex_checks + code_checks + yaml_checks))
            if (tex_checks or code_checks or yaml_checks)
            else False
        )

        status = "present" if all_ok else ("partial" if any_ok else "missing")
        if required and status != "present":
            required_not_present += 1

        items_out.append(
            {
                "id": item_id,
                "required": required,
                "theory_statement": item.get("theory_statement"),
                "status": status,
                "tex_checks": tex_checks,
                "code_checks": code_checks,
                "yaml_checks": yaml_checks,
                "notes": item.get("notes"),
            }
        )

    summary = {
        "item_count": len(items_out),
        "present_count": sum(1 for x in items_out if x["status"] == "present"),
        "partial_count": sum(1 for x in items_out if x["status"] == "partial"),
        "missing_count": sum(1 for x in items_out if x["status"] == "missing"),
        "required_count": required_count,
        "required_not_present_count": required_not_present,
    }
    return {"items": items_out, "summary": summary}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tex", type=Path, default=DEFAULT_TEX)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--fail-on-required-not-present",
        action="store_true",
        help="Exit non-zero when any required item is not 'present'.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    tex_path = Path(args.tex)
    fixture_path = Path(args.fixture)
    manifest_path = Path(args.manifest)
    out_path = Path(args.out)

    tex_text = tex_path.read_text(encoding="utf-8")
    geometry = _load_yaml(fixture_path)
    manifest = _load_yaml(manifest_path)

    result = evaluate_manifest(
        root=ROOT,
        tex_text=tex_text,
        geometry=geometry,
        manifest=manifest,
    )
    report = {
        "meta": {
            "tex": str(tex_path.relative_to(ROOT)),
            "fixture": str(fixture_path.relative_to(ROOT)),
            "manifest": str(manifest_path.relative_to(ROOT)),
            "format": "yaml",
        },
        "summary": result["summary"],
        "items": result["items"],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    print(f"wrote: {out_path}")

    if bool(args.fail_on_required_not_present):
        return 1 if int(report["summary"]["required_not_present_count"]) > 0 else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
