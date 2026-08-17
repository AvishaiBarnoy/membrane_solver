from __future__ import annotations

import sys

import pytest

from tools.diagnostics.flat_disk_kh_runtime_probe import main


def test_runtime_probe_rejects_zero_repeats(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "flat_disk_kh_runtime_probe.py",
            "--repeats",
            "0",
            "--output",
            str(tmp_path / "out.yaml"),
        ],
    )
    with pytest.raises(ValueError, match="repeats must be >= 1"):
        main()
