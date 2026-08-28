from __future__ import annotations

import pytest

import fortran_kernels.loader as kernel_loader
from tools.diagnostics.free_disk_profile_protocol import (
    run_free_disk_two_stage_profile_protocol,
)

pytestmark = pytest.mark.unit


def test_profile_protocol_spawn_isolates_parent_numerical_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def poisoned_kernel_lookup():
        raise RuntimeError("parent kernel cache was used")

    monkeypatch.setattr(
        kernel_loader, "get_tilt_curvature_kernel", poisoned_kernel_lookup
    )

    with pytest.raises(RuntimeError, match="parent kernel cache was used"):
        run_free_disk_two_stage_profile_protocol(shape_steps=1, isolate_process=False)

    mesh, theta_b = run_free_disk_two_stage_profile_protocol(shape_steps=1)
    assert theta_b > 0.0
    assert mesh.global_parameters.get("profile_protocol_process_isolated") is True
