from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml

from tools.free_one_disc_convergence import (
    DEFAULT_BASE_FIXTURE,
    THEORY_THETA_B,
    FreeOneDiscCase,
    build_canonical_free_one_disc_fixture,
)
from tools.free_one_disc_protocol_comparison import (
    DEFAULT_PROTOCOLS,
    coupled_stationarity,
    prepare_feasible_state,
)
from tools.reproduce_theory_parity import _build_context


def test_protocol_matrix_uses_no_mesh_mutating_commands() -> None:
    commands = {
        command for protocol in DEFAULT_PROTOCOLS for command in protocol.commands
    }

    assert not any(command.startswith(("r", "v", "u")) for command in commands)
    assert {
        "gd_fixed_reference",
        "gd_finalize_each_step",
        "gd_adaptive",
        "cg_adaptive",
        "bfgs_adaptive",
        "hessian_hybrid",
    } == {protocol.label for protocol in DEFAULT_PROTOCOLS}


def test_coupled_stationarity_is_finite_and_does_not_mutate_state() -> None:
    base_doc = yaml.safe_load(DEFAULT_BASE_FIXTURE.read_text(encoding="utf-8"))
    fixture = build_canonical_free_one_disc_fixture(
        base_doc=base_doc,
        case=FreeOneDiscCase(
            "stationarity_unit",
            trace_epsilon=0.02,
            near_spacing=0.02,
            outer_radius=8.0,
        ),
        theta_b=THEORY_THETA_B,
    )
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as stream:
        yaml.safe_dump(fixture, stream, sort_keys=False)
        path = Path(stream.name)
    try:
        context = _build_context(path)
        positions = context.mesh.positions_view().copy()
        tilts_in = context.mesh.tilts_in_view().copy()
        tilts_out = context.mesh.tilts_out_view().copy()

        audit = coupled_stationarity(context)

        assert np.isfinite(audit["combined"]["l2"])
        assert audit["combined"]["l2"] > 0.0
        assert audit["shape"]["l2"] > 0.0
        assert audit["tilt_in"]["l2"] > 0.0
        assert audit["tilt_out"]["l2"] > 0.0
        np.testing.assert_array_equal(context.mesh.positions_view(), positions)
        np.testing.assert_array_equal(context.mesh.tilts_in_view(), tilts_in)
        np.testing.assert_array_equal(context.mesh.tilts_out_view(), tilts_out)
    finally:
        path.unlink(missing_ok=True)


def test_feasible_state_preparation_reduces_tilt_residual_after_projection() -> None:
    base_doc = yaml.safe_load(DEFAULT_BASE_FIXTURE.read_text(encoding="utf-8"))
    fixture = build_canonical_free_one_disc_fixture(
        base_doc=base_doc,
        case=FreeOneDiscCase(
            "feasible_state_unit",
            trace_epsilon=0.02,
            near_spacing=0.02,
            outer_radius=8.0,
        ),
        theta_b=THEORY_THETA_B,
    )
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as stream:
        yaml.safe_dump(fixture, stream, sort_keys=False)
        path = Path(stream.name)
    try:
        context = _build_context(path)
        prepare_feasible_state(context)
        audit = coupled_stationarity(context)

        assert audit["tilt_in"]["l2"] < 0.1
        assert audit["tilt_out"]["l2"] < 0.1
    finally:
        path.unlink(missing_ok=True)


def test_stationarity_audit_does_not_change_later_feasible_preparation() -> None:
    base_doc = yaml.safe_load(DEFAULT_BASE_FIXTURE.read_text(encoding="utf-8"))
    fixture = build_canonical_free_one_disc_fixture(
        base_doc=base_doc,
        case=FreeOneDiscCase(
            "audit_cache_unit",
            trace_epsilon=0.02,
            near_spacing=0.02,
            outer_radius=8.0,
        ),
        theta_b=THEORY_THETA_B,
    )
    paths = []
    contexts = []
    try:
        for _ in range(2):
            with tempfile.NamedTemporaryFile(
                "w", suffix=".yaml", delete=False
            ) as stream:
                yaml.safe_dump(fixture, stream, sort_keys=False)
                path = Path(stream.name)
            paths.append(path)
            contexts.append(_build_context(path))

        coupled_stationarity(contexts[0])
        for context in contexts:
            prepare_feasible_state(context)
        audited = coupled_stationarity(contexts[0])
        control = coupled_stationarity(contexts[1])

        assert audited["energy"] == control["energy"]
        assert audited["combined"]["l2"] == pytest.approx(
            control["combined"]["l2"], rel=1.0e-12, abs=1.0e-12
        )
    finally:
        for path in paths:
            path.unlink(missing_ok=True)
