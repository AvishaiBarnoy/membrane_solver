from __future__ import annotations

from pathlib import Path

import yaml

from tools.build_stage_a_fixtures import (
    REMOVED_GLOBAL_KEYS,
    SOURCE_FIXTURE,
    STAGE_A_RIM_SOURCE_STRENGTH,
    STAGE_A_SEED_Z,
    build_stage_a_fixture_docs,
)


def _face_count(doc: dict) -> int:
    faces = doc.get("faces") or doc.get("Faces") or doc.get("Facets") or []
    return int(len(faces))


def test_stage_a_fixture_builder_preserves_geometry_and_remaps_physics() -> None:
    source_doc = yaml.safe_load(Path(SOURCE_FIXTURE).read_text(encoding="utf-8"))
    base_doc, seeded_doc = build_stage_a_fixture_docs(source_doc)

    assert len(base_doc["vertices"]) == len(source_doc["vertices"])
    assert len(base_doc["edges"]) == len(source_doc["edges"])
    assert _face_count(base_doc) == _face_count(source_doc)

    gp = dict(base_doc["global_parameters"])
    for key in REMOVED_GLOBAL_KEYS:
        assert key not in gp

    assert gp["tilt_rim_source_group"] == "disk"
    assert float(gp["tilt_rim_source_strength"]) == STAGE_A_RIM_SOURCE_STRENGTH
    assert gp["tilt_rim_source_edge_mode"] == "all"

    assert "tilt_thetaB_contact_in" not in base_doc["energy_modules"]
    assert "tilt_rim_source_bilayer" in base_doc["energy_modules"]
    assert "tilt_thetaB_boundary_in" not in base_doc["constraint_modules"]
    assert "rim_slope_match_out" in base_doc["constraint_modules"]

    base_rim_z = []
    seeded_rim_z = []
    for base_vertex, seeded_vertex in zip(base_doc["vertices"], seeded_doc["vertices"]):
        base_opts = (
            base_vertex[3]
            if len(base_vertex) > 3 and isinstance(base_vertex[3], dict)
            else {}
        )
        seeded_opts = (
            seeded_vertex[3]
            if len(seeded_vertex) > 3 and isinstance(seeded_vertex[3], dict)
            else {}
        )
        if str(base_opts.get("preset") or "") == "rim":
            base_rim_z.append(float(base_vertex[2]))
        if str(seeded_opts.get("preset") or "") == "rim":
            seeded_rim_z.append(float(seeded_vertex[2]))

    assert base_rim_z
    assert seeded_rim_z
    assert all(abs(z) < 1.0e-15 for z in base_rim_z)
    assert all(z == STAGE_A_SEED_Z for z in seeded_rim_z)
