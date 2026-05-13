"""Verify that geometry.entities re-exports are correct."""

from geometry.entities import (
    Body,
    Edge,
    Facet,
    GlobalParameters,
    Mesh,
    MeshError,
    Vertex,
    _fast_cross,
)


def test_entities_reexports():
    assert Vertex.__name__ == "Vertex"
    assert Edge.__name__ == "Edge"
    assert Facet.__name__ == "Facet"
    assert Body.__name__ == "Body"
    assert Mesh.__name__ == "Mesh"
    assert MeshError.__name__ == "MeshError"
    assert GlobalParameters.__name__ == "GlobalParameters"
    assert callable(_fast_cross)


def test_entities_modules():
    # Verify they actually come from the new modules
    assert Vertex.__module__ == "geometry.vertex"
    assert Edge.__module__ == "geometry.edge"
    assert Facet.__module__ == "geometry.facet"
    assert Body.__module__ == "geometry.body"
    assert Mesh.__module__ == "geometry.mesh"
    assert MeshError.__module__ == "geometry.mesh"
