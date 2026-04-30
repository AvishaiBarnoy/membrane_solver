"""Unit tests for shared plane operations."""

from __future__ import annotations

import numpy as np

from geometry.plane_ops import (
    fit_plane_normal,
    order_by_angle,
    orthonormal_basis_from_normal,
)


def test_orthonormal_basis_standard_z():
    normal = np.array([0.0, 0.0, 1.0])
    u, v = orthonormal_basis_from_normal(normal)

    # Check unit length
    assert np.allclose(np.linalg.norm(u), 1.0)
    assert np.allclose(np.linalg.norm(v), 1.0)

    # Check orthogonality
    assert abs(np.dot(u, normal)) < 1e-15
    assert abs(np.dot(v, normal)) < 1e-15
    assert abs(np.dot(u, v)) < 1e-15

    # Check right-handedness: u x v = normal
    assert np.allclose(np.cross(u, v), normal)


def test_orthonormal_basis_standard_x():
    normal = np.array([1.0, 0.0, 0.0])
    u, v = orthonormal_basis_from_normal(normal)

    assert np.allclose(np.linalg.norm(u), 1.0)
    assert np.allclose(np.linalg.norm(v), 1.0)
    assert abs(np.dot(u, normal)) < 1e-15
    assert abs(np.dot(v, normal)) < 1e-15
    assert abs(np.dot(u, v)) < 1e-15
    assert np.allclose(np.cross(u, v), normal)


def test_fit_plane_normal_xy_plane():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    normal = fit_plane_normal(points)
    assert normal is not None
    # PCA normal could be [0,0,1] or [0,0,-1]
    assert np.allclose(abs(normal), [0.0, 0.0, 1.0])


def test_fit_plane_normal_tilted():
    # Points on x + z = 1 plane => normal proportional to [1, 0, 1]
    points = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ]
    )
    normal = fit_plane_normal(points)
    assert normal is not None
    expected = np.array([1.0, 0.0, 1.0]) / np.sqrt(2.0)
    assert np.allclose(abs(normal), abs(expected))


def test_fit_plane_normal_too_few_points():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    assert fit_plane_normal(points) is None


def test_order_by_angle_simple():
    center = np.array([0.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    # Points at 0, 90, 180, 270 degrees
    positions = np.array(
        [
            [0.0, 1.0, 0.0],  # 90 deg
            [1.0, 0.0, 0.0],  # 0 deg
            [0.0, -1.0, 0.0],  # 270 (-90) deg
            [-1.0, 0.0, 0.0],  # 180 deg
        ]
    )
    # arctan2(y, x) -> [pi/2, 0, -pi/2, pi]
    # sorted: -pi/2, 0, pi/2, pi => index order [2, 1, 0, 3]
    indices = order_by_angle(positions, center=center, normal=normal)
    assert np.array_equal(indices, [2, 1, 0, 3])


def test_order_by_angle_tilted():
    # Normal [1, 1, 1] normalized
    normal = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    center = np.array([0.0, 0.0, 0.0])

    # Get basis
    u, v = orthonormal_basis_from_normal(normal)

    # Create points at known angles in this basis
    angles = np.array([0.5, -1.2, 2.3, 0.1])
    positions = np.zeros((4, 3))
    for i, ang in enumerate(angles):
        positions[i] = np.cos(ang) * u + np.sin(ang) * v

    indices = order_by_angle(positions, center=center, normal=normal)
    expected = np.argsort(angles)
    assert np.array_equal(indices, expected)
