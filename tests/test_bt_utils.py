import numpy as np

from modules.energy.bt_utils import _mean_reconstructed_field


def test_mean_reconstructed_field_purity():
    # 4 vertices, 2 dims
    field = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    # 2 triangles, 2 edges each (simplified)
    endpoint_rows = np.array([[0, 1], [1, 2]])
    recon_idx = np.array([[[2, 0], [0, 0]], [[3, 2], [0, 0]]])
    recon_count = np.array([[1, 0], [2, 0]])

    field_copy = field.copy()

    # Call the function
    out = _mean_reconstructed_field(field, endpoint_rows, recon_idx, recon_count)

    # Check no mutation
    np.testing.assert_array_equal(field, field_copy)

    # Check shape: (n_tri, n_edges, n_dims) = (2, 2, 2)
    assert out.shape == (2, 2, 2)

    # Tri 0, Edge 0: recon_count=1 -> use recon_idx[0,0,0] = 2
    np.testing.assert_allclose(out[0, 0], field[2])

    # Tri 1, Edge 0: recon_count=2 -> use 0.5*(field[3] + field[2])
    np.testing.assert_allclose(out[1, 0], 0.5 * (field[3] + field[2]))


def test_mean_reconstructed_field_logic():
    # 4 vertices, 2 dims
    field = np.array(
        [
            [0.0, 0.0],  # 0
            [1.0, 1.0],  # 1
            [10.0, 10.0],  # 2
            [20.0, 20.0],  # 3
        ]
    )

    # 2 triangles, 3 edges each
    # For one triangle, let's say we have 3 edges.
    # But _mean_reconstructed_field is usually called per triangle-side.
    # In _apply_edge_reconstructed_beltrami_laplacian:
    # edge_endpoint_a is (n_tri, 3), recon_a_idx is (n_tri, 3, 2), recon_a_count is (n_tri, 3)

    endpoint_rows = np.array([[0, 1, 2]])  # (1, 3)

    recon_idx = np.zeros((1, 3, 2), dtype=int)
    recon_count = np.zeros((1, 3), dtype=int)

    # Edge 0 (v1-v2): recon_count=1, use v3=3
    recon_count[0, 0] = 1
    recon_idx[0, 0, 0] = 3

    # Edge 1 (v2-v0): recon_count=2, use v0=0 and v1=1
    recon_count[0, 1] = 2
    recon_idx[0, 1, 0] = 0
    recon_idx[0, 1, 1] = 1

    # Edge 2 (v0-v1): recon_count=0, fallback to field[endpoint_rows]
    recon_count[0, 2] = 0

    out = _mean_reconstructed_field(field, endpoint_rows, recon_idx, recon_count)

    # Edge 0: recon_count=1 -> out[0,0] == field[3]
    np.testing.assert_allclose(out[0, 0], [20.0, 20.0])

    # Edge 1: recon_count=2 -> out[0,1] == 0.5*(field[0]+field[1])
    np.testing.assert_allclose(out[0, 1], [0.5, 0.5])

    # Edge 2: recon_count=0 -> out[0,2] == field[endpoint_rows[0,2]] == field[2]
    np.testing.assert_allclose(out[0, 2], [10.0, 10.0])
