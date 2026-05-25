import math

import numpy as np

from feature_extractor import (
    OCCUPIED,
    UNKNOWN,
    compute_expected_unknown_gain_for_nodes,
    compute_frontier_cluster_lookup,
    compute_frontier_cluster_size_for_nodes,
    line_has_occupied,
    normalize_expected_unknown_gain,
    normalize_frontier_cluster_size,
)


class FakeNode:
    def __init__(self, observable_frontiers):
        self.observable_frontiers = observable_frontiers


def test_line_has_occupied_treats_only_occupied_as_blocking():
    belief = np.ones((5, 5)) * 255
    belief[2, 2] = UNKNOWN

    assert not line_has_occupied((0, 2), (4, 2), belief)

    belief[2, 3] = OCCUPIED

    assert line_has_occupied((0, 2), (4, 2), belief)


def test_expected_unknown_gain_counts_visible_unknown_and_stops_at_occupied():
    belief = np.ones((7, 7)) * 255
    belief[3, 2] = UNKNOWN
    belief[3, 3] = OCCUPIED
    belief[3, 4] = UNKNOWN

    gain = compute_expected_unknown_gain_for_nodes(
        np.array([[1, 3]]),
        belief,
        sensor_range=5,
    )

    assert gain.shape == (1, 1)
    assert gain[0, 0] == 1


def test_expected_unknown_gain_does_not_let_unknown_block_unknown():
    belief = np.ones((7, 7)) * 255
    belief[3, 2] = UNKNOWN
    belief[3, 3] = UNKNOWN

    gain = compute_expected_unknown_gain_for_nodes(
        np.array([[1, 3]]),
        belief,
        sensor_range=5,
    )

    assert gain[0, 0] == 2


def test_expected_unknown_gain_normalization_uses_sensor_disc_area_and_clips():
    raw_gain = np.array([[math.pi * 25], [math.pi * 50]])

    normalized = normalize_expected_unknown_gain(raw_gain, sensor_range=5)

    np.testing.assert_allclose(normalized[0, 0], 1.0)
    np.testing.assert_allclose(normalized[1, 0], 1.0)


def test_frontier_cluster_lookup_uses_eight_neighborhood_with_resolution():
    frontiers = np.array([
        [0, 0],
        [4, 0],
        [20, 20],
        [24, 24],
        [60, 0],
    ])

    lookup = compute_frontier_cluster_lookup(frontiers, resolution=4, connectivity=8)

    assert lookup[(0, 0)] == 2
    assert lookup[(4, 0)] == 2
    assert lookup[(20, 20)] == 2
    assert lookup[(24, 24)] == 2
    assert lookup[(60, 0)] == 1


def test_frontier_cluster_size_for_nodes_uses_max_visible_cluster_size():
    lookup = {
        (0, 0): 2,
        (4, 0): 2,
        (20, 20): 5,
    }
    nodes = [
        FakeNode([np.array([0, 0]), np.array([20, 20])]),
        FakeNode([np.array([4, 0])]),
        FakeNode([]),
    ]

    cluster_sizes = compute_frontier_cluster_size_for_nodes(nodes, lookup)

    np.testing.assert_array_equal(cluster_sizes, np.array([[5], [2], [0]]))


def test_frontier_cluster_size_normalization_uses_fixed_scale_and_clips():
    raw_cluster_size = np.array([[25], [50], [100]])

    normalized = normalize_frontier_cluster_size(raw_cluster_size, normalizer=50)

    np.testing.assert_allclose(normalized, np.array([[0.5], [1.0], [1.0]]))
