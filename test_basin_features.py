import math

import numpy as np

from basin_features import BASIN_ACTION_FEATURE_ORDER, build_basin_action_features
from graph import Graph
from graph_generator import Graph_generator


def make_graph_generator_with_edges(edges):
    graph_generator = Graph_generator(map_size=(10, 10), k_size=4, sensor_range=2)
    graph_generator.GRAPH_DISTANCE_NORMALIZER = 1
    graph_generator.node_coords = np.array(
        [
            [0, 0],
            [1, 0],
            [0, 1],
            [2, 0],
            [0, 2],
        ],
        dtype=float,
    )
    graph_generator.graph = Graph()
    for from_node, to_node, length in edges:
        graph_generator.graph.add_node(str(from_node))
        graph_generator.graph.add_edge(str(from_node), str(to_node), length)
    return graph_generator


def test_basin_action_feature_order_is_basin5():
    assert BASIN_ACTION_FEATURE_ORDER == (
        "basin_utility_sum",
        "basin_expected_unknown_gain_sum",
        "basin_frontier_cluster_max",
        "basin_unvisited_ratio",
        "basin_min_dist_to_utility",
    )


def test_shortest_path_first_hop_uses_neighbor_index_tie_break():
    graph_generator = make_graph_generator_with_edges(
        [
            (0, 1, 1),
            (0, 2, 1),
            (1, 3, 1),
            (2, 3, 1),
            (2, 4, 1),
        ]
    )

    distances, reachable, first_hop = graph_generator.get_normalized_shortest_path_distances(
        0,
        return_first_hop=True,
    )

    np.testing.assert_allclose(distances.reshape(-1), np.array([0, 1, 1, 2, 2], dtype=float))
    np.testing.assert_array_equal(reachable.reshape(-1), np.array([True, True, True, True, True]))
    np.testing.assert_array_equal(first_hop.reshape(-1), np.array([0, 1, 2, 1, 2]))


def test_build_basin_action_features_aggregates_by_first_hop_and_zeroes_padding():
    edge_indices = np.array([0, 1, 2, 0])
    edge_padding_mask = np.array([False, False, False, True])
    first_hop = np.array([0, 1, 2, 1, 2])
    distances = np.array([[0], [1], [1], [2], [2]], dtype=float)
    node_utility = np.array([0, 0, 0, 10, 0], dtype=float)
    visit_count = np.array([[1], [0], [0], [0], [2]], dtype=float)
    expected_unknown_gain = np.array([[0], [0.2], [0.3], [0.4], [0.5]], dtype=float)
    frontier_cluster_size = np.array([[0], [0.1], [0.2], [0.7], [0.4]], dtype=float)

    features = build_basin_action_features(
        edge_indices,
        first_hop,
        distances,
        node_utility,
        visit_count,
        expected_unknown_gain,
        frontier_cluster_size,
        current_node_index=0,
        edge_padding_mask=edge_padding_mask,
        k_size=4,
        utility_sum_normalizer=100,
        expected_unknown_gain_sum_normalizer=10,
    )

    assert features.shape == (4, 5)
    np.testing.assert_allclose(features[0], np.zeros(5))
    np.testing.assert_allclose(features[3], np.zeros(5))
    np.testing.assert_allclose(
        features[1],
        np.array([
            math.log1p(10) / math.log1p(100),
            math.log1p(0.6) / math.log1p(10),
            0.7,
            1.0,
            2.0,
        ]),
    )
    np.testing.assert_allclose(
        features[2],
        np.array([
            0.0,
            math.log1p(0.8) / math.log1p(10),
            0.4,
            0.5,
            2.0,
        ]),
    )


def test_build_basin_action_features_uses_fixed_distance_when_basin_has_no_utility():
    features = build_basin_action_features(
        edge_indices=np.array([1]),
        first_hop=np.array([0, 1, 1]),
        graph_dist_to_current=np.array([[0], [1], [2]], dtype=float),
        node_utility=np.array([0, 0, 0], dtype=float),
        visit_count=np.array([[1], [0], [0]], dtype=float),
        expected_unknown_gain=np.array([[0], [0.1], [0.2]], dtype=float),
        frontier_cluster_size=np.array([[0], [0.3], [0.4]], dtype=float),
        current_node_index=0,
        edge_padding_mask=np.array([False]),
        k_size=1,
        no_utility_distance=2.0,
    )

    assert features[0, 4] == 2.0
    assert np.isfinite(features).all()

