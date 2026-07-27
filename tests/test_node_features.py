import numpy as np

from graph_generator import Graph_generator
from node_features import (
    NODE_INPUT_DIM,
    UNREACHABLE_GRAPH_DIST,
    UTILITY_OVER_DIST_EPS,
    build_node_inputs,
    compute_node_input_dim,
)
from parameter import (
    INPUT_DIM,
    USE_NODE_GRAPH_DIST_TO_CURRENT,
    USE_NODE_UTILITY_OVER_DIST,
    USE_NODE_VISIT_COUNT,
)


def make_graph_generator():
    graph_generator = Graph_generator(map_size=(40, 40), k_size=3, sensor_range=1)
    graph_generator.node_coords = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [20.0, 0.0],
        ],
        dtype=np.float32,
    )
    return graph_generator


def test_shortest_distances_from_uses_undirected_edges_and_marks_unreachable():
    graph_generator = make_graph_generator()
    graph_generator.graph.add_edge("1", "0", 5.0)
    graph_generator.graph.add_edge("1", "1", 0.0)
    graph_generator.graph.add_edge("2", "2", 0.0)

    distances = graph_generator.shortest_distances_from(0)

    np.testing.assert_allclose(distances[:2], np.array([0.0, 5.0], dtype=np.float32))
    assert np.isinf(distances[2])

    normalized_distances, reachable = graph_generator.get_normalized_shortest_path_distances(0)
    np.testing.assert_allclose(normalized_distances[:2, 0], np.array([0.0, 5.0 / 640], dtype=np.float32))
    assert normalized_distances[2, 0] == UNREACHABLE_GRAPH_DIST
    np.testing.assert_array_equal(reachable[:, 0], np.array([True, True, False]))


def test_parameter_input_dim_matches_enabled_node_features():
    assert USE_NODE_GRAPH_DIST_TO_CURRENT is False
    assert USE_NODE_UTILITY_OVER_DIST is True
    assert USE_NODE_VISIT_COUNT is False
    assert INPUT_DIM == NODE_INPUT_DIM == 5


def test_build_node_inputs_appends_enabled_utility_over_dist_only():
    node_inputs = build_node_inputs(
        node_coords=np.array([[0.0, 0.0], [640.0, 320.0], [10.0, 10.0]], dtype=np.float32),
        node_utility=np.array([50.0, 25.0, 10.0], dtype=np.float32),
        guidepost=np.array([[1.0], [0.0], [0.0]], dtype=np.float32),
        graph_dist_to_current=np.array([0.0, 0.1, UNREACHABLE_GRAPH_DIST], dtype=np.float32),
        reachable_nodes=np.array([True, True, False]),
        visit_count=None,
        current_node_index=0,
    )

    assert node_inputs.shape == (3, NODE_INPUT_DIM)
    assert np.isfinite(node_inputs).all()
    assert node_inputs[0, 4] == 0.0
    np.testing.assert_allclose(node_inputs[1, 4], 0.5 / (0.1 + UTILITY_OVER_DIST_EPS), rtol=1e-6)
    assert node_inputs[2, 4] == 0.0


def test_build_node_inputs_can_reenable_graph_distance_and_visit_count():
    node_inputs = build_node_inputs(
        node_coords=np.array([[0.0, 0.0], [640.0, 320.0], [10.0, 10.0]], dtype=np.float32),
        node_utility=np.array([50.0, 25.0, 10.0], dtype=np.float32),
        guidepost=np.array([[1.0], [0.0], [0.0]], dtype=np.float32),
        graph_dist_to_current=np.array([0.0, 0.1, UNREACHABLE_GRAPH_DIST], dtype=np.float32),
        reachable_nodes=np.array([True, True, False]),
        visit_count=np.array([[1.0], [0.0], [0.0]], dtype=np.float32),
        current_node_index=0,
        use_graph_dist_to_current=True,
        use_utility_over_dist=True,
        use_visit_count=True,
    )

    assert node_inputs.shape == (3, compute_node_input_dim(True, True, True))
    np.testing.assert_allclose(node_inputs[:, 4], np.array([0.0, 0.1, UNREACHABLE_GRAPH_DIST]))
    assert node_inputs[0, 5] == 0.0
    np.testing.assert_allclose(node_inputs[1, 5], 0.5 / (0.1 + UTILITY_OVER_DIST_EPS), rtol=1e-6)
    assert node_inputs[2, 5] == 0.0
    np.testing.assert_array_equal(node_inputs[:, 6], np.array([1.0, 0.0, 0.0], dtype=np.float32))
