import numpy as np

from observation_features import FEATURE_ORDER_V1, build_node_inputs


def test_feature_order_appends_v1_features_after_existing_seven_features():
    assert FEATURE_ORDER_V1 == (
        "x",
        "y",
        "utility",
        "guidepost",
        "graph_dist_to_current",
        "utility_over_dist",
        "visit_count",
        "expected_unknown_gain",
        "frontier_cluster_size",
    )


def test_build_node_inputs_preserves_existing_feature_order_and_appends_new_features():
    node_coords = np.array([[320, 160], [640, 0]], dtype=float)
    node_utility = np.array([25, 50], dtype=float)
    guidepost = np.array([[1], [0]], dtype=float)
    graph_dist_to_current = np.array([[0], [2]], dtype=float)
    reachable_nodes = np.array([[True], [False]])
    visit_count = np.array([[3], [0]], dtype=float)
    expected_unknown_gain = np.array([[0.2], [0.4]], dtype=float)
    frontier_cluster_size = np.array([[0.6], [0.8]], dtype=float)

    node_inputs = build_node_inputs(
        node_coords,
        node_utility,
        guidepost,
        graph_dist_to_current,
        reachable_nodes,
        visit_count,
        expected_unknown_gain,
        frontier_cluster_size,
        current_node_index=0,
    )

    assert node_inputs.shape == (2, 9)
    np.testing.assert_allclose(
        node_inputs[0],
        np.array([0.5, 0.25, 0.5, 1.0, 0.0, 0.0, 3.0, 0.2, 0.6]),
    )
    np.testing.assert_allclose(
        node_inputs[1],
        np.array([1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 0.4, 0.8]),
    )


def test_build_node_inputs_produces_finite_values_when_distance_is_zero():
    node_inputs = build_node_inputs(
        node_coords=np.array([[0, 0]], dtype=float),
        node_utility=np.array([50], dtype=float),
        guidepost=np.array([[1]], dtype=float),
        graph_dist_to_current=np.array([[0]], dtype=float),
        reachable_nodes=np.array([[True]]),
        visit_count=np.array([[1]], dtype=float),
        expected_unknown_gain=np.array([[0]], dtype=float),
        frontier_cluster_size=np.array([[0]], dtype=float),
        current_node_index=0,
    )

    assert np.isfinite(node_inputs).all()
    assert node_inputs[0, 5] == 0
