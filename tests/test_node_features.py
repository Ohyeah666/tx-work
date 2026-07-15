import numpy as np

from graph_generator import Graph_generator
from node_features import (
    ACTION_FEATURE_BRANCH_GAIN,
    ACTION_FEATURE_BRANCH_MEMORY,
    ACTION_FEATURE_BRANCH_UTILITY,
    ACTION_FEATURE_EDGE_DIST,
    ACTION_FEATURE_IMMEDIATE_REVERSE,
    ACTION_FEATURE_NEXT_NODE_MEMORY,
    BRANCH_GAIN_FEATURE,
    BRANCH_MEMORY_FEATURE,
    BRANCH_UTILITY_FEATURE,
    EDGE_DIST_FEATURE,
    FeatureConfig,
    IMMEDIATE_REVERSE_FEATURE,
    NEXT_NODE_MEMORY_FEATURE,
    build_node_and_action_features,
    get_action_feature_index,
)
from parameter import ACTION_FEATURE_DIM, INPUT_DIM, PADDING_NODE_INDEX


class FakeEnv:
    def __init__(self):
        self.graph_generator = Graph_generator(map_size=(10, 10), k_size=5, sensor_range=80)
        self.node_coords = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ])
        self.graph_generator.node_coords = self.node_coords
        self.graph = self.graph_generator.graph
        for node in range(3):
            self.graph.add_node(str(node))

        self.graph.add_edge('0', '0', 0.0)
        self.graph.add_edge('0', '1', 1.0)
        self.graph.add_edge('1', '1', 0.0)
        self.graph.add_edge('1', '0', 1.0)
        self.graph.add_edge('1', '2', 1.0)
        self.graph.add_edge('2', '2', 0.0)
        self.graph.add_edge('2', '1', 1.0)

        self.node_utility = np.array([0.0, 0.0, 25.0])
        self.guidepost = np.array([[1.0], [1.0], [0.0]])
        self.visit_count = np.array([[1.0], [1.0], [0.0]])
        self.graph_generator.route_node = [self.node_coords[0], self.node_coords[1]]

    def find_index_from_coords(self, position):
        return int(np.argmin(np.linalg.norm(self.node_coords - position, axis=1)))


def test_build_node_and_action_features_outputs_default_edge_distance_and_padding():
    env = FakeEnv()

    features = build_node_and_action_features(env, env.node_coords[1], k_size=5)

    assert features.node_inputs.shape == (3, INPUT_DIM)
    assert features.action_inputs.shape == (5, ACTION_FEATURE_DIM)
    assert ACTION_FEATURE_DIM == 1
    assert EDGE_DIST_FEATURE == 0
    assert features.current_index == 1
    assert features.edge_inputs.tolist() == [1, 0, 2, PADDING_NODE_INDEX, PADDING_NODE_INDEX]
    assert features.edge_padding_mask.tolist() == [0, 0, 0, 1, 1]
    assert np.isfinite(features.node_inputs).all()
    assert np.isfinite(features.action_inputs).all()
    np.testing.assert_allclose(features.action_inputs[3:], 0.0)
    np.testing.assert_allclose(
        features.action_inputs[:, EDGE_DIST_FEATURE],
        [0.0, 1 / 640, 1 / 640, 0.0, 0.0],
    )


def test_full_action_feature_config_computes_reverse_memory_and_branch_signals():
    env = FakeEnv()
    config = FeatureConfig(
        use_action_features=True,
        use_action_feature_edge_dist=True,
        use_action_feature_immediate_reverse=True,
        use_action_feature_next_node_memory=True,
        use_directional_branch_features=True,
        use_action_feature_branch_utility=True,
        use_action_feature_branch_gain=True,
        use_action_feature_branch_memory=True,
    )

    features = build_node_and_action_features(env, env.node_coords[1], k_size=5, config=config)

    assert features.action_inputs.shape == (5, 6)
    edge_dist_idx = get_action_feature_index(ACTION_FEATURE_EDGE_DIST, config)
    reverse_idx = get_action_feature_index(ACTION_FEATURE_IMMEDIATE_REVERSE, config)
    next_memory_idx = get_action_feature_index(ACTION_FEATURE_NEXT_NODE_MEMORY, config)
    branch_utility_idx = get_action_feature_index(ACTION_FEATURE_BRANCH_UTILITY, config)
    branch_gain_idx = get_action_feature_index(ACTION_FEATURE_BRANCH_GAIN, config)
    branch_memory_idx = get_action_feature_index(ACTION_FEATURE_BRANCH_MEMORY, config)
    assert (
        edge_dist_idx,
        reverse_idx,
        next_memory_idx,
        branch_utility_idx,
        branch_gain_idx,
        branch_memory_idx,
    ) == (0, 1, 2, 3, 4, 5)

    reverse_slot = 1
    forward_slot = 2
    assert features.action_inputs[reverse_slot, reverse_idx] == 1
    assert features.action_inputs[forward_slot, reverse_idx] == 0
    np.testing.assert_allclose(features.action_inputs[forward_slot, edge_dist_idx], 1 / 640)
    assert features.action_inputs[forward_slot, next_memory_idx] > 0
    np.testing.assert_allclose(features.action_inputs[forward_slot, branch_utility_idx], 0.5)
    np.testing.assert_allclose(features.action_inputs[forward_slot, branch_gain_idx], 1.0)
    assert 0 < features.action_inputs[forward_slot, branch_memory_idx] < 1


def test_action_feature_switches_remap_columns():
    env = FakeEnv()
    config = FeatureConfig(
        use_action_feature_edge_dist=False,
        use_action_feature_immediate_reverse=True,
        use_action_feature_next_node_memory=False,
        use_directional_branch_features=True,
        use_action_feature_branch_utility=False,
        use_action_feature_branch_gain=True,
        use_action_feature_branch_memory=True,
    )

    features = build_node_and_action_features(env, env.node_coords[1], k_size=5, config=config)

    assert features.action_inputs.shape == (5, 3)
    assert get_action_feature_index(ACTION_FEATURE_EDGE_DIST, config) is None
    assert get_action_feature_index(ACTION_FEATURE_NEXT_NODE_MEMORY, config) is None
    assert get_action_feature_index(ACTION_FEATURE_BRANCH_UTILITY, config) is None

    reverse_idx = get_action_feature_index(ACTION_FEATURE_IMMEDIATE_REVERSE, config)
    gain_idx = get_action_feature_index(ACTION_FEATURE_BRANCH_GAIN, config)
    memory_idx = get_action_feature_index(ACTION_FEATURE_BRANCH_MEMORY, config)
    assert (reverse_idx, gain_idx, memory_idx) == (0, 1, 2)

    reverse_slot = 1
    forward_slot = 2
    assert features.action_inputs[reverse_slot, reverse_idx] == 1
    assert features.action_inputs[forward_slot, reverse_idx] == 0
    np.testing.assert_allclose(features.action_inputs[forward_slot, gain_idx], 1.0)
    assert 0 < features.action_inputs[forward_slot, memory_idx] < 1


def test_disabling_all_action_features_returns_zero_width_inputs():
    env = FakeEnv()
    config = FeatureConfig(use_action_features=False)

    features = build_node_and_action_features(env, env.node_coords[1], k_size=5, config=config)

    assert features.action_inputs.shape == (5, 0)
    assert get_action_feature_index(ACTION_FEATURE_IMMEDIATE_REVERSE, config) is None


def test_edge_distance_feature_is_clipped_by_configured_max_norm():
    env = FakeEnv()
    config = FeatureConfig(distance_attention_max_norm=0.001)

    features = build_node_and_action_features(env, env.node_coords[1], k_size=5, config=config)

    np.testing.assert_allclose(features.action_inputs[1, EDGE_DIST_FEATURE], 0.001)
    np.testing.assert_allclose(features.action_inputs[2, EDGE_DIST_FEATURE], 0.001)
