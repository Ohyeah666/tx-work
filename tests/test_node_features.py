import numpy as np

from graph_generator import Graph_generator
from node_features import (
    BRANCH_MEMORY_FEATURE,
    BRANCH_UTILITY_FEATURE,
    EDGE_DIST_FEATURE,
    IMMEDIATE_REVERSE_FEATURE,
    NEXT_NODE_MEMORY_FEATURE,
    build_node_and_action_features,
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


def test_build_node_and_action_features_shapes_padding_and_reverse_signal():
    env = FakeEnv()

    features = build_node_and_action_features(env, env.node_coords[1], k_size=5)

    assert features.node_inputs.shape == (3, INPUT_DIM)
    assert features.action_inputs.shape == (5, ACTION_FEATURE_DIM)
    assert features.current_index == 1
    assert features.edge_inputs.tolist() == [1, 0, 2, PADDING_NODE_INDEX, PADDING_NODE_INDEX]
    assert features.edge_padding_mask.tolist() == [0, 0, 0, 1, 1]
    assert np.isfinite(features.node_inputs).all()
    assert np.isfinite(features.action_inputs).all()
    np.testing.assert_allclose(features.action_inputs[3:], 0.0)

    reverse_slot = 1
    forward_slot = 2
    assert features.action_inputs[reverse_slot, IMMEDIATE_REVERSE_FEATURE] == 1
    assert features.action_inputs[forward_slot, IMMEDIATE_REVERSE_FEATURE] == 0
    np.testing.assert_allclose(features.action_inputs[forward_slot, EDGE_DIST_FEATURE], 1 / 640)
    assert features.action_inputs[forward_slot, NEXT_NODE_MEMORY_FEATURE] > 0
    np.testing.assert_allclose(features.action_inputs[forward_slot, BRANCH_UTILITY_FEATURE], 0.5)
    assert 0 < features.action_inputs[forward_slot, BRANCH_MEMORY_FEATURE] < 1
