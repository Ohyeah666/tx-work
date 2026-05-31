import numpy as np

import graph_generator as graph_generator_module
from graph_generator import Graph_generator


class FakeNode:
    observable_frontiers = []


def test_expected_unknown_gain_local_cache_recomputes_local_and_new_nodes(monkeypatch):
    calls = []

    def fake_compute_expected_unknown_gain(node_coords, robot_belief, sensor_range, ray_sample_count=0):
        calls.append(node_coords.copy())
        base = 0 if len(calls) == 1 else 100
        return (node_coords[:, 0].reshape(-1, 1) + base).astype(float)

    monkeypatch.setattr(
        graph_generator_module,
        "compute_expected_unknown_gain_for_nodes",
        fake_compute_expected_unknown_gain,
    )

    generator = Graph_generator(
        map_size=(120, 120),
        k_size=2,
        sensor_range=10,
        expected_unknown_gain_update_mode="local",
        expected_unknown_gain_local_radius_factor=2.0,
    )
    generator.node_coords = np.array([[0.0, 0.0], [5.0, 0.0], [25.0, 0.0], [100.0, 0.0]])
    generator.nodes_list = [FakeNode() for _ in range(4)]

    robot_belief = np.zeros((120, 120), dtype=int)
    frontiers = np.empty((0, 2))
    generator.update_semantic_features(robot_belief, frontiers)

    np.testing.assert_array_equal(calls[0], generator.node_coords)
    np.testing.assert_allclose(generator.node_expected_unknown_gain_raw[:, 0], [0, 5, 25, 100])
    assert generator.last_expected_unknown_gain_recompute_count == 4

    generator.node_coords = np.vstack([generator.node_coords, [50.0, 0.0]])
    generator.nodes_list.append(FakeNode())
    generator.update_semantic_features(
        robot_belief,
        frontiers,
        robot_position=np.array([0.0, 0.0]),
        old_node_count=4,
    )

    np.testing.assert_array_equal(calls[1], np.array([[0.0, 0.0], [5.0, 0.0], [50.0, 0.0]]))
    np.testing.assert_allclose(generator.node_expected_unknown_gain_raw[:, 0], [100, 105, 25, 100, 150])
    assert generator.last_expected_unknown_gain_recompute_count == 3


def test_expected_unknown_gain_full_mode_recomputes_all_nodes(monkeypatch):
    calls = []

    def fake_compute_expected_unknown_gain(node_coords, robot_belief, sensor_range, ray_sample_count=0):
        calls.append(node_coords.copy())
        return np.zeros((len(node_coords), 1), dtype=float)

    monkeypatch.setattr(
        graph_generator_module,
        "compute_expected_unknown_gain_for_nodes",
        fake_compute_expected_unknown_gain,
    )

    generator = Graph_generator(
        map_size=(120, 120),
        k_size=2,
        sensor_range=10,
        expected_unknown_gain_update_mode="full",
    )
    generator.node_coords = np.array([[0.0, 0.0], [5.0, 0.0], [25.0, 0.0]])
    generator.nodes_list = [FakeNode() for _ in range(3)]

    robot_belief = np.zeros((120, 120), dtype=int)
    frontiers = np.empty((0, 2))
    generator.update_semantic_features(robot_belief, frontiers)

    generator.node_coords = np.vstack([generator.node_coords, [50.0, 0.0]])
    generator.nodes_list.append(FakeNode())
    generator.update_semantic_features(
        robot_belief,
        frontiers,
        robot_position=np.array([0.0, 0.0]),
        old_node_count=3,
    )

    np.testing.assert_array_equal(calls[1], generator.node_coords)
    assert generator.last_expected_unknown_gain_recompute_count == 4
