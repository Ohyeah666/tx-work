from types import SimpleNamespace

import numpy as np
import torch

from replay_schema import BUFFER_SIZE, MAP_INPUTS, NEXT_MAP_INPUTS
from test_worker import TestWorker
from worker import Worker
from node_features import NODE_INPUT_DIM, UTILITY_OVER_DIST_EPS


class FakeEnv(SimpleNamespace):
    def find_index_from_coords(self, position):
        distances = np.linalg.norm(self.node_coords - position, axis=1)
        return int(np.argmin(distances))


class FakeGraphGenerator:
    def get_normalized_shortest_path_distances(self, start_index):
        assert start_index == 0
        distances = np.array([0, 4 / 640, 8 / 640], dtype=np.float32).reshape(3, 1)
        reachable = np.array([True, True, True]).reshape(3, 1)
        return distances, reachable


def make_fake_env():
    return FakeEnv(
        node_coords=np.array([[0, 0], [4, 0], [8, 0]], dtype=np.float32),
        graph={
            "0": {"0": object(), "1": object(), "2": object()},
            "1": {"1": object(), "0": object(), "2": object()},
            "2": {"2": object(), "1": object(), "0": object()},
        },
        node_utility=np.array([5, 3, 1], dtype=np.float32),
        guidepost=np.array([[1], [0], [0]], dtype=np.float32),
        downsampled_belief=np.array(
            [
                [255, 127, 1, 255],
                [127, 255, 127, 1],
                [1, 127, 255, 255],
            ],
            dtype=np.uint8,
        ),
        frontiers=np.array([[4, 4]], dtype=np.float32),
        robot_belief=np.zeros((12, 16), dtype=np.uint8),
        resolution=4,
        graph_generator=FakeGraphGenerator(),
    )


def make_worker(worker_cls, node_padding_size=None):
    worker = worker_cls.__new__(worker_cls)
    worker.device = torch.device("cpu")
    worker.k_size = 4
    worker.robot_position = np.array([0, 0], dtype=np.float32)
    worker.env = make_fake_env()
    if node_padding_size is not None:
        worker.node_padding_size = node_padding_size
        worker.episode_buffer = [[] for _ in range(BUFFER_SIZE)]
    return worker


def test_worker_observation_includes_semantic_map_and_replay_slots():
    worker = make_worker(Worker, node_padding_size=6)

    observations = worker.get_observations()
    worker.save_observations(observations)
    worker.save_next_observations(observations)

    assert len(observations) == 7
    node_inputs, edge_inputs, _, node_padding_mask, edge_padding_mask, edge_mask, map_inputs = observations
    assert node_inputs.shape == (1, 6, NODE_INPUT_DIM)
    assert edge_inputs.shape == (1, 1, 4)
    assert node_padding_mask.shape == (1, 1, 6)
    assert edge_padding_mask.shape == (1, 1, 4)
    assert edge_mask.shape == (1, 6, 6)
    assert map_inputs.shape == (1, 5, 3, 4)
    assert map_inputs.dtype == torch.uint8
    assert len(worker.episode_buffer) == BUFFER_SIZE
    assert len(worker.episode_buffer[MAP_INPUTS]) == 1
    assert len(worker.episode_buffer[NEXT_MAP_INPUTS]) == 1
    assert node_inputs[0, 0, 4].item() == 0.0
    np.testing.assert_allclose(
        node_inputs[0, 1, 4].item(),
        (3 / 50) / (4 / 640 + UTILITY_OVER_DIST_EPS),
        rtol=1e-6,
    )


def test_test_worker_observation_matches_training_tuple_without_node_padding():
    test_worker = make_worker(TestWorker)

    observations = test_worker.get_observations()

    assert len(observations) == 7
    node_inputs, edge_inputs, _, node_padding_mask, edge_padding_mask, edge_mask, map_inputs = observations
    assert node_inputs.shape == (1, 3, NODE_INPUT_DIM)
    assert edge_inputs.shape == (1, 1, 4)
    assert node_padding_mask is None
    assert edge_padding_mask.shape == (1, 1, 4)
    assert edge_mask.shape == (1, 3, 3)
    assert map_inputs.shape == (1, 5, 3, 4)
    np.testing.assert_allclose(
        node_inputs[0, 2, 4].item(),
        (1 / 50) / (8 / 640 + UTILITY_OVER_DIST_EPS),
        rtol=1e-6,
    )
