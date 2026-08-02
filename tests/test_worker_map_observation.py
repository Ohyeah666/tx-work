from types import SimpleNamespace

import numpy as np
import torch

from parameter import ACTION_FEATURE_DIM, PADDING_NODE_INDEX
from replay_schema import ACTION_INPUTS, BUFFER_SIZE, MAP_INPUTS, NEXT_ACTION_INPUTS, NEXT_MAP_INPUTS
from test_worker import TestWorker
from worker import Worker


class FakeEnv(SimpleNamespace):
    def find_index_from_coords(self, position):
        distances = np.linalg.norm(self.node_coords - position, axis=1)
        return int(np.argmin(distances))


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


def test_worker_observation_disables_semantic_map_and_keeps_action_feature_slots():
    worker = make_worker(Worker, node_padding_size=6)

    observations = worker.get_observations()
    worker.save_observations(observations)
    worker.save_next_observations(observations)

    assert len(observations) == 8
    node_inputs, edge_inputs, _, node_padding_mask, edge_padding_mask, edge_mask, map_inputs, action_inputs = observations
    assert node_inputs.shape == (1, 6, 4)
    assert edge_inputs.shape == (1, 1, 4)
    assert edge_inputs.tolist() == [[[0, 1, 2, PADDING_NODE_INDEX]]]
    assert node_padding_mask.shape == (1, 1, 6)
    assert edge_padding_mask.shape == (1, 1, 4)
    assert edge_padding_mask.tolist() == [[[0, 0, 0, 1]]]
    assert edge_mask.shape == (1, 6, 6)
    assert map_inputs is None
    assert action_inputs.shape == (1, 4, ACTION_FEATURE_DIM)
    torch.testing.assert_close(
        action_inputs[0, :, 0],
        torch.tensor([0.0, 4.0 / 640.0, 8.0 / 640.0, 0.0]),
    )
    assert len(worker.episode_buffer) == BUFFER_SIZE
    assert worker.episode_buffer[MAP_INPUTS] == [None]
    assert worker.episode_buffer[NEXT_MAP_INPUTS] == [None]
    assert len(worker.episode_buffer[ACTION_INPUTS]) == 1
    assert len(worker.episode_buffer[NEXT_ACTION_INPUTS]) == 1


def test_test_worker_observation_matches_training_tuple_without_node_padding():
    test_worker = make_worker(TestWorker)

    observations = test_worker.get_observations()

    assert len(observations) == 8
    node_inputs, edge_inputs, _, node_padding_mask, edge_padding_mask, edge_mask, map_inputs, action_inputs = observations
    assert node_inputs.shape == (1, 3, 4)
    assert edge_inputs.shape == (1, 1, 4)
    assert edge_inputs.tolist() == [[[0, 1, 2, PADDING_NODE_INDEX]]]
    assert node_padding_mask is None
    assert edge_padding_mask.shape == (1, 1, 4)
    assert edge_mask.shape == (1, 3, 3)
    assert map_inputs is None
    assert action_inputs.shape == (1, 4, ACTION_FEATURE_DIM)
