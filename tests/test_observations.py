import numpy as np
import torch

from node_features import ObservationFeatures
from observations import OBSERVATION_FIELDS, build_observation_tensors
from worker import Worker


def make_features():
    return ObservationFeatures(
        node_inputs=np.ones((3, 5), dtype=float),
        edge_inputs=np.array([0, 1, -1], dtype=np.int64),
        action_inputs=np.array([[0.0], [0.25], [0.0]], dtype=float),
        current_index=1,
        edge_padding_mask=np.array([0, 0, 1], dtype=np.int64),
        edge_mask=np.zeros((3, 3), dtype=float),
    )


def test_build_observation_tensors_preserves_schema_and_training_padding():
    observations = build_observation_tensors(
        features=make_features(),
        device='cpu',
        input_dim=5,
        action_feature_dim=1,
        node_padding_size=5,
    )

    assert len(observations) == len(OBSERVATION_FIELDS) == 7
    node_inputs, edge_inputs, action_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask = observations
    assert node_inputs.shape == (1, 5, 5)
    assert edge_inputs.shape == (1, 1, 3)
    assert action_inputs.shape == (1, 3, 1)
    assert current_index.shape == (1, 1, 1)
    assert node_padding_mask.shape == (1, 1, 5)
    assert edge_padding_mask.shape == (1, 1, 3)
    assert edge_mask.shape == (1, 5, 5)
    torch.testing.assert_close(node_padding_mask, torch.tensor([[[0, 0, 0, 1, 1]]]))
    assert torch.all(edge_mask[:, 3:, :] == 1)
    assert torch.all(edge_mask[:, :, 3:] == 1)


def test_build_observation_tensors_keeps_test_observations_unpadded():
    observations = build_observation_tensors(
        features=make_features(),
        device='cpu',
        input_dim=5,
        action_feature_dim=1,
    )

    node_inputs, _, _, _, node_padding_mask, _, edge_mask = observations
    assert node_inputs.shape == (1, 3, 5)
    assert node_padding_mask is None
    assert edge_mask.shape == (1, 3, 3)


def test_worker_replay_buffer_current_and_next_observation_fields_match():
    worker = Worker.__new__(Worker)
    worker.device = 'cpu'
    worker.episode_buffer = [[] for _ in range(17)]
    observations = build_observation_tensors(
        features=make_features(),
        device='cpu',
        input_dim=5,
        action_feature_dim=1,
        node_padding_size=5,
    )

    worker.save_observations(observations)
    worker.save_action(torch.tensor([1]))
    worker.save_reward_done(reward=1.0, done=False)
    worker.save_next_observations(observations)

    assert len(worker.episode_buffer) == 17
    assert all(len(field) == 1 for field in worker.episode_buffer)
    assert worker.episode_buffer[2][0].shape == worker.episode_buffer[12][0].shape
    assert worker.episode_buffer[6][0].shape == worker.episode_buffer[16][0].shape
