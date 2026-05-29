import torch

from model import PolicyNet, QNet
from parameter import ACTION_FEATURE_DIM, EMBEDDING_DIM, INPUT_DIM, K_SIZE, USE_BASIN_IN_Q
from worker import Worker


def test_worker_observation_and_buffer_include_basin_action_features():
    policy = PolicyNet(INPUT_DIM, EMBEDDING_DIM, ACTION_FEATURE_DIM)
    q_net = QNet(INPUT_DIM, EMBEDDING_DIM, ACTION_FEATURE_DIM, USE_BASIN_IN_Q)
    worker = Worker(0, policy, q_net, 0, device='cpu', greedy=True, save_image=False)

    observations = worker.get_observations()

    assert len(observations) == 7
    action_features = observations[-1]
    assert action_features.shape == (1, K_SIZE, ACTION_FEATURE_DIM)
    assert torch.isfinite(action_features).all()

    worker.save_observations(observations)
    worker.save_action(torch.tensor([1]))
    worker.save_reward_done(0.0, False)
    worker.save_next_observations(observations)

    assert len(worker.episode_buffer) == 17
    assert worker.episode_buffer[6][0].shape == (K_SIZE, ACTION_FEATURE_DIM)
    assert worker.episode_buffer[16][0].shape == (K_SIZE, ACTION_FEATURE_DIM)
