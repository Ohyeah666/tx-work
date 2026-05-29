import torch

from model import PolicyNet, QNet


def test_policy_net_accepts_basin_action_features_and_returns_log_probs():
    torch.manual_seed(0)
    policy = PolicyNet(input_dim=9, embedding_dim=16, action_feature_dim=5)
    node_inputs = torch.rand((1, 4, 9))
    edge_inputs = torch.tensor([[[0, 1, 2]]])
    current_index = torch.tensor([[[0]]])
    edge_padding_mask = torch.zeros((1, 1, 3), dtype=torch.int64)
    edge_mask = torch.zeros((1, 4, 4))
    action_features = torch.rand((1, 3, 5))

    logp = policy(
        node_inputs,
        edge_inputs,
        current_index,
        node_padding_mask=None,
        edge_padding_mask=edge_padding_mask,
        edge_mask=edge_mask,
        action_features=action_features,
    )

    assert logp.shape == (1, 3)
    assert torch.isfinite(logp).all()
    torch.testing.assert_close(torch.logsumexp(logp, dim=-1), torch.zeros(1), atol=1e-5, rtol=1e-5)


def test_policy_net_action_features_fallback_keeps_forward_compatible():
    torch.manual_seed(0)
    policy = PolicyNet(input_dim=9, embedding_dim=16, action_feature_dim=5)
    node_inputs = torch.rand((1, 4, 9))
    edge_inputs = torch.tensor([[[0, 1, 2]]])
    current_index = torch.tensor([[[0]]])
    edge_padding_mask = torch.zeros((1, 1, 3), dtype=torch.int64)
    edge_mask = torch.zeros((1, 4, 4))

    logp = policy(
        node_inputs,
        edge_inputs,
        current_index,
        node_padding_mask=None,
        edge_padding_mask=edge_padding_mask,
        edge_mask=edge_mask,
    )

    assert logp.shape == (1, 3)
    assert torch.isfinite(logp).all()


def test_q_net_forward_does_not_require_basin_action_features():
    torch.manual_seed(0)
    q_net = QNet(input_dim=9, embedding_dim=16)
    node_inputs = torch.rand((1, 4, 9))
    edge_inputs = torch.tensor([[[0, 1, 2]]])
    current_index = torch.tensor([[[0]]])
    edge_padding_mask = torch.zeros((1, 1, 3), dtype=torch.int64)
    edge_mask = torch.zeros((1, 4, 4))

    q_values, attention_weights = q_net(
        node_inputs,
        edge_inputs,
        current_index,
        node_padding_mask=None,
        edge_padding_mask=edge_padding_mask,
        edge_mask=edge_mask,
    )

    assert q_values.shape == (1, 3, 1)
    assert attention_weights is not None


def test_q_net_ignores_basin_features_when_disabled():
    torch.manual_seed(0)
    q_net = QNet(input_dim=9, embedding_dim=16, action_feature_dim=5, use_action_features=False)
    node_inputs = torch.rand((1, 4, 9))
    edge_inputs = torch.tensor([[[0, 1, 2]]])
    current_index = torch.tensor([[[0]]])
    edge_padding_mask = torch.zeros((1, 1, 3), dtype=torch.int64)
    edge_mask = torch.zeros((1, 4, 4))
    action_features = torch.rand((1, 3, 5))

    q_without_features, _ = q_net(
        node_inputs,
        edge_inputs,
        current_index,
        node_padding_mask=None,
        edge_padding_mask=edge_padding_mask,
        edge_mask=edge_mask,
    )
    q_with_features, _ = q_net(
        node_inputs,
        edge_inputs,
        current_index,
        node_padding_mask=None,
        edge_padding_mask=edge_padding_mask,
        edge_mask=edge_mask,
        action_features=action_features,
    )

    torch.testing.assert_close(q_without_features, q_with_features)


def test_q_net_basin_residual_bias_is_zero_initialized_when_enabled():
    torch.manual_seed(0)
    q_base = QNet(input_dim=9, embedding_dim=16, action_feature_dim=5, use_action_features=False)
    q_basin = QNet(input_dim=9, embedding_dim=16, action_feature_dim=5, use_action_features=True)
    q_basin.load_state_dict(q_base.state_dict(), strict=False)

    node_inputs = torch.rand((1, 4, 9))
    edge_inputs = torch.tensor([[[0, 1, 2]]])
    current_index = torch.tensor([[[0]]])
    edge_padding_mask = torch.zeros((1, 1, 3), dtype=torch.int64)
    edge_mask = torch.zeros((1, 4, 4))
    action_features = torch.rand((1, 3, 5))

    q_base_values, _ = q_base(
        node_inputs,
        edge_inputs,
        current_index,
        node_padding_mask=None,
        edge_padding_mask=edge_padding_mask,
        edge_mask=edge_mask,
    )
    q_basin_values, _ = q_basin(
        node_inputs,
        edge_inputs,
        current_index,
        node_padding_mask=None,
        edge_padding_mask=edge_padding_mask,
        edge_mask=edge_mask,
        action_features=action_features,
    )

    torch.testing.assert_close(q_base_values, q_basin_values)
    torch.testing.assert_close(q_basin.action_q_bias[-1].weight, torch.zeros_like(q_basin.action_q_bias[-1].weight))
    torch.testing.assert_close(q_basin.action_q_bias[-1].bias, torch.zeros_like(q_basin.action_q_bias[-1].bias))
