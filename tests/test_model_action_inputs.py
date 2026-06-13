import torch

from model import PolicyNet, QNet


def make_model_inputs(batch_size=2, n_nodes=5, k_size=4, input_dim=6, action_dim=6):
    torch.manual_seed(0)
    node_inputs = torch.rand(batch_size, n_nodes, input_dim)
    edge_inputs = torch.tensor([
        [[0, 1, 2, -1]],
        [[1, 2, 3, -1]],
    ])
    action_inputs = torch.rand(batch_size, k_size, action_dim)
    current_index = torch.tensor([[[0]], [[1]]])
    node_padding_mask = torch.zeros(batch_size, 1, n_nodes, dtype=torch.bool)
    edge_padding_mask = torch.tensor([
        [[0, 0, 0, 1]],
        [[0, 0, 0, 1]],
    ], dtype=torch.bool)
    edge_mask = torch.zeros(batch_size, n_nodes, n_nodes, dtype=torch.bool)
    return node_inputs, edge_inputs, action_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask


def test_policy_net_uses_action_inputs_and_masks_padding_index():
    inputs = make_model_inputs()
    policy = PolicyNet(input_dim=6, embedding_dim=16, action_input_dim=6)

    logp = policy(*inputs)

    assert logp.shape == (2, 4)
    assert torch.isfinite(logp).all()
    assert torch.all(logp[:, 0] < -1e7)
    assert torch.all(logp[:, 3] < -1e7)
    torch.testing.assert_close(logp.exp().sum(dim=1), torch.ones(2))


def test_q_net_uses_action_inputs_and_masks_invalid_actions():
    inputs = make_model_inputs()
    q_net = QNet(input_dim=6, embedding_dim=16, action_input_dim=6)

    q_values, attention_weights = q_net(*inputs)

    assert q_values.shape == (2, 4, 1)
    assert attention_weights is not None
    assert torch.isfinite(q_values).all()
    torch.testing.assert_close(q_values[:, 0], torch.zeros_like(q_values[:, 0]))
    torch.testing.assert_close(q_values[:, 3], torch.zeros_like(q_values[:, 3]))
