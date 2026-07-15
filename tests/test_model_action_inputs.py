import torch
import torch.nn as nn

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


def test_policy_net_builds_negative_clipped_distance_logit_bias():
    policy = PolicyNet(
        input_dim=6,
        embedding_dim=16,
        action_input_dim=1,
        distance_attention_init_scale=0.5,
        distance_attention_max_norm=2.0,
    )
    action_inputs = torch.tensor([[[0.0], [1.0], [3.0]]])

    bias = policy.build_pointer_distance_logit_bias(action_inputs)

    assert bias.shape == (1, 1, 3)
    torch.testing.assert_close(bias, torch.tensor([[[-0.0, -0.5, -1.0]]]))


def test_distance_aware_pointer_prefers_shorter_action_when_base_logits_tie():
    policy = PolicyNet(
        input_dim=1,
        embedding_dim=8,
        action_input_dim=1,
        distance_attention_init_scale=1.0,
        distance_attention_max_norm=2.0,
    )
    with torch.no_grad():
        for module in policy.modules():
            if isinstance(module, nn.Linear):
                module.weight.zero_()
                module.bias.zero_()
        policy.pointer.w_query.zero_()
        policy.pointer.w_key.zero_()

    enhanced_node_feature = torch.zeros(1, 3, 8)
    edge_inputs = torch.tensor([[[0, 1, 2, -1]]])
    action_inputs = torch.tensor([[[0.0], [0.1], [0.4], [0.0]]])
    current_index = torch.tensor([[[0]]])
    edge_padding_mask = torch.tensor([[[0, 0, 0, 1]]], dtype=torch.bool)

    logp = policy.output_policy(
        enhanced_node_feature,
        edge_inputs,
        action_inputs,
        current_index,
        edge_padding_mask,
        node_padding_mask=None,
    )

    assert torch.isfinite(logp).all()
    assert logp[0, 0] < -1e7
    assert logp[0, 3] < -1e7
    assert logp[0, 1] > logp[0, 2]
