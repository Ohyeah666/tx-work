import torch

from model import NodeMapFeatureSampler, PolicyNet, QNet, SpatialMapEncoder


def make_model_inputs(batch_size=2, n_nodes=6, k_size=4, input_dim=4, map_size=32):
    torch.manual_seed(7)
    node_inputs = torch.rand(batch_size, n_nodes, input_dim)
    node_inputs[:, :, :2] = torch.rand(batch_size, n_nodes, 2)
    edge_inputs = torch.tensor([[[0, 1, 2, 0]], [[0, 2, 3, 0]]], dtype=torch.long)
    current_index = torch.zeros(batch_size, 1, 1, dtype=torch.long)
    node_padding_mask = torch.zeros(batch_size, 1, n_nodes, dtype=torch.bool)
    node_padding_mask[:, :, -1] = True
    edge_padding_mask = edge_inputs.eq(0)
    original_edge_padding_mask = edge_padding_mask.clone()
    edge_mask = torch.zeros(batch_size, n_nodes, n_nodes, dtype=torch.bool)
    edge_mask[:, :, -1] = True
    map_inputs = torch.zeros(batch_size, 5, map_size, map_size, dtype=torch.uint8)
    map_inputs[:, 0] = 255
    map_inputs[:, 3, map_size // 2, map_size // 2] = 255
    return (
        node_inputs,
        edge_inputs,
        current_index,
        node_padding_mask,
        edge_padding_mask,
        original_edge_padding_mask,
        edge_mask,
        map_inputs,
    )


def test_spatial_map_encoder_outputs_spatial_feature_map():
    encoder = SpatialMapEncoder(input_channels=5, feature_dim=64)
    map_inputs = torch.zeros(2, 5, 120, 160, dtype=torch.uint8)

    feature_map = encoder(map_inputs)

    assert feature_map.shape == (2, 64, 15, 20)
    assert torch.isfinite(feature_map).all()


def test_node_map_feature_sampler_samples_nodes_and_zeros_padding():
    feature_map = torch.arange(16, dtype=torch.float32).view(1, 1, 4, 4)
    feature_map = torch.cat((feature_map, feature_map + 100), dim=1)
    node_inputs = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 0.0],
                [3.0 / 4.0, 3.0 / 4.0, 0.0, 0.0],
            ]
        ]
    )
    sampler = NodeMapFeatureSampler(map_resolution=1, coord_scale=4)

    sampled = sampler(feature_map, node_inputs, map_height=4, map_width=4)
    masked = sampler(
        feature_map,
        node_inputs,
        map_height=4,
        map_width=4,
        node_padding_mask=torch.tensor([[[False, True]]]),
    )

    torch.testing.assert_close(sampled[0, 0], torch.tensor([0.0, 100.0]))
    torch.testing.assert_close(sampled[0, 1], torch.tensor([15.0, 115.0]))
    torch.testing.assert_close(masked[0, 1], torch.zeros(2))


def test_policy_net_forward_returns_log_probs_and_diagnostics_without_mutating_masks():
    inputs = make_model_inputs()
    (
        node_inputs,
        edge_inputs,
        current_index,
        node_padding_mask,
        edge_padding_mask,
        original_edge_padding_mask,
        edge_mask,
        map_inputs,
    ) = inputs
    policy = PolicyNet(
        input_dim=4,
        embedding_dim=16,
        map_input_channels=5,
        map_feature_dim=8,
        map_resolution=4,
        gate_bias_init=-2.0,
    )

    logp, diagnostics = policy(
        node_inputs,
        edge_inputs,
        current_index,
        node_padding_mask,
        edge_padding_mask,
        edge_mask,
        map_inputs,
        return_diagnostics=True,
    )

    assert logp.shape == (2, 4)
    assert torch.isfinite(logp).all()
    torch.testing.assert_close(logp.exp().sum(dim=-1), torch.ones(2))
    torch.testing.assert_close(edge_padding_mask, original_edge_padding_mask)
    assert diagnostics["node_map_feature_norm"].shape == (2, 6)
    assert torch.isfinite(diagnostics["map_feature_std"])
    assert torch.isfinite(diagnostics["fusion_gate_mean"])
    assert torch.isfinite(diagnostics["fusion_gate_std"])


def test_q_net_forward_returns_action_values_and_diagnostics_without_mutating_masks():
    inputs = make_model_inputs()
    (
        node_inputs,
        edge_inputs,
        current_index,
        node_padding_mask,
        edge_padding_mask,
        original_edge_padding_mask,
        edge_mask,
        map_inputs,
    ) = inputs
    q_net = QNet(
        input_dim=4,
        embedding_dim=16,
        map_input_channels=5,
        map_feature_dim=8,
        map_resolution=4,
        gate_bias_init=-2.0,
    )

    q_values, attention_weights, diagnostics = q_net(
        node_inputs,
        edge_inputs,
        current_index,
        node_padding_mask,
        edge_padding_mask,
        edge_mask,
        map_inputs,
        return_diagnostics=True,
    )

    assert q_values.shape == (2, 4, 1)
    assert attention_weights is not None
    assert torch.isfinite(q_values).all()
    torch.testing.assert_close(edge_padding_mask, original_edge_padding_mask)
    torch.testing.assert_close(q_values[:, 0], torch.zeros_like(q_values[:, 0]))
    assert diagnostics["node_map_feature_norm"].shape == (2, 6)
