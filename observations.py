import torch
import torch.nn.functional as F


OBSERVATION_FIELDS = (
    'node_inputs',
    'edge_inputs',
    'action_inputs',
    'current_index',
    'node_padding_mask',
    'edge_padding_mask',
    'edge_mask',
)


def build_observation_tensors(features, device, input_dim, action_feature_dim, node_padding_size=None):
    node_inputs_np = features.node_inputs
    if node_inputs_np.shape[1] != input_dim:
        raise ValueError(f'node_inputs feature dim {node_inputs_np.shape[1]} does not match INPUT_DIM {input_dim}')
    if features.action_inputs.shape[1] != action_feature_dim:
        raise ValueError(
            f'action_inputs feature dim {features.action_inputs.shape[1]} '
            f'does not match ACTION_FEATURE_DIM {action_feature_dim}'
        )

    n_nodes = node_inputs_np.shape[0]
    node_inputs = torch.as_tensor(node_inputs_np, dtype=torch.float32, device=device).unsqueeze(0)
    edge_mask = torch.as_tensor(features.edge_mask, dtype=torch.float32, device=device).unsqueeze(0)
    node_padding_mask = None

    if node_padding_size is not None:
        if n_nodes > node_padding_size:
            raise ValueError(f'n_nodes {n_nodes} exceeds NODE_PADDING_SIZE {node_padding_size}')

        pad_nodes = node_padding_size - n_nodes
        if pad_nodes > 0:
            node_inputs = F.pad(node_inputs, (0, 0, 0, pad_nodes), value=0)
            edge_mask = F.pad(edge_mask, (0, pad_nodes, 0, pad_nodes), value=1)

        valid_node_mask = torch.zeros((1, 1, n_nodes), dtype=torch.int64, device=device)
        padded_node_mask = torch.ones((1, 1, pad_nodes), dtype=torch.int64, device=device)
        node_padding_mask = torch.cat((valid_node_mask, padded_node_mask), dim=-1)

    edge_inputs = torch.as_tensor(features.edge_inputs, dtype=torch.long, device=device).unsqueeze(0).unsqueeze(0)
    action_inputs = torch.as_tensor(features.action_inputs, dtype=torch.float32, device=device).unsqueeze(0)
    current_index = torch.tensor([features.current_index], device=device).unsqueeze(0).unsqueeze(0)
    edge_padding_mask = torch.as_tensor(
        features.edge_padding_mask,
        dtype=torch.int64,
        device=device,
    ).unsqueeze(0).unsqueeze(0)

    return (
        node_inputs,
        edge_inputs,
        action_inputs,
        current_index,
        node_padding_mask,
        edge_padding_mask,
        edge_mask,
    )
