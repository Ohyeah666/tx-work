import numpy as np


ACTION_FEATURE_EDGE_DIST = "edge_dist_norm"


def get_active_action_feature_names(use_action_features=True, use_action_feature_edge_dist=True):
    if not use_action_features:
        return ()
    if not use_action_feature_edge_dist:
        return ()
    return (ACTION_FEATURE_EDGE_DIST,)


def get_action_feature_dim(use_action_features=True, use_action_feature_edge_dist=True):
    return len(
        get_active_action_feature_names(
            use_action_features=use_action_features,
            use_action_feature_edge_dist=use_action_feature_edge_dist,
        )
    )


def get_action_feature_indices(use_action_features=True, use_action_feature_edge_dist=True):
    return {
        feature_name: index
        for index, feature_name in enumerate(
            get_active_action_feature_names(
                use_action_features=use_action_features,
                use_action_feature_edge_dist=use_action_feature_edge_dist,
            )
        )
    }


def build_edge_dist_action_inputs(
    env,
    current_index,
    edge_inputs,
    edge_padding_mask,
    normalizer,
    max_norm=None,
    use_action_features=True,
    use_action_feature_edge_dist=True,
):
    action_feature_dim = get_action_feature_dim(
        use_action_features=use_action_features,
        use_action_feature_edge_dist=use_action_feature_edge_dist,
    )
    edge_inputs = np.asarray(edge_inputs, dtype=np.int64).reshape(-1)
    edge_padding_mask = np.asarray(edge_padding_mask).reshape(-1)
    action_inputs = np.zeros((edge_inputs.shape[0], action_feature_dim), dtype=np.float32)

    if action_feature_dim == 0:
        return action_inputs

    edge_dist_idx = get_action_feature_indices(
        use_action_features=use_action_features,
        use_action_feature_edge_dist=use_action_feature_edge_dist,
    ).get(ACTION_FEATURE_EDGE_DIST)
    if edge_dist_idx is None:
        return action_inputs

    for action_slot, next_index in enumerate(edge_inputs):
        if edge_padding_mask[action_slot] or next_index < 0:
            continue
        action_inputs[action_slot, edge_dist_idx] = get_edge_distance(
            env=env,
            current_index=current_index,
            next_index=int(next_index),
            normalizer=normalizer,
            max_norm=max_norm,
        )

    return np.nan_to_num(action_inputs, nan=0.0, posinf=0.0, neginf=0.0)


def build_padded_current_edge_inputs(graph, current_index, k_size, padding_node_index):
    graph_edges = get_graph_edges(graph)
    current_edges = graph_edges.get(str(current_index), {})
    edge_indices = [int(node_index) for node_index in current_edges.keys()]
    edge_indices = [current_index] + [node_index for node_index in edge_indices if node_index != current_index]

    valid_count = min(len(edge_indices), k_size)
    padded_edges = np.full(k_size, padding_node_index, dtype=np.int64)
    edge_padding_mask = np.ones(k_size, dtype=np.int64)
    if valid_count > 0:
        padded_edges[:valid_count] = edge_indices[:valid_count]
        edge_padding_mask[:valid_count] = 0

    return padded_edges, edge_padding_mask


def get_edge_distance(env, current_index, next_index, normalizer, max_norm=None):
    graph_edges = get_graph_edges(env.graph)
    edge = graph_edges.get(str(current_index), {}).get(str(next_index))
    if edge is not None and hasattr(edge, "length"):
        distance = float(edge.length)
    else:
        current_coords = np.asarray(env.node_coords[current_index], dtype=float)
        next_coords = np.asarray(env.node_coords[next_index], dtype=float)
        distance = float(np.linalg.norm(current_coords - next_coords))

    normalized_distance = distance / normalizer
    if max_norm is not None:
        normalized_distance = np.clip(normalized_distance, 0.0, max_norm)
    return float(normalized_distance)


def get_graph_edges(graph):
    if hasattr(graph, "edges"):
        return graph.edges
    return graph
