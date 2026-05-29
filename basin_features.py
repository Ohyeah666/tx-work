import math

import numpy as np


BASIN_ACTION_FEATURE_ORDER = (
    "basin_utility_sum",
    "basin_expected_unknown_gain_sum",
    "basin_frontier_cluster_max",
    "basin_unvisited_ratio",
    "basin_min_dist_to_utility",
)


def _reshape_feature(feature):
    return np.asarray(feature, dtype=float).reshape(-1)


def _log1p_normalize(value, normalizer):
    if normalizer <= 0:
        return 0.0
    return min(math.log1p(max(value, 0.0)) / math.log1p(normalizer), 1.0)


def build_basin_action_features(edge_indices, first_hop, graph_dist_to_current, node_utility,
                                visit_count, expected_unknown_gain, frontier_cluster_size,
                                current_node_index, edge_padding_mask=None, k_size=None,
                                utility_threshold=0.0, no_utility_distance=2.0,
                                utility_sum_normalizer=18000.0,
                                expected_unknown_gain_sum_normalizer=360.0):
    edge_indices = np.asarray(edge_indices, dtype=int).reshape(-1)
    first_hop = np.asarray(first_hop, dtype=int).reshape(-1)
    graph_dist_to_current = _reshape_feature(graph_dist_to_current)
    node_utility = _reshape_feature(node_utility)
    visit_count = _reshape_feature(visit_count)
    expected_unknown_gain = _reshape_feature(expected_unknown_gain)
    frontier_cluster_size = _reshape_feature(frontier_cluster_size)

    if k_size is None:
        k_size = edge_indices.shape[0]

    if edge_padding_mask is None:
        edge_padding_mask = np.zeros(k_size, dtype=bool)
    else:
        edge_padding_mask = np.asarray(edge_padding_mask, dtype=bool).reshape(-1)

    action_features = np.zeros((k_size, len(BASIN_ACTION_FEATURE_ORDER)), dtype=float)
    n_nodes = first_hop.shape[0]

    for action_slot in range(min(k_size, edge_indices.shape[0])):
        if action_slot < edge_padding_mask.shape[0] and edge_padding_mask[action_slot]:
            continue

        neighbor_index = int(edge_indices[action_slot])
        if neighbor_index == current_node_index or not (0 <= neighbor_index < n_nodes):
            continue

        basin_mask = first_hop == neighbor_index
        if not np.any(basin_mask):
            continue

        basin_utility_sum = float(np.sum(node_utility[basin_mask]))
        basin_expected_unknown_gain_sum = float(np.sum(expected_unknown_gain[basin_mask]))
        basin_frontier_cluster_max = float(np.max(frontier_cluster_size[basin_mask]))
        basin_unvisited_ratio = float(np.mean(visit_count[basin_mask] <= 0))

        utility_nodes = basin_mask & (node_utility > utility_threshold) & np.isfinite(graph_dist_to_current)
        if np.any(utility_nodes):
            basin_min_dist_to_utility = float(np.min(graph_dist_to_current[utility_nodes]))
        else:
            basin_min_dist_to_utility = no_utility_distance

        action_features[action_slot] = (
            _log1p_normalize(basin_utility_sum, utility_sum_normalizer),
            _log1p_normalize(basin_expected_unknown_gain_sum, expected_unknown_gain_sum_normalizer),
            np.clip(basin_frontier_cluster_max, 0.0, 1.0),
            np.clip(basin_unvisited_ratio, 0.0, 1.0),
            max(basin_min_dist_to_utility, 0.0),
        )

    return np.nan_to_num(action_features, nan=0.0, posinf=0.0, neginf=0.0)

