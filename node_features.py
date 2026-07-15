from dataclasses import dataclass
from typing import Optional

import numpy as np

from parameter import (
    ACTION_FEATURE_DIM,
    BRANCH_GAIN_EPS,
    DISTANCE_ATTENTION_MAX_NORM,
    GRAPH_DISTANCE_NORMALIZER,
    PADDING_NODE_INDEX,
    TRAJECTORY_MEMORY_GAMMA,
    TRAJECTORY_MEMORY_SIGMA,
    TRAJECTORY_MEMORY_WINDOW,
    USE_ACTION_FEATURES,
    USE_ACTION_FEATURE_BRANCH_GAIN,
    USE_ACTION_FEATURE_BRANCH_MEMORY,
    USE_ACTION_FEATURE_BRANCH_UTILITY,
    USE_ACTION_FEATURE_EDGE_DIST,
    USE_ACTION_FEATURE_IMMEDIATE_REVERSE,
    USE_ACTION_FEATURE_NEXT_NODE_MEMORY,
    USE_DIRECTIONAL_BRANCH_FEATURES,
    USE_NODE_FEATURE_GRAPH_DIST_TO_CURRENT,
    USE_NODE_FEATURE_TRAJECTORY_MEMORY,
    USE_NODE_FEATURE_UTILITY_OVER_DIST,
    USE_NODE_FEATURE_VISIT_COUNT,
    USE_TRAJECTORY_MEMORY,
)


ACTION_FEATURE_EDGE_DIST = 'edge_dist_norm'
ACTION_FEATURE_IMMEDIATE_REVERSE = 'is_immediate_reverse'
ACTION_FEATURE_NEXT_NODE_MEMORY = 'next_node_memory'
ACTION_FEATURE_BRANCH_UTILITY = 'branch_utility_norm'
ACTION_FEATURE_BRANCH_GAIN = 'branch_gain_norm'
ACTION_FEATURE_BRANCH_MEMORY = 'branch_memory'


@dataclass(frozen=True)
class FeatureConfig:
    use_graph_dist_to_current: bool = USE_NODE_FEATURE_GRAPH_DIST_TO_CURRENT
    use_utility_over_dist: bool = USE_NODE_FEATURE_UTILITY_OVER_DIST
    use_visit_count: bool = USE_NODE_FEATURE_VISIT_COUNT
    use_node_memory: bool = USE_NODE_FEATURE_TRAJECTORY_MEMORY
    use_action_features: bool = USE_ACTION_FEATURES
    use_action_feature_edge_dist: bool = USE_ACTION_FEATURE_EDGE_DIST
    use_action_feature_immediate_reverse: bool = USE_ACTION_FEATURE_IMMEDIATE_REVERSE
    use_action_feature_next_node_memory: bool = USE_ACTION_FEATURE_NEXT_NODE_MEMORY
    use_action_feature_branch_utility: bool = USE_ACTION_FEATURE_BRANCH_UTILITY
    use_action_feature_branch_gain: bool = USE_ACTION_FEATURE_BRANCH_GAIN
    use_action_feature_branch_memory: bool = USE_ACTION_FEATURE_BRANCH_MEMORY
    use_trajectory_memory: bool = USE_TRAJECTORY_MEMORY
    use_directional_branch_features: bool = USE_DIRECTIONAL_BRANCH_FEATURES
    action_feature_dim: Optional[int] = None
    graph_distance_normalizer: float = GRAPH_DISTANCE_NORMALIZER
    distance_attention_max_norm: float = DISTANCE_ATTENTION_MAX_NORM
    trajectory_memory_gamma: float = TRAJECTORY_MEMORY_GAMMA
    trajectory_memory_sigma: float = TRAJECTORY_MEMORY_SIGMA
    trajectory_memory_window: int = TRAJECTORY_MEMORY_WINDOW
    branch_gain_eps: float = BRANCH_GAIN_EPS

    def __post_init__(self):
        object.__setattr__(self, 'action_feature_dim', get_action_feature_dim(self))


@dataclass
class ObservationFeatures:
    node_inputs: np.ndarray
    edge_inputs: np.ndarray
    action_inputs: np.ndarray
    current_index: int
    edge_padding_mask: np.ndarray
    edge_mask: np.ndarray


def get_active_action_feature_names(config):
    if not config.use_action_features:
        return ()

    feature_names = []
    if config.use_action_feature_edge_dist:
        feature_names.append(ACTION_FEATURE_EDGE_DIST)
    if config.use_action_feature_immediate_reverse:
        feature_names.append(ACTION_FEATURE_IMMEDIATE_REVERSE)
    if config.use_action_feature_next_node_memory:
        feature_names.append(ACTION_FEATURE_NEXT_NODE_MEMORY)

    if config.use_directional_branch_features:
        if config.use_action_feature_branch_utility:
            feature_names.append(ACTION_FEATURE_BRANCH_UTILITY)
        if config.use_action_feature_branch_gain:
            feature_names.append(ACTION_FEATURE_BRANCH_GAIN)
        if config.use_action_feature_branch_memory:
            feature_names.append(ACTION_FEATURE_BRANCH_MEMORY)

    return tuple(feature_names)


def get_action_feature_dim(config):
    return len(get_active_action_feature_names(config))


def get_action_feature_indices(config=None):
    config = config or FeatureConfig()
    return {
        feature_name: index
        for index, feature_name in enumerate(get_active_action_feature_names(config))
    }


def get_action_feature_index(feature_name, config=None):
    return get_action_feature_indices(config).get(feature_name)


_DEFAULT_ACTION_FEATURE_INDICES = get_action_feature_indices(FeatureConfig())
EDGE_DIST_FEATURE = _DEFAULT_ACTION_FEATURE_INDICES.get(ACTION_FEATURE_EDGE_DIST)
IMMEDIATE_REVERSE_FEATURE = _DEFAULT_ACTION_FEATURE_INDICES.get(ACTION_FEATURE_IMMEDIATE_REVERSE)
NEXT_NODE_MEMORY_FEATURE = _DEFAULT_ACTION_FEATURE_INDICES.get(ACTION_FEATURE_NEXT_NODE_MEMORY)
BRANCH_UTILITY_FEATURE = _DEFAULT_ACTION_FEATURE_INDICES.get(ACTION_FEATURE_BRANCH_UTILITY)
BRANCH_GAIN_FEATURE = _DEFAULT_ACTION_FEATURE_INDICES.get(ACTION_FEATURE_BRANCH_GAIN)
BRANCH_MEMORY_FEATURE = _DEFAULT_ACTION_FEATURE_INDICES.get(ACTION_FEATURE_BRANCH_MEMORY)


def build_node_and_action_features(env, robot_position, k_size, config=None):
    config = config or FeatureConfig()
    node_coords = np.asarray(env.node_coords)
    node_utility = np.asarray(env.node_utility)
    guidepost = np.asarray(env.guidepost)
    visit_count = np.asarray(env.visit_count)
    n_nodes = node_coords.shape[0]
    current_index = env.find_index_from_coords(robot_position)

    distances, first_hops, reachable = env.graph_generator.get_shortest_path_tree(current_index)
    normalized_distances = distances / config.graph_distance_normalizer
    normalized_distances[~reachable] = env.graph_generator.UNREACHABLE_GRAPH_DISTANCE
    graph_dist_to_current = normalized_distances.reshape(n_nodes, 1)
    reachable_nodes = reachable.reshape(n_nodes, 1)

    route_memory = np.zeros((n_nodes, 1))
    if config.use_trajectory_memory:
        route_memory = env.graph_generator.get_decayed_route_memory(
            gamma=config.trajectory_memory_gamma,
            sigma=config.trajectory_memory_sigma,
            window=config.trajectory_memory_window,
        )

    node_coords_inputs = node_coords / config.graph_distance_normalizer
    node_utility_inputs = (node_utility / 50).reshape(n_nodes, 1)
    node_feature_list = [node_coords_inputs, node_utility_inputs, guidepost]

    if config.use_graph_dist_to_current:
        node_feature_list.append(graph_dist_to_current)

    if config.use_utility_over_dist:
        utility_over_dist = np.zeros_like(node_utility_inputs)
        np.divide(
            node_utility_inputs,
            graph_dist_to_current + config.branch_gain_eps,
            out=utility_over_dist,
            where=reachable_nodes,
        )
        utility_over_dist[current_index] = 0
        node_feature_list.append(utility_over_dist)

    if config.use_visit_count:
        node_feature_list.append(visit_count.reshape(n_nodes, 1))

    if config.use_node_memory:
        node_feature_list.append(route_memory)

    node_inputs = np.concatenate(node_feature_list, axis=1)
    node_inputs = np.nan_to_num(node_inputs, nan=0, posinf=0, neginf=0)

    edge_inputs, edge_padding_mask = build_padded_edge_inputs(
        env.graph,
        current_index,
        k_size,
    )
    edge_mask = build_edge_mask(env.graph, n_nodes)
    action_inputs = build_action_inputs(
        env=env,
        edge_inputs=edge_inputs,
        edge_padding_mask=edge_padding_mask,
        current_index=current_index,
        distances=distances,
        first_hops=first_hops,
        reachable=reachable,
        route_memory=route_memory.reshape(n_nodes),
        config=config,
    )

    return ObservationFeatures(
        node_inputs=node_inputs,
        edge_inputs=edge_inputs,
        action_inputs=action_inputs,
        current_index=current_index,
        edge_padding_mask=edge_padding_mask,
        edge_mask=edge_mask,
    )


def build_padded_edge_inputs(graph, current_index, k_size):
    graph_edges = get_graph_edges(graph)
    current_edges = graph_edges.get(str(current_index), {})
    edge_indices = [int(node_index) for node_index in current_edges.keys()]
    edge_indices = [current_index] + [node_index for node_index in edge_indices if node_index != current_index]

    valid_count = min(len(edge_indices), k_size)
    padded_edges = np.full(k_size, PADDING_NODE_INDEX, dtype=np.int64)
    edge_padding_mask = np.ones(k_size, dtype=np.int64)
    if valid_count > 0:
        padded_edges[:valid_count] = edge_indices[:valid_count]
        edge_padding_mask[:valid_count] = 0

    return padded_edges, edge_padding_mask


def build_edge_mask(graph, n_nodes):
    graph_edges = get_graph_edges(graph)
    edge_mask = np.ones((n_nodes, n_nodes))
    for from_node, edges in graph_edges.items():
        from_index = int(from_node)
        if from_index >= n_nodes:
            continue

        for to_node in edges.keys():
            to_index = int(to_node)
            if to_index < n_nodes:
                edge_mask[from_index, to_index] = 0

    return edge_mask


def build_action_inputs(
    env,
    edge_inputs,
    edge_padding_mask,
    current_index,
    distances,
    first_hops,
    reachable,
    route_memory,
    config,
):
    action_inputs = np.zeros((edge_inputs.shape[0], config.action_feature_dim))
    if not config.use_action_features or config.action_feature_dim == 0:
        return action_inputs

    feature_indices = get_action_feature_indices(config)
    previous_index = find_previous_route_index(env, current_index)
    node_utility_norm = np.asarray(env.node_utility) / 50
    normalized_distances = distances / config.graph_distance_normalizer
    edge_dist_idx = feature_indices.get(ACTION_FEATURE_EDGE_DIST)
    immediate_reverse_idx = feature_indices.get(ACTION_FEATURE_IMMEDIATE_REVERSE)
    next_node_memory_idx = feature_indices.get(ACTION_FEATURE_NEXT_NODE_MEMORY)
    branch_utility_idx = feature_indices.get(ACTION_FEATURE_BRANCH_UTILITY)
    branch_gain_idx = feature_indices.get(ACTION_FEATURE_BRANCH_GAIN)
    branch_memory_idx = feature_indices.get(ACTION_FEATURE_BRANCH_MEMORY)
    need_branch_nodes = any(
        index is not None
        for index in (branch_utility_idx, branch_gain_idx, branch_memory_idx)
    )

    for action_slot, next_index in enumerate(edge_inputs):
        if edge_padding_mask[action_slot] == 1 or next_index == PADDING_NODE_INDEX:
            continue

        if edge_dist_idx is not None:
            action_inputs[action_slot, edge_dist_idx] = get_edge_distance(
                env,
                current_index,
                next_index,
                config.graph_distance_normalizer,
                config.distance_attention_max_norm,
            )

        if immediate_reverse_idx is not None:
            action_inputs[action_slot, immediate_reverse_idx] = int(next_index == previous_index)

        if next_node_memory_idx is not None:
            action_inputs[action_slot, next_node_memory_idx] = route_memory[next_index]

        if config.use_directional_branch_features and need_branch_nodes:
            branch_nodes = np.where((first_hops == next_index) & reachable)[0]
            branch_nodes = branch_nodes[branch_nodes != current_index]
            if branch_nodes.size == 0:
                continue

            if branch_utility_idx is not None:
                branch_utility = np.sum(node_utility_norm[branch_nodes])
                action_inputs[action_slot, branch_utility_idx] = np.clip(branch_utility, 0, 1)

            if branch_gain_idx is not None:
                branch_gain = np.sum(
                    node_utility_norm[branch_nodes] /
                    (normalized_distances[branch_nodes] + config.branch_gain_eps)
                )
                action_inputs[action_slot, branch_gain_idx] = np.clip(branch_gain, 0, 1)

            if branch_memory_idx is not None:
                branch_memory = np.mean(route_memory[branch_nodes])
                action_inputs[action_slot, branch_memory_idx] = branch_memory

    return np.nan_to_num(action_inputs, nan=0, posinf=0, neginf=0)


def find_previous_route_index(env, current_index):
    route = env.graph_generator.route_node
    if len(route) < 2:
        return PADDING_NODE_INDEX

    previous_coords = route[-2]
    coord_to_index = {
        env.graph_generator.coords_to_key(coords): i
        for i, coords in enumerate(env.node_coords)
    }
    previous_index = coord_to_index.get(env.graph_generator.coords_to_key(previous_coords))
    if previous_index is None:
        return PADDING_NODE_INDEX

    if previous_index == current_index and len(route) >= 3:
        previous_coords = route[-3]
        previous_index = coord_to_index.get(env.graph_generator.coords_to_key(previous_coords), PADDING_NODE_INDEX)

    return previous_index


def get_edge_distance(env, current_index, next_index, normalizer, max_norm=None):
    current_edges = get_graph_edges(env.graph).get(str(current_index), {})
    edge = current_edges.get(str(next_index))
    if edge is not None:
        normalized_distance = float(edge.length) / normalizer
        if max_norm is not None:
            normalized_distance = np.clip(normalized_distance, 0, max_norm)
        return float(normalized_distance)

    current_coords = env.node_coords[current_index]
    next_coords = env.node_coords[next_index]
    normalized_distance = float(np.linalg.norm(current_coords - next_coords)) / normalizer
    if max_norm is not None:
        normalized_distance = np.clip(normalized_distance, 0, max_norm)
    return float(normalized_distance)


def get_graph_edges(graph):
    if hasattr(graph, 'edges'):
        return graph.edges
    return graph
