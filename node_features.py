from dataclasses import dataclass

import numpy as np

from parameter import (
    ACTION_FEATURE_DIM,
    BRANCH_GAIN_EPS,
    GRAPH_DISTANCE_NORMALIZER,
    PADDING_NODE_INDEX,
    TRAJECTORY_MEMORY_GAMMA,
    TRAJECTORY_MEMORY_SIGMA,
    TRAJECTORY_MEMORY_WINDOW,
    USE_ACTION_FEATURES,
    USE_DIRECTIONAL_BRANCH_FEATURES,
    USE_NODE_FEATURE_GRAPH_DIST_TO_CURRENT,
    USE_NODE_FEATURE_TRAJECTORY_MEMORY,
    USE_NODE_FEATURE_UTILITY_OVER_DIST,
    USE_NODE_FEATURE_VISIT_COUNT,
    USE_TRAJECTORY_MEMORY,
)


EDGE_DIST_FEATURE = 0
IMMEDIATE_REVERSE_FEATURE = 1
NEXT_NODE_MEMORY_FEATURE = 2
BRANCH_UTILITY_FEATURE = 3
BRANCH_GAIN_FEATURE = 4
BRANCH_MEMORY_FEATURE = 5


@dataclass(frozen=True)
class FeatureConfig:
    use_graph_dist_to_current: bool = USE_NODE_FEATURE_GRAPH_DIST_TO_CURRENT
    use_utility_over_dist: bool = USE_NODE_FEATURE_UTILITY_OVER_DIST
    use_visit_count: bool = USE_NODE_FEATURE_VISIT_COUNT
    use_node_memory: bool = USE_NODE_FEATURE_TRAJECTORY_MEMORY
    use_action_features: bool = USE_ACTION_FEATURES
    use_trajectory_memory: bool = USE_TRAJECTORY_MEMORY
    use_directional_branch_features: bool = USE_DIRECTIONAL_BRANCH_FEATURES
    action_feature_dim: int = ACTION_FEATURE_DIM
    graph_distance_normalizer: float = GRAPH_DISTANCE_NORMALIZER
    trajectory_memory_gamma: float = TRAJECTORY_MEMORY_GAMMA
    trajectory_memory_sigma: float = TRAJECTORY_MEMORY_SIGMA
    trajectory_memory_window: int = TRAJECTORY_MEMORY_WINDOW
    branch_gain_eps: float = BRANCH_GAIN_EPS


@dataclass
class ObservationFeatures:
    node_inputs: np.ndarray
    edge_inputs: np.ndarray
    action_inputs: np.ndarray
    current_index: int
    edge_padding_mask: np.ndarray
    edge_mask: np.ndarray


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
    if not config.use_action_features:
        return action_inputs

    previous_index = find_previous_route_index(env, current_index)
    node_utility_norm = np.asarray(env.node_utility) / 50
    normalized_distances = distances / config.graph_distance_normalizer

    for action_slot, next_index in enumerate(edge_inputs):
        if edge_padding_mask[action_slot] == 1 or next_index == PADDING_NODE_INDEX:
            continue

        branch_nodes = np.where((first_hops == next_index) & reachable)[0]
        branch_nodes = branch_nodes[branch_nodes != current_index]

        action_inputs[action_slot, EDGE_DIST_FEATURE] = get_edge_distance(
            env,
            current_index,
            next_index,
            config.graph_distance_normalizer,
        )
        action_inputs[action_slot, IMMEDIATE_REVERSE_FEATURE] = int(next_index == previous_index)
        action_inputs[action_slot, NEXT_NODE_MEMORY_FEATURE] = route_memory[next_index]

        if config.use_directional_branch_features and branch_nodes.size > 0:
            branch_utility = np.sum(node_utility_norm[branch_nodes])
            branch_gain = np.sum(
                node_utility_norm[branch_nodes] /
                (normalized_distances[branch_nodes] + config.branch_gain_eps)
            )
            branch_memory = np.mean(route_memory[branch_nodes])

            action_inputs[action_slot, BRANCH_UTILITY_FEATURE] = np.clip(branch_utility, 0, 1)
            action_inputs[action_slot, BRANCH_GAIN_FEATURE] = np.clip(branch_gain, 0, 1)
            action_inputs[action_slot, BRANCH_MEMORY_FEATURE] = branch_memory

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


def get_edge_distance(env, current_index, next_index, normalizer):
    current_edges = get_graph_edges(env.graph).get(str(current_index), {})
    edge = current_edges.get(str(next_index))
    if edge is not None:
        return float(edge.length) / normalizer

    current_coords = env.node_coords[current_index]
    next_coords = env.node_coords[next_index]
    return float(np.linalg.norm(current_coords - next_coords)) / normalizer


def get_graph_edges(graph):
    if hasattr(graph, 'edges'):
        return graph.edges
    return graph
