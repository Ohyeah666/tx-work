import numpy as np


FEATURE_ORDER_V1 = (
    "x",
    "y",
    "utility",
    "guidepost",
    "graph_dist_to_current",
    "utility_over_dist",
    "visit_count",
    "expected_unknown_gain",
    "frontier_cluster_size",
)


def build_node_inputs(node_coords, node_utility, guidepost, graph_dist_to_current, reachable_nodes,
                      visit_count, expected_unknown_gain, frontier_cluster_size, current_node_index,
                      coord_normalizer=640, utility_normalizer=50, eps=1e-6):
    node_coords = node_coords / coord_normalizer
    node_utility_inputs = (node_utility / utility_normalizer).reshape((-1, 1))
    guidepost = guidepost.reshape((-1, 1))
    graph_dist_to_current = graph_dist_to_current.reshape((-1, 1))
    reachable_nodes = reachable_nodes.reshape((-1, 1))
    visit_count_inputs = visit_count.reshape((-1, 1))
    expected_unknown_gain = expected_unknown_gain.reshape((-1, 1))
    frontier_cluster_size = frontier_cluster_size.reshape((-1, 1))

    utility_over_dist = np.zeros_like(node_utility_inputs)
    np.divide(
        node_utility_inputs,
        graph_dist_to_current + eps,
        out=utility_over_dist,
        where=reachable_nodes,
    )
    utility_over_dist[current_node_index] = 0

    node_inputs = np.concatenate(
        (
            node_coords,
            node_utility_inputs,
            guidepost,
            graph_dist_to_current,
            utility_over_dist,
            visit_count_inputs,
            expected_unknown_gain,
            frontier_cluster_size,
        ),
        axis=1,
    )
    return np.nan_to_num(node_inputs, nan=0, posinf=0, neginf=0)
