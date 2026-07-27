import numpy as np


COORD_NORMALIZER = 640.0
UTILITY_NORMALIZER = 50.0
GRAPH_DIST_NORMALIZER = 640.0
UNREACHABLE_GRAPH_DIST = 2.0
UTILITY_OVER_DIST_EPS = 1e-6

BASE_NODE_INPUT_DIM = 4
DEFAULT_USE_NODE_GRAPH_DIST_TO_CURRENT = False
DEFAULT_USE_NODE_UTILITY_OVER_DIST = True
DEFAULT_USE_NODE_VISIT_COUNT = False

NODE_INPUT_DIM = (
    BASE_NODE_INPUT_DIM
    + int(DEFAULT_USE_NODE_GRAPH_DIST_TO_CURRENT)
    + int(DEFAULT_USE_NODE_UTILITY_OVER_DIST)
    + int(DEFAULT_USE_NODE_VISIT_COUNT)
)


def compute_node_input_dim(use_graph_dist_to_current=False,
                           use_utility_over_dist=True,
                           use_visit_count=False):
    return (
        BASE_NODE_INPUT_DIM
        + int(use_graph_dist_to_current)
        + int(use_utility_over_dist)
        + int(use_visit_count)
    )


def build_node_inputs(node_coords, node_utility, guidepost, graph_dist_to_current,
                      reachable_nodes, visit_count, current_node_index,
                      use_graph_dist_to_current=DEFAULT_USE_NODE_GRAPH_DIST_TO_CURRENT,
                      use_utility_over_dist=DEFAULT_USE_NODE_UTILITY_OVER_DIST,
                      use_visit_count=DEFAULT_USE_NODE_VISIT_COUNT):
    node_coords = np.asarray(node_coords, dtype=np.float32)
    n_nodes = node_coords.shape[0]

    normalized_coords = node_coords / COORD_NORMALIZER
    normalized_utility = np.asarray(node_utility, dtype=np.float32).reshape(n_nodes, 1) / UTILITY_NORMALIZER
    guidepost = np.asarray(guidepost, dtype=np.float32).reshape(n_nodes, 1)

    feature_blocks = [normalized_coords, normalized_utility, guidepost]

    if use_graph_dist_to_current or use_utility_over_dist:
        graph_dist = np.asarray(graph_dist_to_current, dtype=np.float32).reshape(n_nodes, 1)
        if reachable_nodes is None:
            reachable = np.isfinite(graph_dist)
        else:
            reachable = np.asarray(reachable_nodes, dtype=bool).reshape(n_nodes, 1)
        graph_dist = np.where(reachable, graph_dist, UNREACHABLE_GRAPH_DIST).astype(np.float32)

        if use_graph_dist_to_current:
            feature_blocks.append(graph_dist)

        if use_utility_over_dist:
            utility_over_dist = np.zeros((n_nodes, 1), dtype=np.float32)
            np.divide(
                normalized_utility,
                graph_dist + UTILITY_OVER_DIST_EPS,
                out=utility_over_dist,
                where=reachable,
            )
            utility_over_dist[int(current_node_index)] = 0.0
            feature_blocks.append(utility_over_dist)

    if use_visit_count:
        visit_count = np.asarray(visit_count, dtype=np.float32).reshape(n_nodes, 1)
        feature_blocks.append(visit_count)

    expected_dim = compute_node_input_dim(
        use_graph_dist_to_current=use_graph_dist_to_current,
        use_utility_over_dist=use_utility_over_dist,
        use_visit_count=use_visit_count,
    )
    node_inputs = np.concatenate(tuple(feature_blocks), axis=1)

    if node_inputs.shape[1] != expected_dim:
        raise ValueError(f"Expected {expected_dim} node features, got {node_inputs.shape[1]}.")
    if not np.isfinite(node_inputs).all():
        raise ValueError("Node inputs contain NaN or Inf values.")

    return node_inputs.astype(np.float32)
