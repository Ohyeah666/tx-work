from types import SimpleNamespace

import numpy as np

from action_features import (
    ACTION_FEATURE_EDGE_DIST,
    build_edge_dist_action_inputs,
    build_padded_current_edge_inputs,
    get_action_feature_indices,
    get_edge_distance,
)
from graph import Edge
from parameter import EDGE_DIST_MAX_NORM, GRAPH_DISTANCE_NORMALIZER, PADDING_NODE_INDEX


def make_env():
    return SimpleNamespace(
        node_coords=np.array(
            [
                [0.0, 0.0],
                [64.0, 0.0],
                [0.0, 128.0],
                [640.0, 0.0],
            ],
            dtype=np.float32,
        ),
        graph={
            "0": {"0": Edge("0", 0.0), "1": Edge("1", 64.0), "2": Edge("2", 128.0)},
            "1": {"1": Edge("1", 0.0), "0": Edge("0", 64.0), "3": Edge("3", 2048.0)},
        },
    )


def test_action_feature_indices_default_to_edge_distance_column_zero():
    indices = get_action_feature_indices()

    assert indices == {ACTION_FEATURE_EDGE_DIST: 0}


def test_build_padded_current_edge_inputs_uses_negative_padding_without_masking_node_zero():
    env = make_env()

    edge_inputs, edge_padding_mask = build_padded_current_edge_inputs(
        env.graph,
        current_index=1,
        k_size=4,
        padding_node_index=PADDING_NODE_INDEX,
    )

    assert edge_inputs.tolist() == [1, 0, 3, PADDING_NODE_INDEX]
    assert edge_padding_mask.tolist() == [0, 0, 0, 1]


def test_build_edge_dist_action_inputs_sets_padding_to_zero_and_clips_distances():
    env = make_env()
    edge_inputs = np.array([1, 0, 3, PADDING_NODE_INDEX])
    edge_padding_mask = np.array([0, 0, 0, 1])

    action_inputs = build_edge_dist_action_inputs(
        env,
        current_index=1,
        edge_inputs=edge_inputs,
        edge_padding_mask=edge_padding_mask,
        normalizer=GRAPH_DISTANCE_NORMALIZER,
        max_norm=EDGE_DIST_MAX_NORM,
    )

    assert action_inputs.shape == (4, 1)
    np.testing.assert_allclose(
        action_inputs[:, 0],
        [0.0, 64.0 / GRAPH_DISTANCE_NORMALIZER, EDGE_DIST_MAX_NORM, 0.0],
    )


def test_get_edge_distance_falls_back_to_euclidean_distance_when_edge_length_missing():
    env = make_env()
    env.graph["0"]["2"] = object()

    edge_dist = get_edge_distance(
        env,
        current_index=0,
        next_index=2,
        normalizer=GRAPH_DISTANCE_NORMALIZER,
        max_norm=EDGE_DIST_MAX_NORM,
    )

    assert edge_dist == 128.0 / GRAPH_DISTANCE_NORMALIZER
