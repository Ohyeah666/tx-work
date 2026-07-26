import numpy as np
import pytest

from map_input import build_semantic_map_input


def test_semantic_map_builds_two_uint8_channels_with_frontier_heatmap():
    belief = np.array(
        [
            [255, 1, 127, 255],
            [127, 255, 1, 127],
            [1, 127, 255, 255],
        ],
        dtype=np.uint8,
    )
    frontiers = np.array([[4, 4], [100, -4]])
    robot_position = np.array([100, 100])

    semantic_map = build_semantic_map_input(
        belief,
        frontiers,
        robot_position,
        resolution=4,
        frontier_sigma=2.0,
    )

    assert semantic_map.shape == (2, 3, 4)
    assert semantic_map.dtype == np.uint8
    np.testing.assert_array_equal(semantic_map[0], (belief == 127).astype(np.uint8) * 255)

    assert semantic_map[1, 1, 1] == 255
    assert semantic_map[1, 0, 3] == 255
    assert 0 < semantic_map[1, 1, 2] < 255


def test_semantic_map_handles_empty_frontiers_and_validates_inputs():
    belief = np.full((2, 2), 127, dtype=np.uint8)

    semantic_map = build_semantic_map_input(
        belief,
        frontiers=np.empty((0, 2)),
        robot_position=None,
        resolution=4,
        frontier_sigma=1.0,
    )

    assert semantic_map.shape == (2, 2, 2)
    assert semantic_map[1].sum() == 0

    with pytest.raises(ValueError, match="downsampled_belief"):
        build_semantic_map_input(np.zeros((1, 2, 3)), None, None, resolution=4)

    with pytest.raises(ValueError, match="resolution"):
        build_semantic_map_input(belief, None, None, resolution=0)

    with pytest.raises(ValueError, match="sigma"):
        build_semantic_map_input(belief, np.array([[0, 0]]), None, resolution=4, frontier_sigma=0)
