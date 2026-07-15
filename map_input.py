import numpy as np
from scipy.ndimage import distance_transform_edt


FREE_VALUE = 255
OBSTACLE_VALUE = 1
UNKNOWN_VALUE = 127
UINT8_MAX = 255


def _to_grid_index(coords, resolution, height, width):
    coords = np.asarray(coords)
    x = np.clip((coords[..., 0] / resolution).astype(int), 0, width - 1)
    y = np.clip((coords[..., 1] / resolution).astype(int), 0, height - 1)
    return x, y


def _build_frontier_heatmap(frontiers, shape, resolution, sigma):
    height, width = shape
    heatmap = np.zeros((height, width), dtype=np.float32)
    if frontiers is None:
        return heatmap

    frontiers = np.asarray(frontiers)
    if frontiers.size == 0:
        return heatmap
    if sigma <= 0:
        raise ValueError("frontier heatmap sigma must be positive")

    frontier_points = frontiers.reshape(-1, 2)
    frontier_x, frontier_y = _to_grid_index(frontier_points, resolution, height, width)
    frontier_binary = np.zeros((height, width), dtype=np.uint8)
    frontier_binary[frontier_y, frontier_x] = 1

    distance = distance_transform_edt(1 - frontier_binary)
    heatmap = np.exp(-distance / sigma).astype(np.float32)
    heatmap[frontier_binary == 1] = 1.0
    return heatmap


def build_semantic_map_input(downsampled_belief, frontiers, robot_position, resolution, frontier_sigma=3.0):
    """Build a 5-channel uint8 semantic map for the map-aware policy networks.

    Channels are free, obstacle, unknown, frontier heatmap, and robot position.
    Binary channels use 0/255. The frontier heatmap is quantized to 0..255.
    """
    belief = np.asarray(downsampled_belief)
    if belief.ndim != 2:
        raise ValueError("downsampled_belief must be a 2D array")
    if resolution <= 0:
        raise ValueError("resolution must be positive")

    height, width = belief.shape
    semantic_map = np.zeros((5, height, width), dtype=np.uint8)

    # 五通道语义图：显式提供可通行、障碍、未知、frontier heatmap 和当前位置。
    semantic_map[0] = (belief == FREE_VALUE).astype(np.uint8) * UINT8_MAX
    semantic_map[1] = (belief == OBSTACLE_VALUE).astype(np.uint8) * UINT8_MAX
    semantic_map[2] = (belief == UNKNOWN_VALUE).astype(np.uint8) * UINT8_MAX
    frontier_heatmap = _build_frontier_heatmap(frontiers, belief.shape, resolution, frontier_sigma)
    semantic_map[3] = np.round(np.clip(frontier_heatmap, 0.0, 1.0) * UINT8_MAX).astype(np.uint8)

    if robot_position is not None:
        position_x, position_y = _to_grid_index(np.asarray(robot_position).reshape(1, 2), resolution, height, width)
        semantic_map[4, position_y[0], position_x[0]] = UINT8_MAX

    return semantic_map
