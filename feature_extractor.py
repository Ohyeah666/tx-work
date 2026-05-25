import math
from functools import lru_cache
from collections import deque

import numpy as np


OCCUPIED = 1
UNKNOWN = 127


def coords_to_key(coords):
    return int(round(coords[0])), int(round(coords[1]))


def bresenham_cells(start, end):
    x0 = int(round(start[0]))
    y0 = int(round(start[1]))
    x1 = int(round(end[0]))
    y1 = int(round(end[1]))

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x_step = 1 if x0 < x1 else -1
    y_step = 1 if y0 < y1 else -1
    error = dx - dy

    x = x0
    y = y0
    while True:
        yield x, y
        if x == x1 and y == y1:
            break

        doubled_error = 2 * error
        if doubled_error > -dy:
            error -= dy
            x += x_step
        if doubled_error < dx:
            error += dx
            y += y_step


def line_has_occupied(start, end, robot_belief, occupied_value=OCCUPIED):
    for x, y in bresenham_cells(start, end):
        if not (0 <= x < robot_belief.shape[1] and 0 <= y < robot_belief.shape[0]):
            return True
        if robot_belief[y, x] == occupied_value:
            return True
    return False


@lru_cache(maxsize=16)
def sensor_perimeter_offsets(sensor_range):
    radius = int(math.ceil(sensor_range))
    offsets = []
    for dx in range(-radius, radius + 1):
        offsets.append((dx, -radius))
        offsets.append((dx, radius))
    for dy in range(-radius + 1, radius):
        offsets.append((-radius, dy))
        offsets.append((radius, dy))
    return tuple(dict.fromkeys(offsets))


def compute_expected_unknown_gain_for_nodes(node_coords, robot_belief, sensor_range,
                                            occupied_value=OCCUPIED, unknown_value=UNKNOWN):
    gains = np.zeros((len(node_coords), 1), dtype=float)
    if len(node_coords) == 0:
        return gains

    radius = int(math.ceil(sensor_range))
    radius_sq = sensor_range ** 2
    height, width = robot_belief.shape
    offsets = sensor_perimeter_offsets(radius)

    for i, coords in enumerate(node_coords):
        x0 = int(round(coords[0]))
        y0 = int(round(coords[1]))
        visible_unknown = set()

        for dx, dy in offsets:
            end = (x0 + dx, y0 + dy)
            for x, y in bresenham_cells((x0, y0), end):
                if not (0 <= x < width and 0 <= y < height):
                    break

                rel_x = x - x0
                rel_y = y - y0
                if rel_x * rel_x + rel_y * rel_y > radius_sq:
                    break

                cell_value = robot_belief[y, x]
                if cell_value == occupied_value:
                    break
                if cell_value == unknown_value:
                    visible_unknown.add(y * width + x)

        gains[i, 0] = len(visible_unknown)

    return gains


def normalize_expected_unknown_gain(expected_unknown_gain, sensor_range):
    normalizer = math.pi * (sensor_range ** 2)
    if normalizer <= 0:
        return np.zeros_like(expected_unknown_gain, dtype=float)
    return np.clip(expected_unknown_gain / normalizer, 0, 1)


def compute_frontier_cluster_lookup(frontiers, resolution, connectivity=8):
    if frontiers is None or len(frontiers) == 0:
        return {}
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")

    frontier_keys = {coords_to_key(frontier) for frontier in frontiers}
    if connectivity == 4:
        neighbor_offsets = ((resolution, 0), (-resolution, 0), (0, resolution), (0, -resolution))
    else:
        neighbor_offsets = tuple(
            (dx * resolution, dy * resolution)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            if not (dx == 0 and dy == 0)
        )

    unvisited = set(frontier_keys)
    cluster_size_by_key = {}

    while unvisited:
        start = unvisited.pop()
        component = [start]
        queue = deque([start])

        while queue:
            x, y = queue.popleft()
            for dx, dy in neighbor_offsets:
                neighbor = (x + dx, y + dy)
                if neighbor in unvisited:
                    unvisited.remove(neighbor)
                    component.append(neighbor)
                    queue.append(neighbor)

        component_size = len(component)
        for key in component:
            cluster_size_by_key[key] = component_size

    return cluster_size_by_key


def compute_frontier_cluster_size_for_nodes(nodes_list, frontier_cluster_lookup):
    cluster_sizes = np.zeros((len(nodes_list), 1), dtype=float)
    if not frontier_cluster_lookup:
        return cluster_sizes

    for i, node in enumerate(nodes_list):
        observable_frontiers = getattr(node, "observable_frontiers", [])
        max_cluster_size = 0
        for frontier in observable_frontiers:
            max_cluster_size = max(max_cluster_size, frontier_cluster_lookup.get(coords_to_key(frontier), 0))
        cluster_sizes[i, 0] = max_cluster_size

    return cluster_sizes


def normalize_frontier_cluster_size(frontier_cluster_size, normalizer=50):
    if normalizer <= 0:
        return np.zeros_like(frontier_cluster_size, dtype=float)
    return np.clip(frontier_cluster_size / normalizer, 0, 1)
