import numpy as np
from sklearn.neighbors import NearestNeighbors
import copy
import heapq

from node import Node
from graph import Graph, a_star
from timing_profiler import TimingProfiler
from feature_extractor import (
    compute_expected_unknown_gain_for_nodes,
    compute_frontier_cluster_lookup,
    compute_frontier_cluster_size_for_nodes,
    normalize_expected_unknown_gain,
    normalize_frontier_cluster_size,
)


class Graph_generator:
    GRAPH_DISTANCE_NORMALIZER = 640
    UNREACHABLE_GRAPH_DISTANCE = 2.0

    def __init__(self, map_size, k_size, sensor_range, frontier_resolution=4, plot=False,
                 expected_unknown_gain_update_mode="local",
                 expected_unknown_gain_local_radius_factor=2.0,
                 expected_unknown_gain_ray_sample_count=0,
                 timing_profiler=None,
                 enable_timing_profiler=False,
                 timing_profiler_print_every=0,
                 timing_profiler_prefix="graph"):
        if expected_unknown_gain_update_mode not in ("full", "local"):
            raise ValueError("expected_unknown_gain_update_mode must be 'full' or 'local'")

        self.k_size = k_size
        self.graph = Graph()
        self.node_coords = None
        self.plot = plot
        self.x = []
        self.y = []
        self.map_x = map_size[1]
        self.map_y = map_size[0]
        self.uniform_points = self.generate_uniform_points()
        self.sensor_range = sensor_range
        self.frontier_resolution = frontier_resolution
        self.route_node = []
        self.nodes_list = []
        self.node_utility = None
        self.guidepost = None
        self.visit_count = None
        self.node_expected_unknown_gain = None
        self.node_expected_unknown_gain_raw = None
        self.node_frontier_cluster_size = None
        self.expected_unknown_gain_update_mode = expected_unknown_gain_update_mode
        self.expected_unknown_gain_local_radius_factor = expected_unknown_gain_local_radius_factor
        self.expected_unknown_gain_ray_sample_count = expected_unknown_gain_ray_sample_count
        self.last_expected_unknown_gain_recompute_count = 0
        self.profiler = timing_profiler or TimingProfiler(
            enabled=enable_timing_profiler,
            print_every=timing_profiler_print_every,
            prefix=timing_profiler_prefix,
        )

    def edge_clear_all_nodes(self):
        self.graph = Graph()
        self.x = []
        self.y = []

    def edge_clear(self, coords):
        node_index = str(self.find_index_from_coords(self.node_coords, coords))
        self.graph.clear_edge(node_index)

    def generate_graph(self, robot_location, robot_belief, frontiers):
        # get node_coords by finding the uniform points in free area
        with self.profiler.section("generate.free_nodes"):
            free_area = self.free_area(robot_belief)
            free_area_to_check = free_area[:, 0] + free_area[:, 1] * 1j
            uniform_points_to_check = self.uniform_points[:, 0] + self.uniform_points[:, 1] * 1j
            _, _, candidate_indices = np.intersect1d(
                free_area_to_check,
                uniform_points_to_check,
                return_indices=True,
            )
            node_coords = self.uniform_points[candidate_indices]

            # add robot location as one node coords
            node_coords = np.concatenate((robot_location.reshape(1, 2), node_coords))
            self.node_coords = self.unique_coords(node_coords).reshape(-1, 2)

        # generate the collision free graph
        with self.profiler.section("generate.graph_edges"):
            self.find_k_neighbor_all_nodes(self.node_coords, robot_belief)

        # calculate the utility as the number of observable frontiers of each node
        # save the observable frontiers to be reused
        with self.profiler.section("generate.node_utility"):
            self.node_utility = []
            for coords in self.node_coords:
                node = Node(coords, frontiers, robot_belief)
                self.nodes_list.append(node)
                utility = node.utility
                self.node_utility.append(utility)
            self.node_utility = np.array(self.node_utility)

        with self.profiler.section("generate.semantic_features"):
            self.update_semantic_features(robot_belief, frontiers)

        with self.profiler.section("generate.visit_info"):
            self.update_visit_info()

        self.profiler.maybe_print("generate_graph")

        return (self.node_coords, self.graph.edges, self.node_utility, self.guidepost, self.visit_count,
                self.node_expected_unknown_gain, self.node_frontier_cluster_size)

    def update_graph(self, robot_position, robot_belief, old_robot_belief, frontiers, old_frontiers):
        # add uniform points in the new free area to the node coords
        with self.profiler.section("update.new_nodes"):
            new_free_area = self.free_area((robot_belief - old_robot_belief > 0) * 255)
            free_area_to_check = new_free_area[:, 0] + new_free_area[:, 1] * 1j
            uniform_points_to_check = self.uniform_points[:, 0] + self.uniform_points[:, 1] * 1j
            _, _, candidate_indices = np.intersect1d(
                free_area_to_check,
                uniform_points_to_check,
                return_indices=True,
            )
            new_node_coords = self.uniform_points[candidate_indices]
            old_node_coords = copy.deepcopy(self.node_coords)
            self.node_coords = np.concatenate((self.node_coords, new_node_coords))
            old_node_count = old_node_coords.shape[0]

        # update the collision free graph
        # for coords in new_node_coords:
        #     self.find_k_neighbor(coords, self.node_coords, robot_belief)
        # dist_to_robot = np.linalg.norm(robot_position - old_node_coords, axis=1)
        # nearby_node_indices = np.argwhere(dist_to_robot <= 160)[:, 0].tolist()
        # for index in nearby_node_indices:
        #     coords = old_node_coords[index]
        #     self.edge_clear(coords)
        #     self.find_k_neighbor(coords, self.node_coords, robot_belief)

        with self.profiler.section("update.graph_edges"):
            self.edge_clear_all_nodes()
            self.find_k_neighbor_all_nodes(self.node_coords, robot_belief)

        # update the observable frontiers through the change of frontiers
        with self.profiler.section("update.node_frontiers"):
            old_frontiers_to_check = old_frontiers[:, 0] + old_frontiers[:, 1] * 1j
            new_frontiers_to_check = frontiers[:, 0] + frontiers[:, 1] * 1j
            observed_frontiers_index = np.where(
                np.isin(old_frontiers_to_check, new_frontiers_to_check, assume_unique=True) == False)
            new_frontiers_index = np.where(
                np.isin(new_frontiers_to_check, old_frontiers_to_check, assume_unique=True) == False)
            observed_frontiers = old_frontiers[observed_frontiers_index]
            new_frontiers = frontiers[new_frontiers_index]
            for node in self.nodes_list:
                if np.linalg.norm(node.coords - robot_position) > 2 * self.sensor_range:
                    pass
                elif node.zero_utility_node is True:
                    pass
                else:
                    node.update_observable_frontiers(observed_frontiers, new_frontiers, robot_belief)

            for new_coords in new_node_coords:
                node = Node(new_coords, frontiers, robot_belief)
                self.nodes_list.append(node)

        with self.profiler.section("update.node_utility"):
            self.node_utility = []
            for i, coords in enumerate(self.node_coords):
                utility = self.nodes_list[i].utility
                self.node_utility.append(utility)
            self.node_utility = np.array(self.node_utility)

        with self.profiler.section("update.semantic_features"):
            self.update_semantic_features(
                robot_belief,
                frontiers,
                robot_position=robot_position,
                old_node_count=old_node_count,
            )

        with self.profiler.section("update.visit_info"):
            self.update_visit_info()

        self.profiler.maybe_print("update_graph")

        return (self.node_coords, self.graph.edges, self.node_utility, self.guidepost, self.visit_count,
                self.node_expected_unknown_gain, self.node_frontier_cluster_size)

    def coords_to_key(self, coords):
        return int(round(coords[0])), int(round(coords[1]))

    def update_visit_info(self):
        # route_node records the episode-local visit history, including repeated revisits.
        self.guidepost = np.zeros((self.node_coords.shape[0], 1))
        self.visit_count = np.zeros((self.node_coords.shape[0], 1))
        coord_to_index = {self.coords_to_key(coords): i for i, coords in enumerate(self.node_coords)}

        for node in self.route_node:
            index = coord_to_index.get(self.coords_to_key(node))
            if index is None:
                continue
            self.guidepost[index] = 1
            self.visit_count[index] += 1

    def get_expected_unknown_gain_recompute_indices(self, robot_position, old_node_count):
        n_nodes = self.node_coords.shape[0]
        if old_node_count is None or old_node_count > n_nodes:
            return np.arange(n_nodes, dtype=int)

        recompute_mask = np.zeros(n_nodes, dtype=bool)
        recompute_mask[old_node_count:] = True

        if old_node_count > 0:
            local_radius = self.expected_unknown_gain_local_radius_factor * self.sensor_range
            dist_to_robot = np.linalg.norm(self.node_coords[:old_node_count] - robot_position, axis=1)
            recompute_mask[:old_node_count] = dist_to_robot <= local_radius

        return np.flatnonzero(recompute_mask)

    def update_expected_unknown_gain_cache(self, robot_belief, recompute_indices=None):
        n_nodes = self.node_coords.shape[0]
        needs_full_recompute = (
            recompute_indices is None or
            self.node_expected_unknown_gain_raw is None or
            self.node_expected_unknown_gain_raw.shape[0] > n_nodes
        )

        if needs_full_recompute:
            self.last_expected_unknown_gain_recompute_count = n_nodes
            self.node_expected_unknown_gain_raw = compute_expected_unknown_gain_for_nodes(
                self.node_coords,
                robot_belief,
                self.sensor_range,
                ray_sample_count=self.expected_unknown_gain_ray_sample_count,
            )
            return self.node_expected_unknown_gain_raw

        expected_unknown_gain = np.zeros((n_nodes, 1), dtype=float)
        old_cache_size = self.node_expected_unknown_gain_raw.shape[0]
        expected_unknown_gain[:old_cache_size] = self.node_expected_unknown_gain_raw

        recompute_indices = np.asarray(recompute_indices, dtype=int)
        if old_cache_size < n_nodes:
            new_indices = np.arange(old_cache_size, n_nodes, dtype=int)
            recompute_indices = np.unique(np.concatenate((recompute_indices, new_indices)))

        if recompute_indices.size > 0:
            expected_unknown_gain[recompute_indices] = compute_expected_unknown_gain_for_nodes(
                self.node_coords[recompute_indices],
                robot_belief,
                self.sensor_range,
                ray_sample_count=self.expected_unknown_gain_ray_sample_count,
            )

        self.last_expected_unknown_gain_recompute_count = int(recompute_indices.size)
        self.node_expected_unknown_gain_raw = expected_unknown_gain
        return expected_unknown_gain

    def update_semantic_features(self, robot_belief, frontiers, robot_position=None, old_node_count=None):
        recompute_indices = None
        if (
            self.expected_unknown_gain_update_mode == "local" and
            robot_position is not None and
            self.node_expected_unknown_gain_raw is not None
        ):
            recompute_indices = self.get_expected_unknown_gain_recompute_indices(
                robot_position,
                old_node_count,
            )

        with self.profiler.section("semantic.expected_unknown_gain"):
            expected_unknown_gain = self.update_expected_unknown_gain_cache(
                robot_belief,
                recompute_indices,
            )

        self.node_expected_unknown_gain = normalize_expected_unknown_gain(
            expected_unknown_gain,
            self.sensor_range,
        )

        with self.profiler.section("semantic.frontier_cluster_lookup"):
            frontier_cluster_lookup = compute_frontier_cluster_lookup(
                frontiers,
                resolution=self.frontier_resolution,
                connectivity=8,
            )

        with self.profiler.section("semantic.frontier_cluster_size"):
            frontier_cluster_size = compute_frontier_cluster_size_for_nodes(
                self.nodes_list,
                frontier_cluster_lookup,
            )

        self.node_frontier_cluster_size = normalize_frontier_cluster_size(frontier_cluster_size, normalizer=50)

    def get_normalized_shortest_path_distances(self, start_index, return_first_hop=False):
        n_nodes = self.node_coords.shape[0]
        distances = np.full(n_nodes, np.inf)
        distances[start_index] = 0
        first_hop = np.full(n_nodes, -1, dtype=int)
        first_hop[start_index] = start_index

        adjacency = [[] for _ in range(n_nodes)]
        for from_node, edges in self.graph.edges.items():
            from_index = int(from_node)
            if from_index >= n_nodes:
                continue

            for edge in edges.values():
                to_index = int(edge.to_node)
                if to_index >= n_nodes:
                    continue

                length = float(edge.length)
                adjacency[from_index].append((to_index, length))
                adjacency[to_index].append((from_index, length))

        for edges in adjacency:
            edges.sort(key=lambda item: item[0])

        open_list = [(0, -1, start_index)]
        while open_list:
            current_dist, _, current_index = heapq.heappop(open_list)
            if current_dist > distances[current_index]:
                continue

            for next_index, edge_length in adjacency[current_index]:
                next_dist = current_dist + edge_length
                if current_index == start_index:
                    candidate_first_hop = next_index
                else:
                    candidate_first_hop = first_hop[current_index]

                if candidate_first_hop < 0:
                    continue

                is_shorter = next_dist < distances[next_index] - 1e-9
                is_tie_with_smaller_first_hop = (
                    abs(next_dist - distances[next_index]) <= 1e-9 and
                    (first_hop[next_index] < 0 or candidate_first_hop < first_hop[next_index])
                )
                if is_shorter or is_tie_with_smaller_first_hop:
                    distances[next_index] = next_dist
                    first_hop[next_index] = candidate_first_hop
                    heapq.heappush(open_list, (next_dist, candidate_first_hop, next_index))

        reachable = np.isfinite(distances)
        normalized_distances = distances / self.GRAPH_DISTANCE_NORMALIZER
        normalized_distances[~reachable] = self.UNREACHABLE_GRAPH_DISTANCE

        if return_first_hop:
            first_hop[~reachable] = -1
            return normalized_distances.reshape(n_nodes, 1), reachable.reshape(n_nodes, 1), first_hop.reshape(n_nodes, 1)

        return normalized_distances.reshape(n_nodes, 1), reachable.reshape(n_nodes, 1)

    def generate_uniform_points(self):
        x = np.linspace(0, self.map_x - 1, 30).round().astype(int)
        y = np.linspace(0, self.map_y - 1, 30).round().astype(int)
        t1, t2 = np.meshgrid(x, y)
        points = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
        return points

    def free_area(self, robot_belief):
        index = np.where(robot_belief == 255)
        free = np.asarray([index[1], index[0]]).T
        return free

    def unique_coords(self, coords):
        x = coords[:, 0] + coords[:, 1] * 1j
        indices = np.unique(x, return_index=True)[1]
        coords = np.array([coords[idx] for idx in sorted(indices)])
        return coords

    def find_k_neighbor(self, coords, node_coords, robot_belief):
        dist_list = np.linalg.norm((coords - node_coords), axis=-1)
        sorted_index = np.argsort(dist_list)
        k = 0
        neighbor_index_list = []
        while k < self.k_size and k < node_coords.shape[0]:
            neighbor_index = sorted_index[k]
            neighbor_index_list.append(neighbor_index)
            start = coords
            end = node_coords[neighbor_index]
            if not self.check_collision(start, end, robot_belief):
                a = str(self.find_index_from_coords(node_coords, start))
                b = str(neighbor_index)
                dist = np.linalg.norm(start - end)
                self.graph.add_node(a)
                self.graph.add_edge(a, b, dist)

            k += 1
        return neighbor_index_list

    def find_k_neighbor_all_nodes(self, node_coords, robot_belief):
        X = node_coords
        if len(node_coords) >= self.k_size:
            knn = NearestNeighbors(n_neighbors=self.k_size)
        else:
            knn = NearestNeighbors(n_neighbors=len(node_coords))
        knn.fit(X)
        distances, indices = knn.kneighbors(X)

        for i, p in enumerate(X):
            from_index = i
            for j, neighbour_index in enumerate(indices[i][:]):
                start = p
                end = X[neighbour_index]
                if not self.check_collision(start, end, robot_belief):
                    a = str(from_index)
                    b = str(int(neighbour_index))
                    self.graph.add_node(a)
                    self.graph.add_edge(a, b, distances[i, j])

                    if self.plot:
                        self.x.append([p[0], end[0]])
                        self.y.append([p[1], end[1]])

    def find_index_from_coords(self, node_coords, p):
        return np.where(np.linalg.norm(node_coords - p, axis=1) < 1e-5)[0][0]

    def check_collision(self, start, end, robot_belief):
        # Bresenham line algorithm checking
        collision = False

        x0 = start[0].round()
        y0 = start[1].round()
        x1 = end[0].round()
        y1 = end[1].round()
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        x, y = x0, y0
        error = dx - dy
        x_inc = 1 if x1 > x0 else -1
        y_inc = 1 if y1 > y0 else -1
        dx *= 2
        dy *= 2

        while 0 <= x < robot_belief.shape[1] and 0 <= y < robot_belief.shape[0]:
            k = robot_belief.item(int(y), int(x))
            if x == x1 and y == y1:
                break
            if k == 1:
                collision = True
                break
            if k == 127:
                collision = True
                break
            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx
        return collision

    def find_shortest_path(self, current, destination, node_coords):
        start_node = str(self.find_index_from_coords(node_coords, current))
        end_node = str(self.find_index_from_coords(node_coords, destination))
        route, dist = a_star(int(start_node), int(end_node), self.node_coords, self.graph)
        if start_node != end_node:
            assert route != []
        route = list(map(str, route))
        return dist, route
