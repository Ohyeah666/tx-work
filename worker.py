import copy
import os

import imageio
import numpy as np
import torch
from basin_features import build_basin_action_features
from env import Env
from observation_features import build_node_inputs
from parameter import *


class Worker:
    def __init__(self, meta_agent_id, policy_net, q_net, global_step, device='cuda', greedy=False, save_image=False):
        self.device = device
        self.greedy = greedy
        self.metaAgentID = meta_agent_id
        self.global_step = global_step
        self.node_padding_size = NODE_PADDING_SIZE
        self.k_size = K_SIZE
        self.save_image = save_image

        self.env = Env(
            map_index=self.global_step,
            k_size=self.k_size,
            plot=save_image,
            expected_unknown_gain_update_mode=EXPECTED_UNKNOWN_GAIN_UPDATE_MODE,
            expected_unknown_gain_local_radius_factor=EXPECTED_UNKNOWN_GAIN_LOCAL_RADIUS_FACTOR,
            expected_unknown_gain_ray_sample_count=EXPECTED_UNKNOWN_GAIN_RAY_SAMPLE_COUNT,
            enable_timing_profiler=ENABLE_TIMING_PROFILER,
            timing_profiler_print_every=TIMING_PROFILER_PRINT_EVERY,
            timing_profiler_prefix=f"train-agent-{self.metaAgentID}",
        )
        self.local_policy_net = policy_net
        self.local_q_net = q_net

        self.current_node_index = 0
        self.travel_dist = 0
        self.robot_position = self.env.start_position

        self.episode_buffer = []
        self.perf_metrics = dict()
        self.selected_expected_unknown_gain = []
        self.selected_frontier_cluster_size = []
        self.selected_basin_utility_sum = []
        self.selected_basin_expected_unknown_gain_sum = []
        self.selected_basin_frontier_cluster_max = []
        self.selected_basin_unvisited_ratio = []
        self.selected_basin_min_dist_to_utility = []
        for i in range(17):
            self.episode_buffer.append([])

    def get_observations(self):
        profiler = self.env.graph_generator.profiler
        # get observations
        with profiler.section("observe.copy_state"):
            node_coords = copy.deepcopy(self.env.node_coords)
            graph = copy.deepcopy(self.env.graph)
            node_utility = copy.deepcopy(self.env.node_utility)
            guidepost = copy.deepcopy(self.env.guidepost)
            visit_count = copy.deepcopy(self.env.visit_count)
            expected_unknown_gain = copy.deepcopy(self.env.node_expected_unknown_gain)
            frontier_cluster_size = copy.deepcopy(self.env.node_frontier_cluster_size)

        # get the node index of the current robot position
        with profiler.section("observe.shortest_path"):
            current_node_index = self.env.find_index_from_coords(self.robot_position)
            graph_dist_to_current, reachable_nodes, first_hop = self.env.graph_generator.get_normalized_shortest_path_distances(
                current_node_index, return_first_hop=True)

        # transfer to node inputs tensor
        with profiler.section("observe.node_inputs"):
            n_nodes = node_coords.shape[0]
            node_inputs = build_node_inputs(
                node_coords,
                node_utility,
                guidepost,
                graph_dist_to_current,
                reachable_nodes,
                visit_count,
                expected_unknown_gain,
                frontier_cluster_size,
                current_node_index,
            )
            node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)  # (1, node_padding_size, 9)

            # padding the number of node to a given node padding size
            assert node_coords.shape[0] < self.node_padding_size
            padding = torch.nn.ZeroPad2d((0, 0, 0, self.node_padding_size - node_coords.shape[0]))
            node_inputs = padding(node_inputs)

            # calculate a mask to padded nodes
            node_padding_mask = torch.zeros((1, 1, node_coords.shape[0]), dtype=torch.int64).to(self.device)
            node_padding = torch.ones((1, 1, self.node_padding_size - node_coords.shape[0]), dtype=torch.int64).to(
                self.device)
            node_padding_mask = torch.cat((node_padding_mask, node_padding), dim=-1)

            current_index = torch.tensor([current_node_index]).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,1)

        # prepare the adjacent list as padded edge inputs and the adjacent matrix as the edge mask
        with profiler.section("observe.edge_inputs"):
            graph = list(graph.values())
            edge_inputs = []
            for node in graph:
                node_edges = list(map(int, node))
                edge_inputs.append(node_edges)

            adjacent_matrix = self.calculate_edge_mask(edge_inputs)
            edge_mask = torch.from_numpy(adjacent_matrix).float().unsqueeze(0).to(self.device)

            # padding edge mask
            assert len(edge_inputs) < self.node_padding_size
            padding = torch.nn.ConstantPad2d(
                (0, self.node_padding_size - len(edge_inputs), 0, self.node_padding_size - len(edge_inputs)), 1)
            edge_mask = padding(edge_mask)

            edge = list(edge_inputs[current_node_index])
            while len(edge) < self.k_size:
                edge.append(0)

            edge_array = np.array(edge, dtype=int)
            edge_inputs = torch.tensor(edge_array).unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, k_size)

            # calculate a mask for the padded edges (denoted by 0)
            edge_padding_mask = torch.zeros((1, 1, K_SIZE), dtype=torch.int64).to(self.device)
            one = torch.ones_like(edge_padding_mask, dtype=torch.int64).to(self.device)
            edge_padding_mask = torch.where(edge_inputs == 0, one, edge_padding_mask)

        with profiler.section("observe.basin_features"):
            action_features = build_basin_action_features(
                edge_array,
                first_hop,
                graph_dist_to_current,
                node_utility,
                visit_count,
                expected_unknown_gain,
                frontier_cluster_size,
                current_node_index,
                edge_padding_mask=edge_padding_mask.cpu().numpy().reshape(-1),
                k_size=self.k_size,
                utility_sum_normalizer=BASIN_UTILITY_SUM_NORMALIZER,
                expected_unknown_gain_sum_normalizer=BASIN_EXPECTED_UNKNOWN_GAIN_SUM_NORMALIZER,
            )
            action_features = torch.FloatTensor(action_features).unsqueeze(0).to(self.device)

        observations = node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask, action_features
        return observations

    def select_node(self, observations):
        node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask, action_features = observations
        with torch.no_grad():
            logp_list = self.local_policy_net(node_inputs, edge_inputs, current_index, node_padding_mask,
                                              edge_padding_mask, edge_mask, action_features)

        if self.greedy:
            action_index = torch.argmax(logp_list, dim=1).long()
        else:
            action_index = torch.multinomial(logp_list.exp(), 1).long().squeeze(1)

        next_node_index = edge_inputs[0, 0, action_index.item()]
        next_position = self.env.node_coords[next_node_index]
        self.selected_expected_unknown_gain.append(node_inputs[0, next_node_index, -2].item())
        self.selected_frontier_cluster_size.append(node_inputs[0, next_node_index, -1].item())
        selected_action_features = action_features[0, action_index.item()]
        self.selected_basin_utility_sum.append(selected_action_features[0].item())
        self.selected_basin_expected_unknown_gain_sum.append(selected_action_features[1].item())
        self.selected_basin_frontier_cluster_max.append(selected_action_features[2].item())
        self.selected_basin_unvisited_ratio.append(selected_action_features[3].item())
        self.selected_basin_min_dist_to_utility.append(selected_action_features[4].item())

        return next_position, action_index

    def save_observations(self, observations):
        node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask, action_features = observations
        self.episode_buffer[0] += copy.deepcopy(node_inputs)
        self.episode_buffer[1] += copy.deepcopy(edge_inputs)
        self.episode_buffer[2] += copy.deepcopy(current_index)
        self.episode_buffer[3] += copy.deepcopy(node_padding_mask).bool()
        self.episode_buffer[4] += copy.deepcopy(edge_padding_mask).bool()
        self.episode_buffer[5] += copy.deepcopy(edge_mask).bool()
        self.episode_buffer[6] += copy.deepcopy(action_features)

    def save_action(self, action_index):
        self.episode_buffer[7] += action_index.unsqueeze(0).unsqueeze(0)

    def save_reward_done(self, reward, done):
        self.episode_buffer[8] += copy.deepcopy(torch.FloatTensor([[[reward]]]).to(self.device))
        self.episode_buffer[9] += copy.deepcopy(torch.tensor([[[(int(done))]]]).to(self.device))

    def save_next_observations(self, observations):
        node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask, action_features = observations
        self.episode_buffer[10] += copy.deepcopy(node_inputs)
        self.episode_buffer[11] += copy.deepcopy(edge_inputs)
        self.episode_buffer[12] += copy.deepcopy(current_index)
        self.episode_buffer[13] += copy.deepcopy(node_padding_mask).bool()
        self.episode_buffer[14] += copy.deepcopy(edge_padding_mask).bool()
        self.episode_buffer[15] += copy.deepcopy(edge_mask).bool()
        self.episode_buffer[16] += copy.deepcopy(action_features)

    def run_episode(self, curr_episode):
        done = False

        observations = self.get_observations()
        for i in range(128):
            self.save_observations(observations)
            next_position, action_index = self.select_node(observations)

            self.save_action(action_index)
            reward, done, self.robot_position, self.travel_dist = self.env.step(self.robot_position, next_position, self.travel_dist)
            self.save_reward_done(reward, done)
 
            observations = self.get_observations()
            self.save_next_observations(observations)

            # save a frame
            if self.save_image:
                if not os.path.exists(gifs_path):
                    os.makedirs(gifs_path)
                self.env.plot_env(self.global_step, gifs_path, i, self.travel_dist)

            if done:
                break

        # save metrics
        self.perf_metrics['travel_dist'] = self.travel_dist
        self.perf_metrics['explored_rate'] = self.env.explored_rate
        self.perf_metrics['success_rate'] = done
        self.perf_metrics['selected_expected_unknown_gain'] = float(np.mean(self.selected_expected_unknown_gain)) \
            if self.selected_expected_unknown_gain else 0.0
        self.perf_metrics['selected_frontier_cluster_size'] = float(np.mean(self.selected_frontier_cluster_size)) \
            if self.selected_frontier_cluster_size else 0.0
        self.perf_metrics['selected_basin_utility_sum'] = float(np.mean(self.selected_basin_utility_sum)) \
            if self.selected_basin_utility_sum else 0.0
        self.perf_metrics['selected_basin_expected_unknown_gain_sum'] = float(np.mean(self.selected_basin_expected_unknown_gain_sum)) \
            if self.selected_basin_expected_unknown_gain_sum else 0.0
        self.perf_metrics['selected_basin_frontier_cluster_max'] = float(np.mean(self.selected_basin_frontier_cluster_max)) \
            if self.selected_basin_frontier_cluster_max else 0.0
        self.perf_metrics['selected_basin_unvisited_ratio'] = float(np.mean(self.selected_basin_unvisited_ratio)) \
            if self.selected_basin_unvisited_ratio else 0.0
        self.perf_metrics['selected_basin_min_dist_to_utility'] = float(np.mean(self.selected_basin_min_dist_to_utility)) \
            if self.selected_basin_min_dist_to_utility else 0.0

        # save gif
        if self.save_image:
            path = gifs_path
            self.make_gif(path, curr_episode)

    def work(self, currEpisode):
        self.run_episode(currEpisode)

    def calculate_edge_mask(self, edge_inputs):
        size = len(edge_inputs)
        bias_matrix = np.ones((size, size), dtype=np.float32)
        for i, neighbors in enumerate(edge_inputs):
            neighbor_indices = np.asarray(neighbors, dtype=int)
            neighbor_indices = neighbor_indices[(neighbor_indices >= 0) & (neighbor_indices < size)]
            bias_matrix[i, neighbor_indices] = 0
        return bias_matrix
    
    def make_gif(self, path, n):
        with imageio.get_writer('{}/{}_explored_rate_{:.4g}.gif'.format(path, n, self.env.explored_rate), mode='I', duration=0.5) as writer:
            for frame in self.env.frame_files:
                image = imageio.imread(frame)
                writer.append_data(image)
        print('gif complete\n')

        # Remove files
        for filename in self.env.frame_files[:-1]:
            os.remove(filename)
