import imageio
import csv
import os
import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
from basin_features import build_basin_action_features
from env import Env
from model import PolicyNet
from observation_features import build_node_inputs
from test_parameter import *


class TestWorker:
    def __init__(self, meta_agent_id, policy_net, global_step, device='cuda', greedy=False, save_image=False,
                 gifs_dir=None, test_set_name=TEST_SET_NAME):
        self.device = device
        self.greedy = greedy
        self.metaAgentID = meta_agent_id
        self.global_step = global_step
        self.k_size = K_SIZE
        self.save_image = save_image
        self.gifs_path = gifs_dir or gifs_path

        self.env = Env(map_index=self.global_step, k_size=self.k_size, plot=save_image, test=True,
                       test_set_name=test_set_name)
        self.local_policy_net = policy_net
        self.travel_dist = 0
        self.robot_position = self.env.start_position
        self.perf_metrics = dict()

    def run_episode(self, curr_episode):
        done = False

        observations = self.get_observations()
        for i in range(128):
            next_position, action_index = self.select_node(observations)

            reward, done, self.robot_position, self.travel_dist = self.env.step(self.robot_position, next_position,
                                                                                self.travel_dist)

            observations = self.get_observations()

            # save evaluation data
            if SAVE_TRAJECTORY:
                if not os.path.exists(trajectory_path):
                    os.makedirs(trajectory_path)
                csv_filename = f'results/trajectory/ours_trajectory_result.csv'
                new_file = False if os.path.exists(csv_filename) else True
                field_names = ['dist', 'area']
                with open(csv_filename, 'a') as csvfile:
                    writer = csv.writer(csvfile)
                    if new_file:
                        writer.writerow(field_names)
                    csv_data = np.array([self.travel_dist, np.sum(self.env.robot_belief == 255)]).reshape(1, -1)
                    writer.writerows(csv_data)

            # save a frame
            if self.save_image:
                os.makedirs(self.gifs_path, exist_ok=True)
                self.env.plot_env(self.global_step, self.gifs_path, i, self.travel_dist)

            if done:
                break

        self.perf_metrics['travel_dist'] = self.travel_dist
        self.perf_metrics['explored_rate'] = self.env.explored_rate
        self.perf_metrics['success_rate'] = done

        # save final path length
        if SAVE_LENGTH:
            if not os.path.exists(length_path):
                os.makedirs(length_path)
            csv_filename = f'results/length/ours_length_result.csv'
            new_file = False if os.path.exists(csv_filename) else True
            field_names = ['dist']
            with open(csv_filename, 'a') as csvfile:
                writer = csv.writer(csvfile)
                if new_file:
                    writer.writerow(field_names)
                csv_data = np.array([self.travel_dist]).reshape(-1,1)
                writer.writerows(csv_data)

        # save gif
        if self.save_image:
            self.make_gif(self.gifs_path, curr_episode)

    def get_observations(self):
        # get observations
        node_coords = copy.deepcopy(self.env.node_coords)
        graph = copy.deepcopy(self.env.graph)
        node_utility = copy.deepcopy(self.env.node_utility)
        guidepost = copy.deepcopy(self.env.guidepost)
        visit_count = copy.deepcopy(self.env.visit_count)
        expected_unknown_gain = copy.deepcopy(self.env.node_expected_unknown_gain)
        frontier_cluster_size = copy.deepcopy(self.env.node_frontier_cluster_size)

        # get the node index of the current robot position
        current_node_index = self.env.find_index_from_coords(self.robot_position)
        graph_dist_to_current, reachable_nodes, first_hop = self.env.graph_generator.get_normalized_shortest_path_distances(
            current_node_index, return_first_hop=True)

        # transfer to node inputs tensor
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
        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)  # (1, node_size, 9)

        # calculate a mask for padded node
        node_padding_mask = None

        current_index = torch.tensor([current_node_index]).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,1)

        # prepare the adjacent list as padded edge inputs and the adjacent matrix as the edge mask
        graph = list(graph.values())
        edge_inputs = []
        for node in graph:
            node_edges = list(map(int, node))
            edge_inputs.append(node_edges)

        adjacent_matrix = self.calculate_edge_mask(edge_inputs)
        edge_mask = torch.from_numpy(adjacent_matrix).float().unsqueeze(0).to(self.device)

        edge = list(edge_inputs[current_node_index])
        while len(edge) < self.k_size:
            edge.append(0)

        edge_array = np.array(edge, dtype=int)
        edge_inputs = torch.tensor(edge_array).unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, k_size)

        edge_padding_mask = torch.zeros((1, 1, K_SIZE), dtype=torch.int64).to(self.device)
        one = torch.ones_like(edge_padding_mask, dtype=torch.int64).to(self.device)
        edge_padding_mask = torch.where(edge_inputs == 0, one, edge_padding_mask)

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
            logp_list = self.local_policy_net(
                node_inputs,
                edge_inputs,
                current_index,
                node_padding_mask,
                edge_padding_mask,
                edge_mask,
                action_features,
            )

        if self.greedy:
            action_index = torch.argmax(logp_list, dim=1).long()
        else:
            action_index = torch.multinomial(logp_list.exp(), 1).long().squeeze(1)

        next_node_index = edge_inputs[0, 0, action_index.item()]
        next_position = self.env.node_coords[next_node_index]

        return next_position, action_index

    def calculate_edge_mask(self, edge_inputs):
        size = len(edge_inputs)
        bias_matrix = np.ones((size, size))
        for i in range(size):
            for j in range(size):
                if j in edge_inputs[i]:
                    bias_matrix[i][j] = 0
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

    def work(self, curr_episode):
        self.run_episode(curr_episode)
