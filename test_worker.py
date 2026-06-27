import imageio
import csv
import os
import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
from env import Env
from evaluation_metrics import BacktrackingMetricTracker
from model import PolicyNet
from node_features import build_node_and_action_features
from parameter import TRAJECTORY_MEMORY_GAMMA, TRAJECTORY_MEMORY_SIGMA, TRAJECTORY_MEMORY_WINDOW
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
        self.metric_tracker = BacktrackingMetricTracker()

    def run_episode(self, curr_episode):
        done = False

        observations = self.get_observations()
        self.metric_tracker.reset(int(observations[3].item()))
        for i in range(128):
            previous_position = self.robot_position.copy()
            previous_free_area = np.sum(self.env.robot_belief == 255)
            next_position, action_index = self.select_node(observations)
            edge_inputs = observations[1]
            next_node_index = int(edge_inputs[0, 0, action_index.item()].item())
            next_memory = self.get_metric_memory(next_node_index)

            reward, done, self.robot_position, self.travel_dist = self.env.step(self.robot_position, next_position,
                                                                                self.travel_dist)
            current_free_area = np.sum(self.env.robot_belief == 255)
            step_dist = np.linalg.norm(previous_position - next_position)
            self.metric_tracker.record_step(
                next_index=next_node_index,
                distance=step_dist,
                next_memory=next_memory,
                new_area_gain=current_free_area - previous_free_area,
            )

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
                route_memory = self.env.graph_generator.get_decayed_route_memory(
                    gamma=TRAJECTORY_MEMORY_GAMMA,
                    sigma=TRAJECTORY_MEMORY_SIGMA,
                    window=TRAJECTORY_MEMORY_WINDOW,
                )
                self.env.plot_env(
                    self.global_step,
                    self.gifs_path,
                    i,
                    self.travel_dist,
                    node_color_values=route_memory,
                    node_color_name='memory_i',
                )

            if done:
                break

        self.perf_metrics['travel_dist'] = self.travel_dist
        self.perf_metrics['explored_rate'] = self.env.explored_rate
        self.perf_metrics['success_rate'] = done
        self.perf_metrics.update(self.metric_tracker.compute(self.travel_dist))

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
        features = build_node_and_action_features(self.env, self.robot_position, self.k_size)
        node_inputs_np = features.node_inputs
        if node_inputs_np.shape[1] != INPUT_DIM:
            raise ValueError(f'node_inputs feature dim {node_inputs_np.shape[1]} does not match INPUT_DIM {INPUT_DIM}')
        if features.action_inputs.shape[1] != ACTION_FEATURE_DIM:
            raise ValueError(
                f'action_inputs feature dim {features.action_inputs.shape[1]} '
                f'does not match ACTION_FEATURE_DIM {ACTION_FEATURE_DIM}'
            )
        node_inputs = torch.FloatTensor(node_inputs_np).unsqueeze(0).to(self.device)  # (1, node_size, INPUT_DIM)

        # calculate a mask for padded node
        node_padding_mask = None

        current_index = torch.tensor([features.current_index]).unsqueeze(0).unsqueeze(0).to(self.device)
        edge_mask = torch.from_numpy(features.edge_mask).float().unsqueeze(0).to(self.device)
        edge_inputs = torch.tensor(features.edge_inputs, dtype=torch.long).unsqueeze(0).unsqueeze(0).to(self.device)
        action_inputs = torch.FloatTensor(features.action_inputs).unsqueeze(0).to(self.device)
        edge_padding_mask = torch.tensor(features.edge_padding_mask, dtype=torch.int64).unsqueeze(0).unsqueeze(0).to(
            self.device)

        observations = node_inputs, edge_inputs, action_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask
        return observations

    def select_node(self, observations):
        node_inputs, edge_inputs, action_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask = observations
        with torch.no_grad():
            logp_list = self.local_policy_net(node_inputs, edge_inputs, action_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask)

        if self.greedy:
            action_index = torch.argmax(logp_list, dim=1).long()
        else:
            action_index = torch.multinomial(logp_list.exp(), 1).long().squeeze(1)

        next_node_index = int(edge_inputs[0, 0, action_index.item()].item())
        if next_node_index == PADDING_NODE_INDEX:
            raise ValueError('policy selected a padded edge')
        next_position = self.env.node_coords[next_node_index]

        return next_position, action_index

    def get_metric_memory(self, node_index):
        route_memory = self.env.graph_generator.get_decayed_route_memory(
            gamma=TRAJECTORY_MEMORY_GAMMA,
            sigma=TRAJECTORY_MEMORY_SIGMA,
            window=TRAJECTORY_MEMORY_WINDOW,
        )
        return float(route_memory.reshape(-1)[node_index])

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
