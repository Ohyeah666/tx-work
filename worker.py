import copy
import os

import imageio
import numpy as np
import torch
from env import Env
from node_features import build_node_and_action_features
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

        self.env = Env(map_index=self.global_step, k_size=self.k_size, plot=save_image)
        self.local_policy_net = policy_net
        self.local_q_net = q_net

        self.current_node_index = 0
        self.travel_dist = 0
        self.robot_position = self.env.start_position

        self.episode_buffer = []
        self.perf_metrics = dict()
        for i in range(17):
            self.episode_buffer.append([])

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

        n_nodes = node_inputs_np.shape[0]
        node_inputs = torch.FloatTensor(node_inputs_np).unsqueeze(0).to(self.device)

        # padding the number of node to a given node padding size
        assert n_nodes <= self.node_padding_size
        padding = torch.nn.ZeroPad2d((0, 0, 0, self.node_padding_size - n_nodes))
        node_inputs = padding(node_inputs)

        # calculate a mask to padded nodes
        node_padding_mask = torch.zeros((1, 1, n_nodes), dtype=torch.int64).to(self.device)
        node_padding = torch.ones((1, 1, self.node_padding_size - n_nodes), dtype=torch.int64).to(
            self.device)
        node_padding_mask = torch.cat((node_padding_mask, node_padding), dim=-1)

        current_index = torch.tensor([features.current_index]).unsqueeze(0).unsqueeze(0).to(self.device)
        edge_mask = torch.from_numpy(features.edge_mask).float().unsqueeze(0).to(self.device)

        # padding edge mask
        assert n_nodes <= self.node_padding_size
        padding = torch.nn.ConstantPad2d(
            (0, self.node_padding_size - n_nodes, 0, self.node_padding_size - n_nodes), 1)
        edge_mask = padding(edge_mask)

        edge_inputs = torch.tensor(features.edge_inputs, dtype=torch.long).unsqueeze(0).unsqueeze(0).to(self.device)
        action_inputs = torch.FloatTensor(features.action_inputs).unsqueeze(0).to(self.device)
        edge_padding_mask = torch.tensor(features.edge_padding_mask, dtype=torch.int64).unsqueeze(0).unsqueeze(0).to(
            self.device)

        observations = node_inputs, edge_inputs, action_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask
        return observations

    def select_node(self, observations):
        node_inputs, edge_inputs, action_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask = observations
        with torch.no_grad():
            logp_list = self.local_policy_net(node_inputs, edge_inputs, action_inputs, current_index, node_padding_mask,
                                              edge_padding_mask, edge_mask)

        if self.greedy:
            action_index = torch.argmax(logp_list, dim=1).long()
        else:
            action_index = torch.multinomial(logp_list.exp(), 1).long().squeeze(1)

        next_node_index = int(edge_inputs[0, 0, action_index.item()].item())
        if next_node_index == PADDING_NODE_INDEX:
            raise ValueError('policy selected a padded edge')
        next_position = self.env.node_coords[next_node_index]

        return next_position, action_index

    def save_observations(self, observations):
        node_inputs, edge_inputs, action_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask = observations
        self.episode_buffer[0] += copy.deepcopy(node_inputs)
        self.episode_buffer[1] += copy.deepcopy(edge_inputs)
        self.episode_buffer[2] += copy.deepcopy(action_inputs)
        self.episode_buffer[3] += copy.deepcopy(current_index)
        self.episode_buffer[4] += copy.deepcopy(node_padding_mask).bool()
        self.episode_buffer[5] += copy.deepcopy(edge_padding_mask).bool()
        self.episode_buffer[6] += copy.deepcopy(edge_mask).bool()

    def save_action(self, action_index):
        self.episode_buffer[7] += action_index.unsqueeze(0).unsqueeze(0)

    def save_reward_done(self, reward, done):
        self.episode_buffer[8] += copy.deepcopy(torch.FloatTensor([[[reward]]]).to(self.device))
        self.episode_buffer[9] += copy.deepcopy(torch.tensor([[[(int(done))]]]).to(self.device))

    def save_next_observations(self, observations):
        node_inputs, edge_inputs, action_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask = observations
        self.episode_buffer[10] += copy.deepcopy(node_inputs)
        self.episode_buffer[11] += copy.deepcopy(edge_inputs)
        self.episode_buffer[12] += copy.deepcopy(action_inputs)
        self.episode_buffer[13] += copy.deepcopy(current_index)
        self.episode_buffer[14] += copy.deepcopy(node_padding_mask).bool()
        self.episode_buffer[15] += copy.deepcopy(edge_padding_mask).bool()
        self.episode_buffer[16] += copy.deepcopy(edge_mask).bool()

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

        # save gif
        if self.save_image:
            path = gifs_path
            self.make_gif(path, curr_episode)

    def work(self, currEpisode):
        self.run_episode(currEpisode)

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
