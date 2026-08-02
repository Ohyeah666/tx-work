import copy
import os

import imageio
import numpy as np
import torch
from action_features import build_edge_dist_action_inputs, build_padded_current_edge_inputs
from diagnostics import action_probability_overlay
from env import Env
from parameter import *
from replay_schema import *

if USE_MAP_INPUTS:
    from diagnostics import node_feature_norm_overlay, semantic_map_image
    from map_input import build_semantic_map_input


class Worker:
    def __init__(self, meta_agent_id, policy_net, q_net, global_step, device='cuda', greedy=False, save_image=False,
                 save_diagnostics=False):
        self.device = device
        self.greedy = greedy
        self.metaAgentID = meta_agent_id
        self.global_step = global_step
        self.node_padding_size = NODE_PADDING_SIZE
        self.k_size = K_SIZE
        self.save_image = save_image
        self.save_diagnostics = save_diagnostics

        self.env = Env(map_index=self.global_step, k_size=self.k_size, plot=save_image)
        self.local_policy_net = policy_net
        self.local_q_net = q_net

        self.current_node_index = 0
        self.travel_dist = 0
        self.robot_position = self.env.start_position

        self.episode_buffer = []
        self.perf_metrics = dict()
        self.diagnostic_images = None
        for i in range(BUFFER_SIZE):
            self.episode_buffer.append([])

    def build_map_inputs(self):
        if not USE_MAP_INPUTS:
            return None
        map_inputs = build_semantic_map_input(
            self.env.downsampled_belief,
            self.env.frontiers,
            self.robot_position,
            self.env.resolution,
            FRONTIER_HEATMAP_SIGMA,
        )
        return torch.from_numpy(map_inputs).unsqueeze(0).to(self.device)

    def get_observations(self):
        # get observations
        node_coords = copy.deepcopy(self.env.node_coords)
        graph = copy.deepcopy(self.env.graph)
        node_utility = copy.deepcopy(self.env.node_utility)
        guidepost = copy.deepcopy(self.env.guidepost)
        map_inputs = self.build_map_inputs()

        # normalize observations
        node_coords = node_coords / 640
        node_utility = node_utility / 50

        # transfer to node inputs tensor
        n_nodes = node_coords.shape[0]
        node_utility_inputs = node_utility.reshape((n_nodes, 1))
        node_inputs = np.concatenate((node_coords, node_utility_inputs, guidepost), axis=1)
        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)  # (1, node_padding_size+1, 3)

        # padding the number of node to a given node padding size
        assert node_coords.shape[0] < self.node_padding_size
        padding = torch.nn.ZeroPad2d((0, 0, 0, self.node_padding_size - node_coords.shape[0]))
        node_inputs = padding(node_inputs)

        # calculate a mask to padded nodes
        node_padding_mask = torch.zeros((1, 1, node_coords.shape[0]), dtype=torch.int64).to(self.device)
        node_padding = torch.ones((1, 1, self.node_padding_size - node_coords.shape[0]), dtype=torch.int64).to(
            self.device)
        node_padding_mask = torch.cat((node_padding_mask, node_padding), dim=-1)

        # get the node index of the current robot position
        current_node_index = self.env.find_index_from_coords(self.robot_position)
        current_index = torch.tensor([current_node_index]).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,1)

        # prepare the adjacent list as padded edge inputs and the adjacent matrix as the edge mask
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

        edge, edge_padding = build_padded_current_edge_inputs(
            self.env.graph,
            current_node_index,
            self.k_size,
            PADDING_NODE_INDEX,
        )
        action_inputs = build_edge_dist_action_inputs(
            self.env,
            current_node_index,
            edge,
            edge_padding,
            GRAPH_DISTANCE_NORMALIZER,
            EDGE_DIST_MAX_NORM,
            USE_ACTION_FEATURES,
            USE_ACTION_FEATURE_EDGE_DIST,
        )

        edge_inputs = torch.tensor(edge).unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, k_size)
        edge_padding_mask = torch.tensor(edge_padding).unsqueeze(0).unsqueeze(0).to(self.device)
        action_inputs = torch.tensor(action_inputs).unsqueeze(0).to(self.device)

        observations = (
            node_inputs,
            edge_inputs,
            current_index,
            node_padding_mask,
            edge_padding_mask,
            edge_mask,
            map_inputs,
            action_inputs,
        )
        return observations

    def select_node(self, observations):
        node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask, map_inputs, action_inputs = observations
        with torch.no_grad():
            logp_list = self.local_policy_net(node_inputs, edge_inputs, current_index, node_padding_mask,
                                              edge_padding_mask, edge_mask, map_inputs, action_inputs=action_inputs)

        if self.greedy:
            action_index = torch.argmax(logp_list, dim=1).long()
        else:
            action_index = torch.multinomial(logp_list.exp(), 1).long().squeeze(1)

        next_node_index = int(edge_inputs[0, 0, action_index.item()].item())
        if next_node_index == PADDING_NODE_INDEX:
            raise ValueError("policy selected a padded edge")
        next_position = self.env.node_coords[next_node_index]

        return next_position, action_index

    def build_diagnostic_images(self, observations):
        node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask, map_inputs, action_inputs = observations
        with torch.no_grad():
            if USE_MAP_INPUTS:
                logp_list, diagnostics = self.local_policy_net(
                    node_inputs,
                    edge_inputs,
                    current_index,
                    node_padding_mask,
                    edge_padding_mask,
                    edge_mask,
                    map_inputs,
                    action_inputs=action_inputs,
                    return_diagnostics=True,
                )
            else:
                logp_list = self.local_policy_net(
                    node_inputs,
                    edge_inputs,
                    current_index,
                    node_padding_mask,
                    edge_padding_mask,
                    edge_mask,
                    map_inputs,
                    action_inputs=action_inputs,
                )

        n_nodes = self.env.node_coords.shape[0]
        action_probs = logp_list.exp()[0].detach().cpu().numpy()
        edge_indices = edge_inputs[0, 0].detach().cpu().numpy()
        edge_valid = ~edge_padding_mask[0, 0].detach().cpu().numpy().astype(bool)
        edge_indices = edge_indices[edge_valid]
        action_probs = action_probs[edge_valid]
        current_node_index = int(current_index.item())

        images = {
            "action_probability": action_probability_overlay(
                self.env.robot_belief,
                self.env.node_coords,
                current_node_index,
                edge_indices,
                action_probs,
            ),
        }
        if USE_MAP_INPUTS:
            node_feature_norm = diagnostics["node_map_feature_norm"][0, :n_nodes].detach().cpu().numpy()
            map_array = map_inputs[0].detach().cpu().numpy()
            images.update({
                "semantic_map": semantic_map_image(map_array),
                "node_map_feature_norm": node_feature_norm_overlay(
                    self.env.robot_belief,
                    self.env.node_coords,
                    node_feature_norm,
                ),
            })
        return images

    def save_observations(self, observations):
        node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask, map_inputs, action_inputs = observations
        self.episode_buffer[NODE_INPUTS] += copy.deepcopy(node_inputs)
        self.episode_buffer[EDGE_INPUTS] += copy.deepcopy(edge_inputs)
        self.episode_buffer[CURRENT_INDEX] += copy.deepcopy(current_index)
        self.episode_buffer[NODE_PADDING_MASK] += copy.deepcopy(node_padding_mask).bool()
        self.episode_buffer[EDGE_PADDING_MASK] += copy.deepcopy(edge_padding_mask).bool()
        self.episode_buffer[EDGE_MASK] += copy.deepcopy(edge_mask).bool()
        if USE_MAP_INPUTS:
            self.episode_buffer[MAP_INPUTS] += copy.deepcopy(map_inputs)
        else:
            self.episode_buffer[MAP_INPUTS].append(None)
        self.episode_buffer[ACTION_INPUTS] += copy.deepcopy(action_inputs)

    def save_action(self, action_index):
        self.episode_buffer[ACTION] += action_index.unsqueeze(0).unsqueeze(0)

    def save_reward_done(self, reward, done):
        self.episode_buffer[REWARD] += copy.deepcopy(torch.FloatTensor([[[reward]]]).to(self.device))
        self.episode_buffer[DONE] += copy.deepcopy(torch.tensor([[[(int(done))]]]).to(self.device))

    def save_next_observations(self, observations):
        node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask, map_inputs, action_inputs = observations
        self.episode_buffer[NEXT_NODE_INPUTS] += copy.deepcopy(node_inputs)
        self.episode_buffer[NEXT_EDGE_INPUTS] += copy.deepcopy(edge_inputs)
        self.episode_buffer[NEXT_CURRENT_INDEX] += copy.deepcopy(current_index)
        self.episode_buffer[NEXT_NODE_PADDING_MASK] += copy.deepcopy(node_padding_mask).bool()
        self.episode_buffer[NEXT_EDGE_PADDING_MASK] += copy.deepcopy(edge_padding_mask).bool()
        self.episode_buffer[NEXT_EDGE_MASK] += copy.deepcopy(edge_mask).bool()
        if USE_MAP_INPUTS:
            self.episode_buffer[NEXT_MAP_INPUTS] += copy.deepcopy(map_inputs)
        else:
            self.episode_buffer[NEXT_MAP_INPUTS].append(None)
        self.episode_buffer[NEXT_ACTION_INPUTS] += copy.deepcopy(action_inputs)

    def run_episode(self, curr_episode):
        done = False

        observations = self.get_observations()
        for i in range(128):
            self.save_observations(observations)
            if self.save_diagnostics and self.diagnostic_images is None:
                self.diagnostic_images = self.build_diagnostic_images(observations)
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
        if self.diagnostic_images is not None:
            self.perf_metrics['diagnostic_images'] = self.diagnostic_images

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
