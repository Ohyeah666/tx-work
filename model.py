import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def _compatible_group_count(num_channels, preferred_groups=8):
    groups = min(preferred_groups, num_channels)
    while num_channels % groups != 0:
        groups -= 1
    return groups


# a pointer network layer for policy output
class SingleHeadAttention(nn.Module):
    def __init__(self, embedding_dim):
        super(SingleHeadAttention, self).__init__()
        self.input_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.value_dim = embedding_dim
        self.key_dim = self.value_dim
        self.tanh_clipping = 10
        self.norm_factor = 1 / math.sqrt(self.key_dim)

        self.w_query = nn.Parameter(torch.Tensor(self.input_dim, self.key_dim))
        self.w_key = nn.Parameter(torch.Tensor(self.input_dim, self.key_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, k, mask=None):

        n_batch, n_key, n_dim = k.size()
        n_query = q.size(1)

        k_flat = k.reshape(-1, n_dim)
        q_flat = q.reshape(-1, n_dim)

        shape_k = (n_batch, n_key, -1)
        shape_q = (n_batch, n_query, -1)

        Q = torch.matmul(q_flat, self.w_query).view(shape_q)
        K = torch.matmul(k_flat, self.w_key).view(shape_k)

        U = self.norm_factor * torch.matmul(Q, K.transpose(1, 2))
        U = self.tanh_clipping * torch.tanh(U)

        if mask is not None:
            U = U.masked_fill(mask == 1, -1e8)
        attention = torch.log_softmax(U, dim=-1)  # n_batch*n_query*n_key

        return attention


# standard multi head attention layer
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, n_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.input_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.value_dim = self.embedding_dim // self.n_heads
        self.key_dim = self.value_dim
        self.norm_factor = 1 / math.sqrt(self.key_dim)

        self.w_query = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.key_dim))
        self.w_key = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.key_dim))
        self.w_value = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.value_dim))
        self.w_out = nn.Parameter(torch.Tensor(self.n_heads, self.value_dim, self.embedding_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, k=None, v=None, key_padding_mask=None, attn_mask=None):
        if k is None:
            k = q
        if v is None:
            v = q

        n_batch, n_key, n_dim = k.size()
        n_query = q.size(1)
        n_value = v.size(1)

        k_flat = k.contiguous().view(-1, n_dim)
        v_flat = v.contiguous().view(-1, n_dim)
        q_flat = q.contiguous().view(-1, n_dim)
        shape_v = (self.n_heads, n_batch, n_value, -1)
        shape_k = (self.n_heads, n_batch, n_key, -1)
        shape_q = (self.n_heads, n_batch, n_query, -1)

        Q = torch.matmul(q_flat, self.w_query).view(shape_q)  # n_heads*batch_size*n_query*key_dim
        K = torch.matmul(k_flat, self.w_key).view(shape_k)  # n_heads*batch_size*targets_size*key_dim
        V = torch.matmul(v_flat, self.w_value).view(shape_v)  # n_heads*batch_size*targets_size*value_dim

        U = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))  # n_heads*batch_size*n_query*targets_size

        if attn_mask is not None:
            attn_mask = attn_mask.view(1, n_batch, n_query, n_key).expand_as(U)

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.repeat(1, n_query, 1)
            key_padding_mask = key_padding_mask.view(1, n_batch, n_query, n_key).expand_as(U)  # copy for n_heads times

        if attn_mask is not None and key_padding_mask is not None:
            mask = (attn_mask + key_padding_mask)
        elif attn_mask is not None:
            mask = attn_mask
        elif key_padding_mask is not None:
            mask = key_padding_mask
        else:
            mask = None

        if mask is not None:
            U = U.masked_fill(mask > 0, -1e8)

        attention = torch.softmax(U, dim=-1)  # n_heads*batch_size*n_query*targets_size

        heads = torch.matmul(attention, V)  # n_heads*batch_size*n_query*value_dim

        # out = heads.permute(1, 2, 0, 3).reshape(n_batch, n_query, n_dim)
        out = torch.mm(
            heads.permute(1, 2, 0, 3).reshape(-1, self.n_heads * self.value_dim),
            # batch_size*n_query*n_heads*value_dim
            self.w_out.view(-1, self.embedding_dim)
            # n_heads*value_dim*embedding_dim
        ).view(-1, n_query, self.embedding_dim)

        return out, attention  # batch_size*n_query*embedding_dim


class Normalization(nn.Module):
    def __init__(self, embedding_dim):
        super(Normalization, self).__init__()
        self.normalizer = nn.LayerNorm(embedding_dim)

    def forward(self, input):
        return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_head):
        super(EncoderLayer, self).__init__()
        self.multiHeadAttention = MultiHeadAttention(embedding_dim, n_head)
        self.normalization1 = Normalization(embedding_dim)
        self.feedForward = nn.Sequential(nn.Linear(embedding_dim, 512), nn.ReLU(inplace=True),
                                         nn.Linear(512, embedding_dim))
        self.normalization2 = Normalization(embedding_dim)

    def forward(self, src, key_padding_mask=None, attn_mask=None):
        h0 = src
        h = self.normalization1(src)
        h, _ = self.multiHeadAttention(q=h, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        h = h + h0
        h1 = h
        h = self.normalization2(h)
        h = self.feedForward(h)
        h2 = h + h1
        return h2


class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_head):
        super(DecoderLayer, self).__init__()
        self.multiHeadAttention = MultiHeadAttention(embedding_dim, n_head)
        self.normalization1 = Normalization(embedding_dim)
        self.feedForward = nn.Sequential(nn.Linear(embedding_dim, 512),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(512, embedding_dim))
        self.normalization2 = Normalization(embedding_dim)

    def forward(self, tgt, memory, key_padding_mask=None, attn_mask=None):
        h0 = tgt
        tgt = self.normalization1(tgt)
        memory = self.normalization1(memory)
        h, w = self.multiHeadAttention(q=tgt, k=memory, v=memory, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        h = h + h0
        h1 = h
        h = self.normalization2(h)
        h = self.feedForward(h)
        h2 = h + h1
        return h2, w


class Encoder(nn.Module):
    def __init__(self, embedding_dim=128, n_head=8, n_layer=1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(EncoderLayer(embedding_dim, n_head) for i in range(n_layer))

    def forward(self, src, key_padding_mask=None, attn_mask=None):
        for layer in self.layers:
            src = layer(src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return src


class Decoder(nn.Module):
    def __init__(self, embedding_dim=128, n_head=8, n_layer=1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(embedding_dim, n_head) for i in range(n_layer)])

    def forward(self, tgt, memory, key_padding_mask=None, attn_mask=None):
        for layer in self.layers:
            tgt, w = layer(tgt, memory, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return tgt, w


class SpatialMapEncoder(nn.Module):
    def __init__(self, input_channels=5, feature_dim=64):
        super(SpatialMapEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, feature_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups=_compatible_group_count(feature_dim, 8), num_channels=feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, map_inputs):
        if map_inputs is None:
            raise ValueError("map_inputs is required")
        if map_inputs.dim() != 4:
            raise ValueError("map_inputs must have shape [batch, channels, height, width]")
        if map_inputs.dtype == torch.uint8:
            map_inputs = map_inputs.float() / 255.0
        else:
            map_inputs = map_inputs.float()
        return self.encoder(map_inputs)


class NodeMapFeatureSampler(nn.Module):
    def __init__(self, map_resolution=4, coord_scale=640):
        super(NodeMapFeatureSampler, self).__init__()
        self.map_resolution = map_resolution
        self.coord_scale = coord_scale

    def forward(self, feature_map, node_inputs, map_height, map_width, node_padding_mask=None):
        batch_size, node_count, _ = node_inputs.size()
        node_xy = node_inputs[:, :, :2] * self.coord_scale

        original_width = map_width * self.map_resolution
        original_height = map_height * self.map_resolution
        x_norm = 2 * (node_xy[..., 0] / max(original_width - 1, 1)) - 1
        y_norm = 2 * (node_xy[..., 1] / max(original_height - 1, 1)) - 1
        grid = torch.stack((x_norm, y_norm), dim=-1)
        grid = grid.view(batch_size, node_count, 1, 2)

        sampled = F.grid_sample(
            feature_map,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        node_map_features = sampled.squeeze(-1).permute(0, 2, 1)

        if node_padding_mask is not None:
            if node_padding_mask.dim() == 3:
                mask = node_padding_mask.squeeze(1)
            elif node_padding_mask.dim() == 2:
                mask = node_padding_mask
            else:
                raise ValueError("node_padding_mask must have shape [batch, nodes] or [batch, 1, nodes]")
            mask = mask.unsqueeze(-1).bool()
            node_map_features = node_map_features.masked_fill(mask, 0)

        return node_map_features


class NodeMapFusion(nn.Module):
    def __init__(self, embedding_dim=128, map_feature_dim=64, gate_bias_init=-2.0):
        super(NodeMapFusion, self).__init__()
        self.map_projection = nn.Sequential(
            nn.Linear(map_feature_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.map_gate = nn.Linear(embedding_dim * 2, embedding_dim)
        nn.init.constant_(self.map_gate.bias, gate_bias_init)

    def forward(self, node_feature, node_map_features):
        map_projection = self.map_projection(node_map_features)
        gate_inputs = torch.cat((node_feature, map_projection), dim=-1)
        gate = torch.sigmoid(self.map_gate(gate_inputs))
        fused_node_feature = node_feature + gate * map_projection
        return fused_node_feature, gate


class PolicyNet(nn.Module):
    def __init__(self, input_dim, embedding_dim, map_input_channels=5, map_feature_dim=64, map_resolution=4,
                 gate_bias_init=-2.0, action_input_dim=0, use_map_inputs=True):
        super(PolicyNet, self).__init__()
        self.action_input_dim = action_input_dim
        self.use_map_inputs = use_map_inputs
        self.initial_embedding = nn.Linear(input_dim, embedding_dim) # layer for non-end position
        self.current_embedding = nn.Linear(embedding_dim * 2, embedding_dim)
        if self.action_input_dim > 0:
            self.action_input_embedding = nn.Linear(action_input_dim, embedding_dim)
            self.neighbor_action_fusion = nn.Linear(embedding_dim * 2, embedding_dim)
        if self.use_map_inputs:
            self.map_encoder = SpatialMapEncoder(map_input_channels, map_feature_dim)
            self.node_map_sampler = NodeMapFeatureSampler(map_resolution=map_resolution)
            self.node_map_fusion = NodeMapFusion(embedding_dim, map_feature_dim, gate_bias_init)
        else:
            self.map_encoder = None
            self.node_map_sampler = None
            self.node_map_fusion = None

        self.encoder = Encoder(embedding_dim=embedding_dim, n_head=8, n_layer=6)
        self.decoder = Decoder(embedding_dim=embedding_dim, n_head=8, n_layer=1)
        self.pointer = SingleHeadAttention(embedding_dim)

    def encode_graph(self, node_inputs, node_padding_mask, edge_mask, map_inputs=None):
        node_feature = self.initial_embedding(node_inputs)
        diagnostics = {}
        if self.use_map_inputs:
            map_feature_map = self.map_encoder(map_inputs)
            map_height, map_width = map_inputs.shape[-2:]
            node_map_features = self.node_map_sampler(
                map_feature_map,
                node_inputs,
                map_height,
                map_width,
                node_padding_mask,
            )
            node_feature, gate = self.node_map_fusion(node_feature, node_map_features)
            diagnostics = {
                "map_feature_std": node_map_features.std(unbiased=False).detach(),
                "fusion_gate_mean": gate.mean().detach(),
                "fusion_gate_std": gate.std(unbiased=False).detach(),
                "node_map_feature_norm": torch.norm(node_map_features, dim=-1).detach(),
            }
        enhanced_node_feature = self.encoder(src=node_feature, key_padding_mask=node_padding_mask, attn_mask=edge_mask)

        return enhanced_node_feature, diagnostics

    def _validate_action_inputs(self, action_inputs, edge_inputs):
        if self.action_input_dim <= 0:
            return
        if action_inputs is None:
            raise ValueError("action_inputs is required when action_input_dim > 0")
        if action_inputs.size(0) != edge_inputs.size(0) or action_inputs.size(1) != edge_inputs.size(2):
            raise ValueError("action_inputs must have shape [batch, k_size, action_input_dim]")
        if action_inputs.size(2) != self.action_input_dim:
            raise ValueError(
                f"action_inputs feature dim {action_inputs.size(2)} does not match "
                f"action_input_dim {self.action_input_dim}"
            )

    def output_policy(self, enhanced_node_feature, edge_inputs, current_index, edge_padding_mask, node_padding_mask,
                      action_inputs=None):
        safe_edge_inputs = edge_inputs.clamp(min=0)
        current_edge = safe_edge_inputs.permute(0, 2, 1)
        embedding_dim = enhanced_node_feature.size()[2]

        neigboring_feature = torch.gather(enhanced_node_feature, 1, current_edge.repeat(1, 1, embedding_dim))
        if self.action_input_dim > 0:
            self._validate_action_inputs(action_inputs, edge_inputs)
            action_feature = self.action_input_embedding(action_inputs)
            neigboring_feature = self.neighbor_action_fusion(
                torch.cat((neigboring_feature, action_feature), dim=-1)
            )

        current_node_feature = torch.gather(enhanced_node_feature, 1, current_index.repeat(1, 1, embedding_dim))

        if edge_padding_mask is not None:
            current_mask = edge_padding_mask.clone()
            # print(current_mask)
        else:
            current_mask = torch.zeros_like(edge_inputs, dtype=torch.bool, device=edge_inputs.device)

        current_mask[:,:,0] = 1 # don't stay at current position
        #assert 0 in current_mask

        enhanced_current_node_feature, _ = self.decoder(current_node_feature, enhanced_node_feature, node_padding_mask)
        enhanced_current_node_feature = self.current_embedding(torch.cat((enhanced_current_node_feature, current_node_feature), dim=-1))
        logp = self.pointer(enhanced_current_node_feature, neigboring_feature, current_mask)
        logp= logp.squeeze(1) # batch_size*k_size

        return logp

    def forward(self, node_inputs, edge_inputs, current_index, node_padding_mask=None, edge_padding_mask=None,
                edge_mask=None, map_inputs=None, action_inputs=None, return_diagnostics=False):
        enhanced_node_feature, diagnostics = self.encode_graph(node_inputs, node_padding_mask, edge_mask, map_inputs)
        logp = self.output_policy(
            enhanced_node_feature,
            edge_inputs,
            current_index,
            edge_padding_mask,
            node_padding_mask,
            action_inputs=action_inputs,
        )
        if return_diagnostics:
            return logp, diagnostics
        return logp


class QNet(nn.Module):
    def __init__(self, input_dim, embedding_dim, map_input_channels=5, map_feature_dim=64, map_resolution=4,
                 gate_bias_init=-2.0, action_input_dim=0, use_map_inputs=True):
        super(QNet, self).__init__()
        self.action_input_dim = action_input_dim
        self.use_map_inputs = use_map_inputs
        self.initial_embedding = nn.Linear(input_dim, embedding_dim) # layer for non-end position
        if self.action_input_dim > 0:
            self.action_input_embedding = nn.Linear(action_input_dim, embedding_dim)
            self.action_embedding = nn.Linear(embedding_dim * 4, embedding_dim)
        else:
            self.action_embedding = nn.Linear(embedding_dim * 3, embedding_dim)
        if self.use_map_inputs:
            self.map_encoder = SpatialMapEncoder(map_input_channels, map_feature_dim)
            self.node_map_sampler = NodeMapFeatureSampler(map_resolution=map_resolution)
            self.node_map_fusion = NodeMapFusion(embedding_dim, map_feature_dim, gate_bias_init)
        else:
            self.map_encoder = None
            self.node_map_sampler = None
            self.node_map_fusion = None

        self.encoder = Encoder(embedding_dim=embedding_dim, n_head=8, n_layer=6)
        self.decoder = Decoder(embedding_dim=embedding_dim, n_head=8, n_layer=1)

        self.q_values_layer = nn.Linear(embedding_dim, 1)

    def encode_graph(self, node_inputs, node_padding_mask, edge_mask, map_inputs=None):
        embedding_feature = self.initial_embedding(node_inputs)
        diagnostics = {}
        if self.use_map_inputs:
            map_feature_map = self.map_encoder(map_inputs)
            map_height, map_width = map_inputs.shape[-2:]
            node_map_features = self.node_map_sampler(
                map_feature_map,
                node_inputs,
                map_height,
                map_width,
                node_padding_mask,
            )
            embedding_feature, gate = self.node_map_fusion(embedding_feature, node_map_features)
            diagnostics = {
                "map_feature_std": node_map_features.std(unbiased=False).detach(),
                "fusion_gate_mean": gate.mean().detach(),
                "fusion_gate_std": gate.std(unbiased=False).detach(),
                "node_map_feature_norm": torch.norm(node_map_features, dim=-1).detach(),
            }
        embedding_feature = self.encoder(src=embedding_feature, key_padding_mask=node_padding_mask, attn_mask=edge_mask)

        return embedding_feature, diagnostics

    def _validate_action_inputs(self, action_inputs, edge_inputs):
        if self.action_input_dim <= 0:
            return
        if action_inputs is None:
            raise ValueError("action_inputs is required when action_input_dim > 0")
        if action_inputs.size(0) != edge_inputs.size(0) or action_inputs.size(1) != edge_inputs.size(2):
            raise ValueError("action_inputs must have shape [batch, k_size, action_input_dim]")
        if action_inputs.size(2) != self.action_input_dim:
            raise ValueError(
                f"action_inputs feature dim {action_inputs.size(2)} does not match "
                f"action_input_dim {self.action_input_dim}"
            )

    def output_q_values(self, enhanced_node_feature, edge_inputs, current_index, edge_padding_mask, node_padding_mask,
                        action_inputs=None):
        k_size = edge_inputs.size()[2]
        current_edge = edge_inputs.clamp(min=0).permute(0, 2, 1)
        embedding_dim = enhanced_node_feature.size()[2]

        neigboring_feature = torch.gather(enhanced_node_feature, 1, current_edge.repeat(1, 1, embedding_dim))

        current_node_feature = torch.gather(enhanced_node_feature, 1, current_index.repeat(1, 1, embedding_dim))

        enhanced_current_node_feature, attention_weights = self.decoder(current_node_feature, enhanced_node_feature, node_padding_mask)
        feature_list = [
            enhanced_current_node_feature.repeat(1, k_size, 1),
            current_node_feature.repeat(1, k_size, 1),
            neigboring_feature,
        ]
        if self.action_input_dim > 0:
            self._validate_action_inputs(action_inputs, edge_inputs)
            feature_list.append(self.action_input_embedding(action_inputs))
        action_features = torch.cat(feature_list, dim=-1)
        action_features = self.action_embedding(action_features)
        q_values = self.q_values_layer(action_features)

        if edge_padding_mask is not None:
            current_mask = edge_padding_mask.clone()
        else:
            current_mask = torch.zeros_like(edge_inputs, dtype=torch.bool, device=edge_inputs.device)
        current_mask[:, :, 0] = 1  # don't stay at current position
        #assert 0 in current_mask
        current_mask = current_mask.permute(0, 2, 1)
        zero = torch.zeros_like(q_values).to(q_values.device)
        q_values = torch.where(current_mask == 1, zero, q_values)

        return q_values, attention_weights

    def forward(self, node_inputs, edge_inputs, current_index, node_padding_mask=None, edge_padding_mask=None,
                edge_mask=None, map_inputs=None, action_inputs=None, return_diagnostics=False):
        enhanced_node_feature, diagnostics = self.encode_graph(node_inputs, node_padding_mask, edge_mask, map_inputs)
        q_values, attention_weights = self.output_q_values(
            enhanced_node_feature,
            edge_inputs,
            current_index,
            edge_padding_mask,
            node_padding_mask,
            action_inputs=action_inputs,
        )
        if return_diagnostics:
            return q_values, attention_weights, diagnostics
        return q_values, attention_weights
