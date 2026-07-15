# Lightweight Distance-Aware Attention 需求 Proposal

## 1. 背景

当前代码已经在原始 ARIADNE 的基础上加入了节点级 trajectory memory：

- `parameter.py` 中 `USE_NODE_FEATURE_TRAJECTORY_MEMORY = True`，当前默认配置是 `node_memory_only2`。
- `node_features.py` 中 `build_node_and_action_features()` 统一构造训练和测试 observation。
- `graph_generator.py` 中已有 `get_shortest_path_tree()` 和 `get_decayed_route_memory()`，可以提供当前节点到全图节点的单源图距离，以及基于历史路径的 max-decay memory。
- `model.py` 中 `PolicyNet.output_policy()` 使用 pointer attention 对当前节点的候选邻居输出动作概率。
- 当前动作空间不是任意全图节点，而是当前节点的 `K_SIZE` 个邻居；`edge_inputs` 的第 0 个位置是当前节点本身，随后才是可移动候选邻居，padding 使用 `PADDING_NODE_INDEX = -1`。

本 proposal 关注在当前 memory 版本基础上加入轻量版 distance-aware attention，用显式距离先验帮助策略减少不必要长距离动作，从而提高探索效率并缩短总路径长度。

## 2. 已明确需求

以下需求来自当前讨论，后续实现时可以视为已确认：

1. 基于当前 ARIADNE 代码增量修改，不重写整体模型。
2. 保留现有 node-level trajectory memory，不把 memory 方案移除。
3. 第一版做轻量方案，不先实现全图 all-pairs shortest-path distance attention。
4. 重点利用当前节点到候选邻居的直接边距离，因为当前动作本身就是选择邻居。
5. 目标指标包括探索总路径长度 `travel_dist`，并关注探索效率。
6. 本文档是后续写代码的需求参考；未确认事项必须先向用户确认，不应由 Codex 自行假定。

## 3. 非目标

第一版 lightweight distance-aware attention 不默认做以下事情：

- 不默认计算每个 observation 的 `[N, N]` 全图最短路距离矩阵。
- 不默认修改 reward 函数。
- 不默认 mask 掉远距离动作。
- 不默认禁止回头或访问高 memory 节点。
- 不默认修改地图数据集、episode 长度、SAC 主训练逻辑或测试集划分。
- 不默认兼容旧 checkpoint；若模型参数或输入维度变化，应重新训练。

## 4. 当前代码可复用部分

### 4.1 候选动作距离

`node_features.py` 已经实现 `get_edge_distance(env, current_index, next_index, normalizer)`：

```text
edge_dist_norm_j = edge.length / GRAPH_DISTANCE_NORMALIZER
```

如果当前边存在，则直接读取 graph edge 的 `length`；如果边不存在，则 fallback 到当前节点和候选节点坐标的欧氏距离。

对于当前动作空间来说，候选动作 `current -> next` 本身就是图中的一条边。因此候选邻居的 graph shortest path distance 通常等于该 edge distance。第一版 pointer-level distance bias 可以直接使用 `edge_dist_norm_j`，避免额外 Dijkstra。

### 4.2 动作级 feature 管线

当前代码已经有 action feature 管线：

- `USE_ACTION_FEATURES`
- `USE_ACTION_FEATURE_EDGE_DIST`
- `ACTION_FEATURE_DIM`
- `action_inputs`
- `PolicyNet` 中的 `action_input_embedding`
- `QNet` 中的 `action_input_embedding`

这条管线可以作为距离信息进入网络的一种方式。但它和本 proposal 说的显式 attention bias 不完全相同：

```text
action feature: 让网络自己学习如何使用距离
attention bias: 在 softmax 前直接改变候选动作 logit
```

## 5. 推荐的轻量方案轮廓

### 5.1 Pointer-level distance-aware attention

第一版建议优先在 `SingleHeadAttention` 对候选动作打分时加入距离 bias。

当前 pointer attention 逻辑：

```text
U_j = tanh_clipping * tanh(Q K_j^T / sqrt(d))
logp = log_softmax(mask(U))
```

待确认方案的核心形式：

```text
U_j = tanh_clipping * tanh(Q K_j^T / sqrt(d))
U_j = U_j - softplus(eta) * edge_dist_norm_j
logp = log_softmax(mask(U))
```

其中：

- `edge_dist_norm_j` 形状为 `[batch_size, 1, K_SIZE]`。
- `edge_dist_norm_j` 对 padding slot 应为 0，最终仍由 `edge_padding_mask` mask 掉。
- 第 0 个 stay/current slot 仍应由现有 mask 逻辑屏蔽，不靠 distance bias 处理。
- `eta` 是可学习参数，建议保证为非负，从而让距离项始终是惩罚项。

这一步直接作用在策略动作概率上，是最符合“缩短总路径长度”目标的最小改动。

### 5.2 QNet 是否同步使用距离

`QNet` 不是 pointer attention 输出，而是拼接当前节点、增强当前节点、候选邻居和 action feature 后回归 Q value。

若仅在 PolicyNet 加 pointer distance bias，而 QNet 不感知相同的距离先验，policy 和 critic 的表达可能不完全对齐。因此有两个候选方向：

```text
方案 A：只在 PolicyNet pointer logits 加 distance bias。
方案 B：PolicyNet 加 distance bias，同时启用 QNet 的 edge_dist action feature。
方案 C：PolicyNet 加 distance bias，并在 QNet action_features 中额外拼接同一个 edge_dist_norm。
```

这部分需要用户确认，不能默认决定。

### 5.3 Decoder current-to-all distance bias

当前 `PolicyNet.output_policy()` 在 pointer 前会让当前节点通过 `Decoder` attend 到全图节点：

```text
enhanced_current_node_feature = decoder(current_node_feature, enhanced_node_feature)
```

`node_features.py` 已经计算了当前节点到所有节点的单源图距离：

```text
distances, first_hops, reachable = get_shortest_path_tree(current_index)
graph_dist_to_current = distances / GRAPH_DISTANCE_NORMALIZER
```

因此可以选择在 decoder cross-attention 中增加：

```text
score_i = Q_current K_i^T / sqrt(d) - softplus(alpha) * d_G(current, i)_norm
```

这仍然是轻量的，因为只需要 `[1, N]` 单源距离 bias，而不是 `[N, N]` all-pairs 距离矩阵。

但这会扩大实现范围：需要 observation 新增 `decoder_attn_bias`，训练 replay buffer 也要保存 current/next 的该 bias。是否第一版加入，需要确认。

### 5.4 Encoder edge-length bias

Encoder 当前通过 `edge_mask` 限制节点只 attend 到 graph 邻居：

```text
enhanced_node_feature = encoder(node_feature, attn_mask=edge_mask)
```

可以只对已有边加入边长 bias：

```text
score_ij = Q_i K_j^T / sqrt(d) - softplus(alpha) * edge_length(i, j)_norm
```

这不需要 all-pairs graph shortest path，但仍需要构造 `[N, N]` 的 edge-length bias 矩阵，并随 observation 存入 replay buffer。该方案比 pointer-only 更重，且第一版是否必要需要确认。

## 6. 数据结构改动候选

### 6.1 Pointer-only 最小改动

如果只实现 pointer-level distance bias，建议新增：

```python
ObservationFeatures.action_distance_inputs: np.ndarray  # shape [K_SIZE]
```

训练和测试 observation tuple 需要从当前：

```text
node_inputs, edge_inputs, action_inputs, current_index,
node_padding_mask, edge_padding_mask, edge_mask
```

扩展为：

```text
node_inputs, edge_inputs, action_inputs, action_distance_inputs, current_index,
node_padding_mask, edge_padding_mask, edge_mask
```

或将 `action_distance_inputs` 作为 `action_inputs` 的固定列复用。两种方式需要确认。

### 6.2 Decoder bias 扩展

如果加入 decoder current-to-all distance bias，建议新增：

```python
ObservationFeatures.decoder_distance_bias: np.ndarray  # shape [1, n_nodes]
```

训练时需要和 `node_inputs` 一样 padding 到 `NODE_PADDING_SIZE`：

```text
[1, n_nodes] -> [1, NODE_PADDING_SIZE]
```

padding 位置可以填 0，因为 `node_padding_mask` 会屏蔽 padding 节点。

### 6.3 Encoder edge bias 扩展

如果加入 encoder edge-length bias，建议新增：

```python
ObservationFeatures.encoder_distance_bias: np.ndarray  # shape [n_nodes, n_nodes]
```

padding 后形状：

```text
[NODE_PADDING_SIZE, NODE_PADDING_SIZE]
```

非边位置可以填 0，因为 `edge_mask` 已经屏蔽非边；已有边位置填负距离或原始距离，具体符号取决于模型内是否统一乘以 `-softplus(scale)`。

## 7. 参数改动候选

建议在 `parameter.py` 中增加开关和初始值，但具体默认值需要确认：

```python
USE_DISTANCE_AWARE_POINTER = True
USE_DISTANCE_AWARE_DECODER = False
USE_DISTANCE_AWARE_ENCODER = False
DISTANCE_ATTENTION_INIT_SCALE = 0.5
DISTANCE_ATTENTION_MAX_NORM = 2.0
FOLDER_NAME = 'node_memory_dist_pointer'
```

其中：

- `USE_DISTANCE_AWARE_POINTER` 控制 pointer logits 是否使用 edge distance bias。
- `USE_DISTANCE_AWARE_DECODER` 控制 decoder current-to-all distance bias。
- `USE_DISTANCE_AWARE_ENCODER` 控制 encoder edge-length bias。
- `DISTANCE_ATTENTION_INIT_SCALE` 是距离惩罚强度的初始目标值。
- `DISTANCE_ATTENTION_MAX_NORM` 用于 clip normalized distance，防止极端距离导致 logit 变化过大。

这些名字和默认值只是候选，不能视为最终决定。

## 8. 模型改动候选

### 8.1 `SingleHeadAttention`

候选接口：

```python
def forward(self, q, k, mask=None, logit_bias=None):
```

候选逻辑：

```python
U = self.norm_factor * torch.matmul(Q, K.transpose(1, 2))
U = self.tanh_clipping * torch.tanh(U)
if logit_bias is not None:
    U = U + logit_bias
if mask is not None:
    U = U.masked_fill(mask == 1, -1e8)
attention = torch.log_softmax(U, dim=-1)
```

距离惩罚可以在 `PolicyNet` 中构造：

```python
distance_logit_bias = -softplus(pointer_distance_scale) * action_distance_inputs
```

然后传入 pointer。

### 8.2 `MultiHeadAttention`

如果确认需要 decoder 或 encoder distance bias，则候选接口：

```python
def forward(self, q, k=None, v=None, key_padding_mask=None, attn_mask=None, attn_bias=None):
```

候选逻辑：

```python
U = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))
if attn_bias is not None:
    U = U + attn_bias.view(1, n_batch, n_query, n_key).expand_as(U)
```

若需要 per-head scale，可以使用：

```python
U = U - softplus(distance_scale).view(n_heads, 1, 1, 1) * distance_norm
```

但第一版是否使用 per-head scale 需要确认。

## 9. 测试需求

后续实现时至少应补充或更新以下测试：

1. `node_features` 能为候选动作输出正确的 edge distance，padding slot 为 0。
2. `PolicyNet` 在给定正距离 bias 时仍能输出有限 log probability，并保持 padding/stay slot 被 mask。
3. 当所有有效动作 embedding 相同但距离不同，启用 distance-aware pointer 后较短动作的 log probability 应更高。
4. `QNet` 若启用 edge distance action feature，应确认输入维度、mask 和输出 shape 正确。
5. `Worker` 和 `TestWorker` observation tuple 顺序一致。
6. Replay buffer current/next observation 字段数量一致。

当前环境中可能没有安装 `pytest`。如果本地环境缺少 `pytest`，实现后应在具备依赖的训练环境中运行测试。

## 10. 待确认问题

以下问题需要用户确认。建议和分析仅供参考，不代表已经授权实现。

### Q1. 第一版 lightweight 范围选哪一档？

可选范围：

```text
A. 只做 pointer-level edge distance bias
B. pointer-level edge distance bias + 启用 edge_dist action feature
C. pointer-level bias + decoder current-to-all graph distance bias
D. pointer-level bias + decoder bias + encoder edge-length bias
```

建议：优先选 B。

分析：A 是最小改动，但 critic 可能看不到同样明确的距离信息。B 仍然很轻量，能让 PolicyNet 的 pointer logits 显式受距离影响，同时让 PolicyNet/QNet 的 action embedding 也看到 edge distance。C 和 D 更接近完整 distance-aware attention，但 observation/replay/model 接口改动更大，第一版调试成本更高。

回答：B。

### Q2. Pointer distance 使用独立字段还是复用 `action_inputs` 的 `edge_dist_norm`？

可选方式：

```text
A. 新增 action_distance_inputs 专门给 pointer bias 使用
B. 复用 action_inputs 中的 edge_dist_norm 列
```

建议：如果 Q1 选择 B，则复用 `action_inputs` 中的 `edge_dist_norm`；如果 Q1 选择 A，则新增 `action_distance_inputs`。

分析：复用 action feature 可以减少 observation 字段数量，但要求 `USE_ACTION_FEATURES=True` 且 `USE_ACTION_FEATURE_EDGE_DIST=True`。独立字段更清晰，也允许不启用 action feature 时仍使用 pointer distance bias，但 replay buffer 字段会增加。

回答：B。

### Q3. Distance penalty 的符号是否强制为负？

可选方式：

```text
A. 强制负惩罚：U_j -= softplus(eta) * dist_j
B. 自由学习正负：U_j += eta * dist_j
```

建议：选 A。

分析：本需求目标是缩短路径长度，负惩罚符合先验且更可解释。B 给模型更大自由度，但可能学出偏好长边的异常行为，和“显式 distance-aware”目标不够一致。

回答：A。

### Q4. Distance bias 加在 pointer 的哪个位置？

可选方式：

```text
A. tanh clipping 后、mask 前
B. tanh clipping 前
C. mask 后
```

建议：选 A。

分析：A 保留原始 attention logits 的 bounded 行为，同时让距离 bias 以清晰的 additive prior 进入 softmax。B 会让距离 bias 被 tanh 压缩，效果不直观。C 不建议，因为 mask 后有 `-1e8`，再加 bias 容易引入无意义操作。

回答：A。

### Q5. 距离归一化和 clipping 怎么设置？

可选方式：

```text
A. dist_norm = edge_length / GRAPH_DISTANCE_NORMALIZER，不额外 clip
B. dist_norm = clip(edge_length / GRAPH_DISTANCE_NORMALIZER, 0, max_norm)
C. 使用 sensor_range 或局部邻域最大边长归一化
```

建议：选 B，`max_norm` 初始候选为 `2.0`。

分析：当前 `GRAPH_DISTANCE_NORMALIZER = 640`，局部邻居边通常远小于 640，因此 dist_norm 多数较小。clip 不会影响正常值，但能防止异常边长或未来地图尺度变化导致 logit 过大。C 的局部归一化会让同一绝对距离在不同状态下含义变化，不利于跨地图泛化。

回答：选 B，`max_norm` 初始候选为 `2.0`。

### Q6. `eta` 初始化为多少？

可选方式：

```text
A. 0.1
B. 0.5
C. 1.0
```

建议：选 B。

分析：pointer 原始 logits 被 `tanh_clipping=10` 限制在约 `[-10, 10]`。如果局部边长归一化后常在 `0.05-0.25`，`eta=0.5` 对 logit 的初始影响约 `0.025-0.125`，属于温和先验。`0.1` 可能太弱，`1.0` 仍可接受但更可能早期过度偏近。

回答：B。

### Q7. 是否同时修改 QNet？

可选方式：

```text
A. 不修改 QNet，仅 PolicyNet pointer 使用 distance bias
B. QNet 通过 action_inputs 接收 edge_dist_norm
C. QNet 也增加独立 distance scale 或 bias
```

建议：选 B。

分析：QNet 的动作价值需要知道动作代价，否则 policy 的距离先验和 critic 表达可能不一致。B 利用已有 action feature 管线，改动较小。C 会增加额外机制，但 QNet 当前没有 pointer logits，直接加 bias 的位置不如 PolicyNet 清楚。

回答：B。

### Q8. 是否保留当前 node memory-only 作为 baseline？

可选方式：

```text
A. 保留当前 run/config 作为 baseline，新建 FOLDER_NAME
B. 直接覆盖当前 FOLDER_NAME
```

建议：选 A。

分析：该改动会影响模型结构或输入，旧 checkpoint 大概率不兼容。保留 baseline 便于比较 `travel_dist`、success rate、backtracking metrics 和训练曲线。

回答：B。

### Q9. 实验命名用什么？

候选名称：

```text
A. node_memory_dist_pointer
B. dtm_dist_pointer
C. dist_aware_pointer
```

建议：选 A 或 B。

分析：A 清楚表达是在 node memory 基础上加 pointer distance。B 更贴近现有 `Directional Topological Memory` proposal 命名。如果主要论文叙述希望强调 DTM，可用 B；如果主要做工程 ablation，可用 A。

回答：dist_pointer_memory_i

### Q10. 是否把 decoder current-to-all distance bias 纳入第一版？

可选方式：

```text
A. 不纳入第一版，作为第二阶段
B. 纳入第一版
```

建议：选 A。

分析：decoder bias 仍然是轻量的 `[1, N]` 单源距离，但会增加 observation 字段、padding、replay buffer 和模型接口复杂度。若 pointer-only/B 方案效果不足，再加入 decoder bias 更利于 ablation。

回答：A。

### Q11. 是否把 encoder edge-length bias 纳入第一版？

可选方式：

```text
A. 不纳入第一版
B. 纳入第一版
```

建议：选 A。

分析：encoder bias 需要 `[N, N]` edge-length bias，虽然不是 all-pairs shortest path，但存储和接口成本明显更高。当前 encoder 已经由 `edge_mask` 限制邻接注意力，第一版优先把距离作用在动作选择层更直接。

回答：A。

### Q12. 评估指标是否新增专门的 distance-aware ablation 指标？

可选方式：

```text
A. 使用现有 travel_dist、success_rate、explored_rate 和 backtracking metrics
B. 新增 average_step_distance、long_edge_action_ratio 等指标
```

建议：选 B，但不阻塞第一版模型实现。

分析：总路径长度是最终指标，但它受探索成功率和地图难度影响。新增平均单步距离、长边动作比例可以更直接判断 distance bias 是否真的改变了动作偏好。

回答：B。

## 11. 建议的第一版实现顺序

如果用户确认 Q1 选择 B，建议后续代码实现顺序为：

1. 在 `parameter.py` 增加 distance-aware pointer 开关和 scale 初始化参数，并启用 `USE_ACTION_FEATURES` + `USE_ACTION_FEATURE_EDGE_DIST`。
2. 确认 `node_features.py` 的 `edge_dist_norm` 能覆盖全部有效候选动作，padding slot 为 0。
3. 在 `PolicyNet` 中从 `action_inputs` 取 `edge_dist_norm`，构造 pointer logit bias。
4. 修改 `SingleHeadAttention.forward()` 支持 `logit_bias`。
5. 保证 QNet 继续通过 action feature 接收 edge distance。
6. 更新 `FOLDER_NAME`，避免覆盖当前 memory-only baseline。
7. 补充单测和一个小 batch forward smoke test。

如果用户确认 Q1 选择 A，则需要改为新增独立 `action_distance_inputs`，并同步更新 Worker、TestWorker、driver replay buffer 和模型 forward 签名。

## 12. Definition of Done

后续实现完成后，应满足：

- 训练和测试 observation 字段一致。
- PolicyNet 和 QNet forward shape 正确。
- padding action 和 stay/current action 仍被正确 mask。
- distance bias 不会让 logp 出现 NaN 或 inf。
- 新模型配置不会误加载旧 checkpoint。
- 至少能运行单元测试或 smoke test，验证较短候选动作在相同 embedding 下获得更高初始 log probability。
- 测试输出能继续报告 `travel_dist`，并可和 `node_memory_only2` baseline 做对比。
