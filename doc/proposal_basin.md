# Proposal Basin: 基于当前动作的拓扑盆地特征

## 1. 背景

当前代码已经在原始 ARiADNE 的节点输入基础上扩展到 9 维：

```text
[
  x,
  y,
  utility,
  guidepost,
  graph_dist_to_current,
  utility_over_dist,
  visit_count,
  expected_unknown_gain,
  frontier_cluster_size
]
```

这些特征定义在 `observation_features.py`，训练和测试分别在 `worker.py` 与 `test_worker.py` 的 `get_observations()` 中构造。当前 `parameter.py` 与 `test_parameter.py` 中 `INPUT_DIM = 9`。

当前策略仍然遵循 ARiADNE 的核心动作形式：在每个 decision step，从当前节点的邻居集合中选择一个节点作为下一步 waypoint。也就是说，最终决策发生在 `PolicyNet.output_policy()` 的 decoder/pointer 阶段：

```text
current node feature + neighboring node features -> pointer logits over K neighbors
```

现有 9 维节点特征主要回答的是：

```text
某个节点本身是否值得去？
```

但当前动作 `current -> neighbor_j` 更接近一个“方向选择”或“分支入口选择”。对于探索路径长度而言，一个邻居节点的价值不仅取决于该邻居自身，还取决于它背后通过当前图最短路可优先到达的整片区域。因此，本 proposal 讨论新增“拓扑盆地特征”。

## 2. 目标

本阶段目标是为每个候选动作 `current -> neighbor_j` 构造 action-level basin features，让策略在 pointer 阶段比较邻居时，不仅看到邻居节点 embedding，还能看到：

- 选择该邻居作为第一步后，背后区域的总潜在探索收益。
- 该方向是否主要通向已访问/低收益/死胡同区域。
- 该方向是否包含更远处但整体收益更高的 frontier 或 unknown gain。

一句话概括：

```text
从“比较 K 个邻居点”改为“比较 K 个方向背后的区域价值”。
```

## 3. 非目标

本 proposal 第一阶段不主动引入以下内容，除非后续确认：

- 不修改当前 9 维 `node_inputs` 的语义和顺序。
- 不改变图构造方式，不修改均匀采样节点逻辑。
- 不引入 ground truth、privileged expert、oracle reward。
- 不修改 `Env.calculate_reward()`。
- 不引入全局层级图、community detection、TSP global path。
- 不引入 edge swept gain；该方向属于另一个 proposal。
- 不改变动作空间，仍然从当前节点邻居中选一个动作。

## 4. 核心定义

### 4.1 动作特征

这里的“动作特征”指的是：特征不是附加到每个全局图节点上的固定 `node_inputs`，而是附加到当前 decision step 的每个候选动作上。

若当前节点为 `c`，其邻居为：

```text
N(c) = [n0, n1, ..., nK-1]
```

则新增特征形状建议为：

```text
action_features: (batch_size, K_SIZE, ACTION_FEATURE_DIM)
```

其中 `action_features[:, j, :]` 描述动作：

```text
c -> nj
```

### 4.2 拓扑盆地

对当前节点 `c` 运行一次单源最短路，得到从 `c` 到所有可达节点 `v` 的当前图最短路。

对于每个可达节点 `v`，记录从 `c` 到 `v` 的最短路径第一跳：

```text
first_hop[v] = 从 c 到 v 的最短路径中，c 之后的第一个邻居节点
```

然后定义某个候选邻居 `nj` 的 basin：

```text
Basin(nj) = { v | first_hop[v] == nj }
```

直观解释：

```text
Basin(nj) 是选择 nj 作为第一步后，当前最短路意义下优先通向的那批节点。
```

这里的最短路基于当前 collision-free graph 的边长，不使用 ground truth，也不是欧氏直线距离。

### 4.3 为什么不对每个节点单独跑 A*

本方案不应对每个目标节点单独调用 A*，也不应对每个邻居重复跑最短路。正确做法是在当前节点 `c` 处运行一次单源 Dijkstra，同时维护 `dist[v]` 与 `first_hop[v]`。

复杂度为：

```text
Dijkstra: O(E log N)
basin aggregation: O(N * ACTION_FEATURE_DIM)
```

当前代码已经在 `Graph_generator.get_normalized_shortest_path_distances()` 中每步计算 `graph_dist_to_current`，该逻辑可作为扩展基础。

## 5. 推荐第一版特征候选

以下特征是候选集合，最终是否全部实现需要确认。

### 5.1 basin_utility_sum

```text
sum(node_utility[v] for v in Basin(nj))
```

含义：该方向背后总共有多少 frontier 可观测收益。

建议归一化：除以一个固定尺度，或除以当前 observation 内所有 basin utility sum 的最大值。

### 5.2 basin_expected_unknown_gain_sum

```text
sum(expected_unknown_gain[v] for v in Basin(nj))
```

含义：该方向背后整体可能打开多少 unknown 区域。

当前 `expected_unknown_gain` 已经归一化到 `[0, 1]`，但 sum 之后可能超过 1，需要重新缩放或裁剪。

### 5.3 basin_frontier_cluster_max

```text
max(frontier_cluster_size[v] for v in Basin(nj))
```

含义：该方向背后是否存在较大的 frontier cluster。

使用 `max` 的原因是：大入口/大房间通常由某个较大 cluster 指示，直接求和可能被多个零碎小 cluster 放大。

### 5.4 basin_unvisited_ratio

```text
count(visit_count[v] == 0 for v in Basin(nj)) / max(1, count(Basin(nj)))
```

含义：该方向背后的区域有多少比例尚未访问。

该特征用于帮助策略减少反复进入已访问、低收益区域。

### 5.5 basin_min_dist_to_utility

```text
min(dist[v] for v in Basin(nj) if node_utility[v] > threshold)
```

含义：沿该方向走，最近的有效收益节点还有多远。

若该 basin 中不存在有效收益节点，应使用固定大值或 0，需要确认。

### 5.6 basin_gain_over_entry_cost

一个候选定义：

```text
basin_gain_over_entry_cost =
  basin_expected_unknown_gain_sum / (edge_length(current, nj) + eps)
```

也可以使用：

```text
(basin_utility_sum + basin_expected_unknown_gain_sum) / (edge_length(current, nj) + eps)
```

含义：进入该方向的单位入口成本收益。

## 6. 推荐计算流程

### 6.1 在 observation 阶段计算

训练路径：

```text
Worker.get_observations()
```

测试路径：

```text
TestWorker.get_observations()
```

当前这两个函数都会拿到：

- `node_coords`
- `graph`
- `node_utility`
- `visit_count`
- `expected_unknown_gain`
- `frontier_cluster_size`
- `current_node_index`
- `edge_inputs`

basin features 也应在这里构造，以保证训练和测试 observation 对齐。

### 6.2 扩展最短路函数

建议新增或扩展 graph generator 中的单源最短路函数，使其返回：

```text
normalized_distances
reachable_nodes
first_hop
```

`first_hop` 的形状：

```text
(n_nodes, 1)
```

语义：

```text
first_hop[v] = 从 current 到 v 的最短路径第一跳节点 index
first_hop[current] = current 或 -1
first_hop[unreachable] = -1
```

### 6.3 聚合 basin features

伪代码：

```python
def build_basin_action_features(edge_indices, first_hop, distances,
                                node_utility, visit_count,
                                expected_unknown_gain, frontier_cluster_size):
    action_features = zeros((K_SIZE, ACTION_FEATURE_DIM))

    for action_slot, neighbor_index in enumerate(edge_indices):
        if neighbor_index is padding or neighbor_index == current:
            continue

        basin_mask = first_hop == neighbor_index
        basin_nodes = nodes[basin_mask]

        action_features[action_slot] = aggregate_features(basin_nodes)

    return action_features
```

### 6.4 padding 行为

`edge_inputs` 当前会被 padding 到 `K_SIZE`，padding 值是 0，并通过 `edge_padding_mask` 屏蔽。

建议第一版：

```text
padding action 的 basin features 全部置 0。
```

真正动作不会选 padding，因此置 0 不应影响策略输出。

## 7. 网络接入位置

### 7.1 PolicyNet

basin features 最适合接入 decoder/pointer 阶段，而不是 encoder 阶段。

原因：

- basin features 是当前节点相关的 action-level 信息。
- 同一个节点在不同 current node 下可能属于不同 basin。
- 把它放进全局 `node_inputs` 会混淆“节点固有特征”和“当前动作特征”。

建议实现方式有两种，需要确认。

#### 方案 A：拼接到 neighboring feature

```text
neighbor_action_feature = MLP(action_features)
neighboring_feature = concat(enhanced_neighboring_feature, neighbor_action_feature)
pointer_key = Linear(neighboring_feature)
```

优点：

- pointer 直接比较增强后的候选动作 key。
- 实现直观。

缺点：

- 会改变 pointer 输入维度。
- 容易把 action-level 统计信息和 node embedding 混在一起。

#### 方案 B：给 pointer logits 增加 action bias

```text
base_logits = pointer(query, neighboring_feature)
action_bias = MLP([current_feature, neighboring_feature, action_features])
logits = base_logits + action_bias
```

优点：

- 保留原 pointer 的节点比较能力。
- basin features 只作为动作偏置参与最终打分。
- 更符合“动作特征”的语义。

缺点：

- 需要轻微改造 `SingleHeadAttention` 或在其外部组合 logits。

当前建议：方案 B 更干净，但最终需要确认。

### 7.2 QNet

如果 PolicyNet 使用 basin action features，QNet 也需要同步接入，否则 actor 和 critic 对动作价值使用的信息不一致。

建议 QNet 在 `output_q_values()` 中对每个候选动作构造：

```text
[
  enhanced_current_node_feature,
  current_node_feature,
  neighboring_feature,
  action_feature_embedding
]
```

再输出每个候选动作的 Q 值。

## 8. Replay Buffer 与 Observation 结构影响

当前 replay buffer 每个 transition 存 15 项：

```text
0  node_inputs
1  edge_inputs
2  current_index
3  node_padding_mask
4  edge_padding_mask
5  edge_mask
6  action
7  reward
8  done
9  next_node_inputs
10 next_edge_inputs
11 next_current_index
12 next_node_padding_mask
13 next_edge_padding_mask
14 next_edge_mask
```

若引入 `action_features`，训练必须存储当前和下一步的 action features。

一种可能的新结构：

```text
0  node_inputs
1  edge_inputs
2  current_index
3  node_padding_mask
4  edge_padding_mask
5  edge_mask
6  action_features
7  action
8  reward
9  done
10 next_node_inputs
11 next_edge_inputs
12 next_current_index
13 next_node_padding_mask
14 next_edge_padding_mask
15 next_edge_mask
16 next_action_features
```

这会影响 `Worker.save_*()`、`driver.py` 的 batch stack、`PolicyNet.forward()`、`QNet.forward()`、`TestWorker.get_observations()` 与 `test_driver.py`。

## 9. 兼容性

该方案会改变模型输入签名，即使 `INPUT_DIM` 仍保持 9，旧 checkpoint 也不能直接兼容新模型。

如果未来实现 basin features，需要为新实验设置新的 `FOLDER_NAME`，例如：

```text
ae_clean_node_9_basin
```

具体命名需要确认。

## 10. 验收标准

建议第一版实现后的基本验收标准：

- 训练和测试路径都能构造 `action_features`。
- `action_features.shape == (1, K_SIZE, ACTION_FEATURE_DIM)`。
- padding action 的特征为 0，且仍由 `edge_padding_mask` 屏蔽。
- 当前节点到自身、不可达节点、多条最短路场景不会产生 `NaN` 或 `Inf`。
- `PolicyNet` 和 `QNet` 均能接收并使用 action features。
- `test_driver.py` 能加载新 basin 模型并完成 `NUM_TEST` 评估。
- 对比基线至少包括当前 9 维模型在同一测试集上的平均路径长度。

## 11. 建议消融实验

建议逐步验证，不要一次加入过多特征：

### 11.1 Basin-3

```text
basin_utility_sum
basin_expected_unknown_gain_sum
basin_frontier_cluster_max
```

目的：验证“方向背后总收益”是否有用。

### 11.2 Basin-5

```text
basin_utility_sum
basin_expected_unknown_gain_sum
basin_frontier_cluster_max
basin_unvisited_ratio
basin_min_dist_to_utility
```

目的：加入访问历史和最近收益距离。

### 11.3 Basin-6

```text
basin_utility_sum
basin_expected_unknown_gain_sum
basin_frontier_cluster_max
basin_unvisited_ratio
basin_min_dist_to_utility
basin_gain_over_entry_cost
```

目的：加入单位入口成本收益。

## 12. 待确认问题

以下问题在实现前需要确认。每个问题下方附有建议与分析，但建议不等于最终需求。

### Q1. 第一版 basin features 具体包含哪些？

候选：

```text
basin_utility_sum
basin_expected_unknown_gain_sum
basin_frontier_cluster_max
basin_unvisited_ratio
basin_min_dist_to_utility
basin_gain_over_entry_cost
```

我的建议：第一版使用 Basin-5，不立即加入 `basin_gain_over_entry_cost`。

分析：前 5 个特征分别覆盖总 frontier 收益、总 unknown 收益、大 cluster、访问冗余、最近有效收益距离。`basin_gain_over_entry_cost` 是组合特征，可能和前几个特征存在强相关，第一版先不加入可以让消融更清晰。

回答：使用 Basin-5

### Q2. basin 统计是否只统计有收益节点？

选项：

```text
A. 统计 basin 内所有可达节点
B. 只统计 utility > 0 或 expected_unknown_gain > threshold 的节点
C. 所有节点参与，但用收益特征自然加权，0 收益节点只影响 unvisited_ratio 等计数特征
```

我的建议：选择 C。

分析：只统计有收益节点会丢失“这个方向有大量未访问但当前暂时无 frontier 的节点”这类信息；统计所有节点但让收益项自然为 0，可以保留结构规模和访问比例，同时不稀释收益 sum。

回答：C

### Q3. `basin_min_dist_to_utility` 的有效收益阈值如何定义？

候选：

```text
node_utility > 0
node_utility > 5
expected_unknown_gain > 0
node_utility > 0 or expected_unknown_gain > threshold
```

我的建议：先用 `node_utility > 0`。

分析：`node_utility` 是 ARiADNE 原始语义，最容易解释；`expected_unknown_gain` 可能在大面积 unknown 附近较密集，使最近收益距离变得过于乐观。后续可以做对比。

回答：先用 `node_utility > 0`

### Q4. 不存在有效收益节点时，`basin_min_dist_to_utility` 如何赋值？

候选：

```text
A. 固定大值，例如 2.0
B. 0
C. 使用当前图最大归一化距离
```

我的建议：使用固定大值 2.0。

分析：当前不可达距离已经使用过 2.0 作为固定大值语义；不存在有效收益时用 0 会和“收益就在当前附近”混淆。

回答：使用固定大值 2.0

### Q5. 多条等长最短路径的 first_hop 如何处理？

候选：

```text
A. 硬分配给 Dijkstra 中先出现的 first_hop
B. 按 neighbor index 固定 tie-break
C. 软分配给多个 first_hop
```

我的建议：第一版使用 B。

分析：软分配更精细，但实现复杂且会改变聚合含义。固定 tie-break 可复现，比依赖 heap 弹出顺序更稳定。

回答：B

### Q6. basin features 如何归一化？

候选：

```text
A. 固定尺度归一化
B. 每个 observation 内按 max 归一化
C. log1p 后固定尺度归一化
```

我的建议：sum 类特征使用 `log1p` 后固定尺度归一化，max/ratio 类特征保持 `[0, 1]`。

分析：sum 类特征会随节点数和地图阶段变化，直接 clip 可能损失大 basin 差异；observation 内 max 归一化会导致不同状态之间尺度不一致。`log1p` 更稳，但需要确认尺度。

回答：sum 类特征使用 `log1p` 后固定尺度归一化，max/ratio 类特征保持 `[0, 1]`

### Q7. basin features 接入 PolicyNet 的方式选哪种？

候选：

```text
A. 拼接到 neighboring feature 后进入 pointer
B. pointer logits + action_bias
```

我的建议：选择 B。

分析：basin features 是当前动作相关统计，不是邻居节点固有语义。作为 action bias 更清楚，也能保留原 pointer 的结构。

回答：B

### Q8. QNet 是否必须同步使用 basin features？

候选：

```text
A. 必须同步接入
B. 只给 PolicyNet 使用
```

我的建议：选择 A。

分析：SAC 中 actor 依赖 critic 的 Q 值学习。如果 critic 没有看到 action features，而 policy 看到 action features，会导致 actor 和 critic 的信息不对称，训练信号可能不稳定。

回答：B

### Q9. 是否保持 `INPUT_DIM = 9`？

候选：

```text
A. 保持 node input 9 维，新增 ACTION_FEATURE_DIM
B. 把 basin 聚合结果也塞进 node_inputs，增加 INPUT_DIM
```

我的建议：选择 A。

分析：basin 是 current-dependent action feature，同一个节点在不同 current 下 basin 归属会变。放进 `node_inputs` 会污染节点固定语义。

回答：选择 A

### Q10. replay buffer 是否接受从 15 项扩展到 17 项？

候选：

```text
A. 接受，显式存 action_features 和 next_action_features
B. 不存，训练时从 node_inputs/edge_inputs 动态重算
```

我的建议：选择 A。

分析：动态重算需要保留 graph 结构和原始环境状态，当前 replay buffer 只存 tensor observation，不足以可靠重建 basin。显式存储最直接。

回答：A

### Q11. 是否需要单独记录 basin 相关训练指标？

候选指标：

```text
selected_basin_utility_sum
selected_basin_expected_unknown_gain_sum
selected_basin_frontier_cluster_max
selected_basin_unvisited_ratio
selected_basin_min_dist_to_utility
```

我的建议：至少记录前三个。

分析：当前训练已经记录 selected expected unknown gain 和 selected frontier cluster size。新增 selected basin 指标能帮助判断模型是否真的在使用 basin 信号，而不是只靠原节点特征。

回答：5个都记录

### Q12. 新实验的 `FOLDER_NAME` 如何命名？

候选：

```text
ae_clean_node_9_basin
ae_clean_node_9_action_basin
ae_clean_node_9_basin5
```

我的建议：如果第一版用 5 个 basin 特征，命名为 `ae_clean_node_9_basin5`。

分析：名字直接编码特征数量，方便后续区分 Basin-3、Basin-5、Basin-6 的消融结果。

回答：ae_clean_node_9_basin5

### Q13. 是否要求第一版保持 reward 完全不变？

候选：

```text
A. 保持 reward 完全不变，只改 observation/model
B. 同时修改 reward
```

我的建议：选择 A。

分析：本 proposal 的目标是验证 action-level basin features 的收益。如果同时改 reward，后续很难判断路径缩短来自 basin 特征还是 reward 变化。

回答：A

### Q14. 是否要求第一版同时支持测试集保存 GIF/trajectory/length？

候选：

```text
A. 完全保持当前测试保存逻辑
B. 只保证 NUM_TEST 平均路径长度评估
```

我的建议：选择 A。

分析：路径可视化对诊断 basin 特征是否减少折返很重要。即使第一阶段主要看平均路径长度，也应保持现有 GIF/trajectory 逻辑可用。

回答：B

### Q15. 是否需要为 basin 计算写单元测试？

候选：

```text
A. 需要，至少覆盖 first_hop 分组、padding、无收益 basin
B. 暂不需要，只做端到端训练/测试
```

我的建议：选择 A。

分析：basin 分组一旦错，模型仍可能训练但语义会偏。这个模块是纯图算法，适合写小规模确定性测试。

回答:A
