# Directional Topological Memory 需求 Proposal

## 1. 背景

当前代码的策略在每一步只从当前节点的邻居中选择下一个节点：

- `Worker.get_observations()` 和 `TestWorker.get_observations()` 构造 `node_inputs`、`edge_inputs`、`current_index`、padding mask 和 `edge_mask`。
- `PolicyNet.output_policy()` 对当前节点的 `K_SIZE` 个邻居输出动作分布。
- `QNet.output_q_values()` 对同一批候选邻居输出 Q value。
- `Env.step()` 将机器人移动到 `next_position`，把位置追加到 `graph_generator.route_node`，更新 belief、frontiers、graph 和 reward。
- `Node.set_visited()` 会把被访问节点的 utility 置为 0。

当前已存在或曾尝试的历史信息主要是：

- `guidepost`：节点是否访问过的二值特征。
- `visit_count`：节点访问次数特征，可通过 `USE_NODE_FEATURE_VISIT_COUNT` 控制。
- `Graph_generator.route_node`：episode 内的路径历史。

问题是，`visit_count_i` 是节点级稀疏信息。机器人走回头路时，不一定每次都经过完全相同的图节点；即使经过访问过的节点，单个节点的访问次数也无法表达“这是无效折返”还是“必须穿过已访问区域去到有价值分支”。

因此，本 proposal 关注的不是简单惩罚回头，而是让网络学会区分：

```text
高历史访问痕迹 + 低后续收益 = 无效回头
高历史访问痕迹 + 高后续收益 = 必要回头
低历史访问痕迹 + 高后续收益 = 新方向优先
```

## 2. 目标

拟设计一个面向回头路判别的表示增强模块，暂命名为：

```text
Directional Topological Memory, DTM
```

核心目标：

1. 将历史轨迹从稀疏节点计数扩展成图上的稠密记忆场。
2. 为每个候选动作估计其对应方向背后的剩余探索价值。
3. 将这些信息作为 action-level features 输入策略和 Q 网络。
4. 不硬性禁止回头，使网络仍能在必要时选择回头。

## 3. 非目标

本方案不应默认做以下事情，除非后续确认：

- 不默认把回头动作直接 mask 掉。
- 不默认对所有访问过的节点增加 reward penalty。
- 不默认移除现有 `utility`、`guidepost`、`utility_over_dist` 等节点特征。
- 不默认继续使用 `visit_count` 作为主创新点。
- 不默认修改 episode 长度、地图数据集、SAC 训练流程和测试集划分。

## 4. 核心方案

### 4.1 轨迹记忆场

将 `route_node` 中的历史访问位置投影到当前 graph 上，构造每个节点的 memory feature：

```text
memory_i = f(节点 i 距离历史路径有多近, 历史访问有多新)
```

推荐语义：

```text
memory_i 越大，表示节点 i 越像处于机器人近期走过的区域或走廊中。
memory_i = 0 表示没有明显历史访问痕迹。
memory_i 接近 1 表示非常接近近期轨迹。
```

可选公式 A，sum-decay：

```text
memory_i = sum_t gamma^(T - t) * exp(-d_G(i, route_t) / sigma)
```

可选公式 B，max-decay：

```text
memory_i = max_t gamma^(T - t) * exp(-d_G(i, route_t) / sigma)
```

其中：

- `d_G` 是当前 collision-free graph 上的最短路距离。
- `T` 是当前 step。
- `t` 是历史访问 step。
- `gamma` 控制时间衰减。
- `sigma` 控制空间扩散半径。

#### 4.1.1 公式选择建议

第一版主方案建议使用公式 B，max-decay：

```text
recent_memory_i = max_t gamma^(T - t) * exp(-d_G(i, route_t) / sigma)
```

公式 A，sum-decay，不建议作为第一版主 memory。它更适合作为后续 ablation 或额外辅助特征。

两种公式的语义不同：

```text
sum-decay 回答：这个区域累计被路径覆盖了多少？
max-decay 回答：这个区域是否接近近期走过的轨迹？
```

本方案的核心目标是让网络意识到“我是不是又走回了近期来过的区域”，并结合方向级收益判断这个回头是否必要。因此，`max-decay` 的语义更直接。

#### 4.1.2 sum-decay 的特点

sum-decay 会把所有历史访问点的影响相加：

```text
memory_i = sum_t gamma^(T - t) * exp(-d_G(i, route_t) / sigma)
```

其中每个历史点的贡献由两部分组成：

- `gamma^(T - t)`：时间衰减。越新的访问点贡献越大，越早的访问点贡献越小。
- `exp(-d_G(i, route_t) / sigma)`：空间衰减。节点 `i` 离历史路径点越近，贡献越大；离得越远，贡献越小。

它的优点是能表达“反复经过”。如果机器人在局部来回震荡，同一条走廊附近的节点会得到较高 memory。

但它有几个风险：

1. 数值会随历史访问累积变大，通常需要 clipping 或归一化。
2. 可能把必须反复经过的主干通道、瓶颈通道标成很高 memory，从而误导网络。
3. episode 后期整体 memory 可能偏高，网络可能学到与时间步相关的偏差。
4. 它表达的是累计交通量，不是“是否接近近期轨迹”。

因此，如果使用 sum-decay，更建议把它命名为：

```text
traffic_memory_i
```

并使用压缩形式避免数值失控：

```text
traffic_memory_i = 1 - exp(-beta * sum_t gamma^(T - t) * exp(-d_G(i, route_t) / sigma))
```

#### 4.1.3 max-decay 的特点

max-decay 只取历史访问点中影响最大的那一个：

```text
memory_i = max_t gamma^(T - t) * exp(-d_G(i, route_t) / sigma)
```

它的直观含义是：

```text
节点 i 是否靠近最近走过的轨迹？
```

优点：

1. 数值天然稳定。因为 `gamma^(T-t) <= 1` 且 `exp(...) <= 1`，所以 `memory_i` 通常在 `[0, 1]`。
2. 不会因为必要通道被多次经过就无限变大。
3. 更适合与 `branch_gain` 搭配，让网络学习：

```text
memory 高 + branch_gain 低 = 不值得回去
memory 高 + branch_gain 高 = 可以回去
```

4. 第一版更容易训练和解释，适合论文中作为主要方法。

因此，第一版推荐：

```text
recent_memory_i = max_t gamma^(T - t) * exp(-d_G(i, route_t) / sigma)
gamma = 0.95
sigma = sensor_range = 80
window = 64
```

然后在特征中使用：

```text
node_inputs: memory_i
action_inputs: next_node_memory, branch_memory_mean
```

如果后续担心 max-decay 不能表达“反复走回同一区域”，可以增加第二个记忆通道，但不建议替代 max-decay：

```text
recent_memory_i  = max-decay 语义，表示是否接近近期轨迹
traffic_memory_i = clipped sum-decay 语义，表示是否反复经过
```

推荐实验顺序：

```text
主实验：max-decay DTM
对比实验：sum-decay DTM
可选增强：max-decay + clipped sum-decay 双通道
```

### 4.2 方向级分支收益

当前策略的动作不是“选择任意节点”，而是“选择当前节点的某个邻居”。因此，回头判别应当是 action-level，而不是纯 node-level。

对当前节点 `c` 的每个候选邻居 `j`，将图中其他节点按最短路第一步归入不同方向：

```text
branch(j) = { i | shortest_path(c -> i) 的第一步是 j }
```

然后为每个候选动作 `c -> j` 聚合该方向背后的价值：

```text
branch_utility_j = sum_{i in branch(j)} utility_i
branch_gain_j = sum_{i in branch(j)} utility_i / (d_G(c, i) + eps)
branch_memory_j = aggregate_{i in branch(j)} memory_i
```

这样，网络可以判断一个动作虽然走向访问过区域，但它背后是否仍然连接着有价值的 frontier。

### 4.3 动作级特征

新增 `action_inputs`，形状建议为：

```text
(batch_size, K_SIZE, ACTION_INPUT_DIM)
```

每个候选邻居对应一行 action feature。建议的最小特征集合：

```text
edge_dist_norm_j
is_immediate_reverse_j
next_node_memory_j
branch_utility_norm_j
branch_gain_norm_j
branch_memory_j
```

含义：

- `edge_dist_norm_j`：当前节点到候选邻居的边长度归一化值。
- `is_immediate_reverse_j`：候选邻居是否是上一步所在节点。
- `next_node_memory_j`：候选邻居本身的轨迹记忆强度。
- `branch_utility_norm_j`：该方向所有节点 utility 的聚合值。
- `branch_gain_norm_j`：该方向单位路径收益聚合值。
- `branch_memory_j`：该方向整体历史访问强度。

这些特征不直接告诉网络“不要回头”，而是提供足够信息让网络学习什么时候该回头。

## 5. 与当前代码的接口变化

### 5.1 Observation 返回值

当前 observation：

```python
observations = (
    node_inputs,
    edge_inputs,
    current_index,
    node_padding_mask,
    edge_padding_mask,
    edge_mask,
)
```

建议改为：

```python
observations = (
    node_inputs,
    edge_inputs,
    action_inputs,
    current_index,
    node_padding_mask,
    edge_padding_mask,
    edge_mask,
)
```

训练 replay buffer 需要同步保存：

```text
current action_inputs
next action_inputs
```

因此 `Worker.episode_buffer`、`save_observations()`、`save_next_observations()` 和 `driver.py` 的 batch stack 逻辑都需要调整。

### 5.2 PolicyNet

当前 `PolicyNet` 用当前节点 embedding 作为 query，用邻居节点 embedding 作为 key 计算 pointer logits。

建议新增 action feature embedding：

```python
action_feature = action_embedding(action_inputs)
neighboring_feature = neighbor_action_fusion(
    concat(neighboring_node_feature, action_feature)
)
```

然后再送入 pointer：

```python
logp = pointer(current_context, neighboring_feature, current_mask)
```

### 5.3 QNet

当前 QNet 对每个动作拼接：

```python
enhanced_current_node_feature
current_node_feature
neighboring_feature
```

建议额外拼接 `action_feature`：

```python
action_features = concat(
    enhanced_current_node_feature,
    current_node_feature,
    neighboring_feature,
    action_feature,
)
```

再经过 `action_embedding` 和 `q_values_layer` 输出 Q value。

### 5.4 Graph_generator

建议新增图算法辅助函数：

```python
get_shortest_path_tree(start_index)
```

返回：

```text
distances: 从 start_index 到所有节点的最短路距离
first_hops: 从 start_index 到每个目标节点的最短路第一步邻居
reachable: 每个节点是否可达
```

注意：当前 `get_normalized_shortest_path_distances()` 已经实现了单源 Dijkstra 的一部分，但没有返回 `first_hops`。可以在它的基础上扩展，避免逐目标调用 `a_star()`。

### 5.5 特征计算位置

建议将特征计算封装到独立函数中，避免 `worker.py` 和 `test_worker.py` 各自复制复杂逻辑。

候选位置：

```text
node_features.py
```

或者：

```text
graph_generator.py
```

建议优先使用独立模块，例如：

```python
build_node_and_action_features(env, robot_position, k_size)
```

这样训练和测试路径可以共享同一套实现，减少不一致风险。

## 6. 参数建议

以下只是建议值，需要确认后才能作为实现默认值：

```python
USE_ACTION_FEATURES = True
USE_TRAJECTORY_MEMORY = True
USE_DIRECTIONAL_BRANCH_FEATURES = True

ACTION_FEATURE_DIM = 6
TRAJECTORY_MEMORY_GAMMA = 0.95
TRAJECTORY_MEMORY_SIGMA = 80
TRAJECTORY_MEMORY_WINDOW = 64
BRANCH_GAIN_EPS = 1e-6
GRAPH_DISTANCE_NORMALIZER = 640
```

如果采用 max-decay memory，则 `memory_i` 通常不需要额外裁剪。若采用 sum-decay memory，建议再做 clipping 或归一化。

## 7. Reward 策略

建议第一阶段不修改 reward，只修改 observation 和网络结构。

原因：

- 如果同时修改 reward 和 representation，后续很难判断性能提升来自哪里。
- 本方案的目标是让网络学习“何时回头”，而不是人工规定“回头不好”。
- 当前 reward 已经包含距离惩罚 `-dist / 64`，过早加入回头惩罚可能误伤必要回头。

可作为第二阶段 ablation 的 reward shaping：

```text
nonproductive_backtrack_penalty =
    action_memory_high
    and branch_gain_low
    and future_frontier_delta_low
```

但该项不建议作为第一版默认实现。

## 8. 实验与 Ablation

建议至少比较：

```text
Baseline ARiADNE
+ visit_count
+ graph_dist_to_current / utility_over_dist
+ trajectory memory only
+ directional branch features only
+ trajectory memory + directional branch features
+ full DTM action-conditioned policy
```

推荐评估指标：

```text
Travel Distance
Explored Rate
Success Rate
Immediate Reverse Rate
Revisited Node Ratio
Revisited Distance Ratio
Local Oscillation Count
Nonproductive Backtrack Distance Ratio
Necessary Backtrack Ratio
Backtrack Productivity Score
Unique Exploration Efficiency
```

其中，前三个是常规探索指标，后八个是回头相关 evaluation metrics。建议八个回头指标都记录，便于后续从不同角度分析“减少无效回头”和“保留必要回头”。

### 8.1 回头相关 Evaluation Metrics

记：

- `c_t`：第 `t` 步所在节点。
- `d_t`：第 `t` 步动作移动距离。
- `memory(c_t)`：节点 `c_t` 的轨迹记忆值。
- `m_th`：判断是否处于历史区域的 memory 阈值。
- `H`：未来收益统计窗口。
- `future_gain(t, H)`：从第 `t` 步开始未来 `H` 步内产生的新探索收益，可用 explored free area 增量或 frontier delta 表示。
- `g_th`：判断未来收益是否有效的阈值。

#### 8.1.1 Immediate Reverse Rate

定义：

```text
IRR = count(c_{t+1} == c_{t-1}) / total_steps
```

含义：机器人是否一步走回上一个节点。

用途：检测最直接的局部来回震荡，例如 `A -> B -> A`。

#### 8.1.2 Revisited Node Ratio

定义：

```text
RNR = count(c_{t+1} has been visited before) / total_steps
```

含义：有多少动作走向已访问节点。

用途：衡量节点级重复访问程度。该指标实现简单，但不能单独说明策略不好，因为必要回头也会访问旧节点。

#### 8.1.3 Revisited Distance Ratio

定义：

```text
RDR = sum(d_t where c_{t+1} has been visited before) / total_travel_distance
```

若使用 DTM 记忆场，也可以定义为：

```text
Visited-Corridor Distance Ratio =
    sum(d_t where memory(c_{t+1}) > m_th) / total_travel_distance
```

含义：在历史区域或已访问节点上行走的距离占比。

用途：相比只统计步数，该指标更贴近探索路径长度效率。

#### 8.1.4 Local Oscillation Count

定义：

```text
LOC_k = count(c_t appears in {c_{t-k}, ..., c_{t-1}})
```

含义：最近 `k` 步内是否重复进入同一节点或局部区域。

用途：检测非一步回头的局部绕圈，例如：

```text
A -> B -> C -> B
A -> B -> C -> A
```

建议第一版记录 `k=4` 和 `k=6` 两个版本，或先固定 `k=6`。

#### 8.1.5 Nonproductive Backtrack Distance Ratio

定义：

```text
NBD = sum(d_t where memory(c_{t+1}) > m_th
                 and future_gain(t, H) < g_th)

NBD Ratio = NBD / total_travel_distance
```

含义：机器人走在历史区域里，并且接下来 `H` 步没有带来明显新探索收益的距离占比。

用途：这是最重要的核心指标之一，直接衡量“无效回头”。

#### 8.1.6 Necessary Backtrack Ratio

定义：

```text
necessary_backtrack_distance =
    sum(d_t where memory(c_{t+1}) > m_th
               and future_gain(t, H) >= g_th)

all_backtrack_distance =
    sum(d_t where memory(c_{t+1}) > m_th)

NBR = necessary_backtrack_distance / (all_backtrack_distance + eps)
```

含义：所有进入历史区域的路径中，有多少最终带来了有效探索收益。

用途：证明方法不是简单禁止回头，而是减少无效回头并保留必要回头。

#### 8.1.7 Backtrack Productivity Score

定义：

```text
BPS = sum(future_gain(t, H) for backtrack steps)
      / (sum(d_t for backtrack steps) + eps)
```

其中 backtrack step 可定义为：

```text
memory(c_{t+1}) > m_th
```

含义：单位回头距离带来的未来探索收益。

用途：提供一个连续值指标，比二值阈值更平滑，可辅助解释回头动作的生产力。

#### 8.1.8 Unique Exploration Efficiency

定义：

```text
UEE = newly_explored_area / total_travel_distance
```

也可以基于 frontier 写成：

```text
UEE_frontier = total_frontier_delta / total_travel_distance
```

含义：单位路径长度带来的独特探索收益。

用途：这不是专门的回头指标，但能反映路径效率。应与回头指标一起报告。

第一版建议优先关注：

```text
Immediate Reverse Rate
Revisited Distance Ratio
Nonproductive Backtrack Distance Ratio
Necessary Backtrack Ratio
Unique Exploration Efficiency
```

但实际日志中应记录全部八个回头相关指标，便于后续论文分析和 ablation。

这些指标需要在 `TestWorker` 或额外 evaluation script 中统计，不建议混入训练 reward。

## 9. 当前代码中的风险点

### 9.1 edge padding 使用节点 0 作为 padding sentinel

当前代码中：

```python
while len(edge) < self.k_size:
    edge.append(0)

edge_padding_mask = torch.where(edge_inputs == 0, one, edge_padding_mask)
```

这会把真实节点 `0` 也当作 padding。若节点 `0` 是当前节点的有效邻居，则会被错误 mask。

建议在正式实验前确认是否修复。可选方案：

```text
padding sentinel 改为 -1
gather 前将 -1 临时替换为 0
edge_padding_mask 根据有效邻居长度构造
```

### 9.2 旧 checkpoint 不兼容

新增 `action_inputs` 和网络层后，旧模型 checkpoint 默认不兼容。建议重新训练，并在 `FOLDER_NAME` 中明确标记新配置。

### 9.3 计算开销

DTM 需要额外图最短路和分支聚合。当前节点数由 `NODE_PADDING_SIZE = 360` 限制，理论上单源 Dijkstra 可以接受，但 32 个 worker 并行训练时仍需要关注开销。

建议：

- branch first-hop 每个 observation 只从当前节点跑一次 Dijkstra。
- trajectory memory 使用有限历史窗口。
- 如果采用 max-decay，可以考虑多源 Dijkstra 或近似扩散。

## 10. 验收标准

实现完成后应满足：

1. 训练和测试路径使用同一套 feature 构造逻辑。
2. `action_inputs.shape == (1, K_SIZE, ACTION_FEATURE_DIM)`。
3. padding action 的特征不会影响策略和 Q 值。
4. 所有新增特征均为有限值，不出现 `NaN` 或 `Inf`。
5. `is_immediate_reverse` 能正确标记返回上一步节点的动作。
6. `branch_gain` 对每个候选邻居只聚合该邻居方向背后的节点。
7. 必要回头动作不会被硬 mask。
8. `PolicyNet` 和 `QNet` 都使用相同的 action features。
9. 旧 checkpoint 不被静默加载到不兼容结构中。
10. 至少有基础单元测试或 smoke test 验证 observation shape 和一次 episode rollout。

## 11. 待确认问题

### Q1. 第一阶段是否只做 observation 与网络结构，不改 reward？

建议：第一阶段不改 reward。

分析：如果同时修改 reward，实验中无法区分是表示增强有效，还是 reward shaping 在起作用。当前目标是论文创新，最好先证明网络能利用 DTM 自己学会减少无效回头。

回答：第一阶段不改 reward。

### Q2. 是否接受新增 `action_inputs`，而不是继续只往 `node_inputs` 里加特征？

建议：接受 `action_inputs`。

分析：当前策略输出就是对当前邻居动作打分。回头问题天然是动作级问题，例如 `current -> previous` 是回头，但 `previous` 这个节点本身不一定永远是坏节点。action-level features 更符合问题结构。

回答：接受 `action_inputs`。

### Q3. 轨迹记忆场使用 graph distance 还是 Euclidean distance？

建议：优先使用 graph distance。

分析：迷宫和障碍环境中，欧氏距离近不代表路径近。graph distance 更符合“是否走回同一条通道”的语义，也更适合作为论文中的拓扑记忆。

回答：优先使用 graph distance。

### Q4. 轨迹记忆公式使用 sum-decay 还是 max-decay？

建议：第一版主方案使用 max-decay。sum-decay 只作为后续 ablation，或经过 clipping 后作为第二个辅助通道。

分析：sum-decay 表达的是累计交通量，适合发现反复经过，但数值会累积，且容易把必要经过的主干通道或瓶颈通道标得过高。max-decay 表达的是是否接近近期轨迹，数值天然稳定，更符合第一版“让网络意识到它来过这里附近”的目标，也更适合与方向级 `branch_gain` 组合，用来区分无效回头和必要回头。

回答：第一版主方案使用 max-decay。

### Q5. 轨迹历史是否只使用最近 `W` 步？

建议：使用最近 `W=64` 步。

分析：当前 episode 最多 128 步，全部历史也不长。但有限窗口可以突出近期行为，降低计算量，并避免很早的访问对当前策略持续产生过强影响。

回答：使用最近 `W=64` 步。

### Q6. `TRAJECTORY_MEMORY_GAMMA` 和 `TRAJECTORY_MEMORY_SIGMA` 取值如何设定？

建议：先用 `gamma=0.95`，`sigma=80`。

分析：`sensor_range=80`，因此 `sigma=80` 有明确物理含义：一个传感器范围内的历史轨迹应有明显影响。`gamma=0.95` 可以让较早轨迹逐渐淡化，但不会立刻消失。

回答：先用 `gamma=0.95`，`sigma=80`。

### Q7. 方向级分支收益是否使用 shortest-path first-hop 划分？

建议：使用 shortest-path first-hop。

分析：这能把“选择某个邻居后会进入哪个拓扑方向”表达清楚，比只看邻居节点自己的 utility 更有效。当前需要在 Dijkstra 中额外记录 `first_hop`。

回答：使用 shortest-path first-hop。

### Q8. action feature 最小集合是否采用本文档中的 6 维？

建议：第一版使用 6 维：

```text
edge_dist_norm
is_immediate_reverse
next_node_memory
branch_utility_norm
branch_gain_norm
branch_memory
```

分析：这组特征覆盖移动成本、直接回头、节点历史、方向收益和方向历史。维度较小，便于 ablation。若效果不足，再考虑加入 `branch_node_count`、`branch_unvisited_ratio`、`best_frontier_progress` 等特征。

回答：第一版使用 6 维：

```text
edge_dist_norm
is_immediate_reverse
next_node_memory
branch_utility_norm
branch_gain_norm
branch_memory
```

### Q9. 是否继续保留现有 `visit_count`？

建议：保留代码开关，但默认不作为 DTM 主方案启用。

分析：`visit_count` 可以作为对比实验，但它不应作为主创新点。保留开关方便证明 DTM 优于简单节点访问计数。

回答：保留代码开关，但默认不作为 DTM 主方案启用。

### Q10. 是否继续保留现有 `utility_over_dist`？

建议：保留。

分析：该特征已经提供节点级单位收益信息，和新的方向级 `branch_gain` 不完全重复。后续 ablation 可以分别验证节点级收益和方向级收益的作用。

回答：保留。

### Q11. 是否修复 edge padding sentinel 为节点 0 的问题？

建议：修复。

分析：这个问题可能影响动作空间和回头统计。如果真实节点 0 被当作 padding mask，实验结果会混入实现偏差。建议在 DTM 实现前或同时修复，并单独记录为 bug fix。

回答：修复。

### Q12. `branch_utility_norm` 如何归一化？

建议：先除以 50 或除以分支聚合上限后 clip 到合理范围。

分析：当前 `node_utility` 在 observation 中使用 `/ 50`。分支聚合是多个节点 utility 的和，数值可能大于单点 utility。需要确认是保留总量信息，还是压缩到稳定范围。

回答：先除以 50 或除以分支聚合上限后 clip 到合理范围。

### Q13. `branch_memory` 使用 mean、max 还是 sum？

建议：第一版使用 mean 和 max 二选一时，优先使用 mean；如果只保留一个，建议 mean。

分析：max 只说明该方向是否存在一个高历史节点，mean 更能表达该方向整体是否是已走区域。若 action feature 维度允许，也可以同时加入 `branch_memory_mean` 和 `branch_memory_max`。

回答：使用 mean。

### Q14. 是否需要把 `memory_i` 也加入 `node_inputs`？

建议：加入。

分析：虽然核心决策使用 action features，但 node encoder 如果能看到每个节点的 memory，attention 也能学习全局历史结构。这样 `node_inputs` 和 `action_inputs` 都携带历史信息。

回答：加入。

### Q15. 是否新建 `node_features.py` 统一训练和测试特征逻辑？

建议：新建。

分析：当前 `worker.py` 和 `test_worker.py` 有重复的 observation 构造逻辑。DTM 特征更复杂，如果继续复制，训练和测试不一致的风险会变高。

回答：新建。

### Q16. 是否接受旧 checkpoint 全部不兼容并重新训练？

建议：接受。

分析：新增 action feature embedding 会改变模型结构，强行迁移旧 checkpoint 成本高且容易引入不可控变量。论文实验应重新训练。

回答：接受。

### Q17. 是否需要在第一版就加入回头相关 evaluation metrics？

建议：需要。

分析：如果论文主张是减少无效回头，仅报告总路径长度不够。建议记录全部八个回头相关指标：`Immediate Reverse Rate`、`Revisited Node Ratio`、`Revisited Distance Ratio`、`Local Oscillation Count`、`Nonproductive Backtrack Distance Ratio`、`Necessary Backtrack Ratio`、`Backtrack Productivity Score`、`Unique Exploration Efficiency`。其中 `Nonproductive Backtrack Distance Ratio` 和 `Necessary Backtrack Ratio` 最能支撑论文主张，因为它们能区分无效回头和必要回头。

回答：记录全部八个回头相关 evaluation metrics。

### Q18. 论文中的方法名是否使用 DTM？

建议：可以暂用 DTM 作为代码和实验名，论文最终命名后再改。

分析：`Directional Topological Memory` 能表达两个重点：方向级动作判别和拓扑历史记忆。若后续更强调“necessary vs redundant backtracking”，可以再调整命名。

回答：FOLDER_NAME = "dtm_action_memory"
