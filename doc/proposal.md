# 节点输入关键特征增强需求 Proposal

## 1. 背景

当前 ARiADNE 的节点输入特征在 `worker.py` 与 `test_worker.py` 的 `get_observations()` 中构造：

- `node_coords / 640`
- `node_utility / 50`
- `guidepost`

拼接后形成 `node_inputs = [x, y, utility, guidepost]`，对应 `parameter.py` 与 `test_parameter.py` 中的 `INPUT_DIM = 4`。`PolicyNet` 与 `QNet` 在 `model.py` 中通过 `nn.Linear(input_dim, embedding_dim)` 接收该维度。

当前代码已有图结构和最短路基础：

- `graph_generator.py` 负责生成 collision-free graph。
- `graph.py` 中的 `Edge.length` 表示图边长度。
- `graph.py` 中的 `a_star()` 可返回从起点到终点的最短路径和加权路径距离。
- `Graph_generator.find_shortest_path()` 已封装基于坐标的最短路查询。

当前访问历史只有 `Graph_generator.route_node` 和二值 `guidepost`，尚未记录每个节点的累计访问次数。

## 2. 目标

第一阶段只在现有节点输入上新增三个最关键特征：

- `graph_dist_to_current_i`
- `utility_over_dist_i`
- `visit_count_i`

目标是让网络更直接获得：

- 从当前位置到每个节点的真实图路径移动成本。
- 每个节点的单位路径收益。
- 每个节点的历史访问次数。

本阶段不新增 `dx`、`dy`、`euclidean_dist_to_current`、`hop_dist_to_current`、`steps_since_last_visit`、`is_dead_end_like`。这些特征可作为后续 ablation 或第二阶段需求再评估。

## 3. 现有代码约束

- 训练路径：`Worker.get_observations()` 负责构造节点输入、节点 padding mask、当前节点索引、邻接节点输入、边 padding mask、edge mask。
- 测试路径：`TestWorker.get_observations()` 有一份相似逻辑，但当前不做节点 padding，`node_padding_mask = None`。
- 当前动作只能从 `edge_inputs[current_index]` 中选择邻接节点；`PolicyNet` 输出当前节点邻居上的动作分布。
- `Env.step()` 根据 `robot_position` 与 `next_position` 的欧氏距离更新 `travel_dist` 与 reward，并将新位置追加到 `graph_generator.route_node`。
- `Node.set_visited()` 会将该节点 utility 置为 0；这会影响后续 `node_utility`。
- `Graph_generator.update_graph()` 会在新探索区域追加新节点并重建 collision-free graph。
- `graph.py` 的 `a_star()` 当前返回加权路径距离，可用于计算 `graph_dist_to_current_i`。

## 4. 新增特征定义

### 4.1 graph_dist_to_current_i

含义：从当前机器人所在图节点到第 `i` 个节点，沿当前 collision-free graph 行走的最短路径总边长。

它不是节点坐标之间的直线距离，而是图上的 shortest path distance。因此在迷宫或有墙体阻隔的场景中，它比欧氏距离更接近真实移动成本。

计算方式：

1. 通过 `env.find_index_from_coords(robot_position)` 得到当前节点 `current_idx`。
2. 对每个目标节点 `target_idx`，在当前 graph 上求最短路径。
3. 将路径上所有边的 `Edge.length` 相加，得到 `graph_dist_to_current_i`。

当前代码可复用 `graph.py` 的 `a_star()`：

```python
route, dist = a_star(current_idx, target_idx, node_coords, graph)
graph_dist_to_current_i = dist
```

当前节点到自身时：

```text
graph_dist_to_current_i = 0
```

待确认：

- 最短路是否使用当前 `Graph` 的有向边，还是将边视为无向边。
- 不可达节点的 `graph_dist_to_current_i` 使用什么哨兵值。
- 是否允许为效率改为单源最短路实现，只要输出语义一致。

### 4.2 utility_over_dist_i

含义：第 `i` 个节点的单位路径收益。

计算方式：

```text
utility_over_dist_i = utility_i / (graph_dist_to_current_i + eps)
```

该特征把节点收益和到达成本组合在一起，比单独的 `utility` 更贴近短路径探索目标。

需要保证该值不会出现 `NaN` 或 `Inf`。

待确认：

- `utility_i` 使用原始 utility，还是使用当前 observation 中的 `utility / 50`。
- `graph_dist_to_current_i` 使用原始路径距离，还是使用归一化后的路径距离。
- `eps` 取值。
- 当前节点 `graph_dist_to_current_i = 0` 时，`utility_over_dist_i` 是否强制设为 0。
- 不可达节点的 `utility_over_dist_i` 如何赋值。

### 4.3 visit_count_i

含义：第 `i` 个节点在当前 episode 内累计被机器人访问的次数。

该特征用于让网络区分“从未访问过”“访问过一次”“多次重复访问”的节点，减少来回走和重复探索。

当前代码只有：

- `Graph_generator.route_node`：记录访问过的位置序列。
- `guidepost`：二值特征，表示节点是否访问过。

因此需要新增访问计数状态。建议由 `Env` 或 `Graph_generator` 管理，并在 `Env.step()` 中每次机器人移动到新节点后更新。

待确认：

- 起点是否计入 `visit_count_i`。
- 同一节点被多次回访时是否每次都递增。
- 新增节点首次出现在图中时，`visit_count_i` 初始值是否为 0。
- `visit_count_i` 是否只统计当前 episode，不跨 episode 累积。

## 5. 节点输入维度

本阶段基于现有输入追加三个特征，不主动移除现有 `guidepost`。

若保留当前特征并追加三项，则节点输入从 4 维变为 7 维：

```text
[x, y, utility, guidepost,
 graph_dist_to_current,
 utility_over_dist,
 visit_count]
```

对应需要将：

- `parameter.py` 中的 `INPUT_DIM` 从 `4` 改为 `7`
- `test_parameter.py` 中的 `INPUT_DIM` 从 `4` 改为 `7`

待确认：是否继续保留 `guidepost`。如果后续决定用 `visit_count > 0` 替代 `guidepost`，则最终输入维度为 6。

## 6. 数据结构与代码影响范围

预计涉及文件：

- `worker.py`：训练 observation 拼接新增节点特征，保持 padding 和 mask 行为不变。
- `test_worker.py`：测试 observation 拼接同样的新增节点特征。
- `parameter.py`：更新训练 `INPUT_DIM`。
- `test_parameter.py`：更新测试 `INPUT_DIM`。
- `graph_generator.py`：提供图最短路距离计算能力，并维护或暴露访问计数状态。
- `env.py`：初始化和每次 `step()` 后更新访问计数。

可选优化：

- 若逐节点调用 `a_star()` 性能不足，可新增单源最短路计算，从当前节点一次性得到到所有节点的 `graph_dist_to_current`。

## 7. 验收标准

- 训练与测试的 `node_inputs.shape[-1]` 均等于确认后的 `INPUT_DIM`。
- 当前节点的 `graph_dist_to_current_i = 0`。
- 所有新增数值特征均为有限值，不出现 `NaN` 或 `Inf`。
- `utility_over_dist_i` 对距离为 0 和不可达节点的行为符合确认后的规则。
- `visit_count_i` 在节点被访问后按确认规则递增。
- `guidepost` 的行为保持现状，除非确认由 `visit_count > 0` 替代。
- `edge_inputs`、`edge_padding_mask`、`node_padding_mask`、`edge_mask` 的语义不因新增节点特征改变。
- 旧模型 checkpoint 的加载行为符合确认结果：要么明确不兼容并重新训练，要么提供迁移策略。

## 8. 待确认问题

1. `graph_dist_to_current_i` 是否归一化？如果归一化，是否除以 640？
除以 640。
2. `utility_over_dist_i` 中的 utility 和 distance 使用原始值还是归一化值？
用已经归一化后的 utility 和 graph_dist 计算。
3. `eps` 应取多少？
eps = 1e-6。
4. 当前节点的 `utility_over_dist_i` 是否强制设为 0？
强制设为 0。
5. 图最短路使用当前图的有向边，还是将边视为无向边？
按无向图语义计算
6. 不可达节点的 `graph_dist_to_current_i` 和 `utility_over_dist_i` 应如何赋值？
归一化后设为 1.0 或 2.0 以上的固定大值。utility_over_dist 对不可达节点设为 0
7. `visit_count_i` 是否包含起点初始访问？
起点初始化为 1，并且 guidepost 对起点也保持已访问
8. 同一节点被多次回访时，`visit_count_i` 是否每次递增？
每次机器人移动到该节点都递增 1
9. `visit_count_i` 是否只统计当前 episode？
只统计当前 episode
10. 是否继续保留现有 `guidepost`？
保留
11. 本阶段是否只改 observation 输入特征，不调整 reward、动作空间、episode 长度和训练超参数？
只改 observation 输入特征
12. 是否接受新增特征后旧 checkpoint 不兼容，并重新训练模型？
接受不兼容，重新训练