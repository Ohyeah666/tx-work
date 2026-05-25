# Proposal V1: Expected Unknown Gain 与 Frontier Cluster Size 节点特征扩展

## 1. 目标

本阶段只做层级 1：直接节点特征扩展。

在现有 ARiADNE observation 中，为每个图节点新增两个语义特征：

- `expected_unknown_gain`
- `frontier_cluster_size`

本阶段不引入辅助网络、不修改 SAC 训练目标、不修改 attention logits、不改变动作空间、不改变图构造方式。目标是先验证这两个语义量本身是否能进一步缩短测试集总轨迹长度。

该方案应接在已验证有效的三项特征之后：

- `visit_count`
- `graph_dist_to_current`
- `utility_over_dist`

因此，若继续保留原始四维输入和上述三维输入，V1 节点输入建议为 9 维：

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

如果当前工作分支尚未包含前三项特征，则本 V1 应明确作为“在前三项特征基础上的增量方案”，而不是从原始 4 维 ARiADNE 直接扩展。

## 2. 设计动机

ARiADNE 原始 `utility` 表示一个节点可观测 frontier 的数量。这个量有用，但它有两个局限：

1. frontier 数量不等价于实际可打开的未知空间面积。
2. 单个节点看到的 frontier 可能来自小角落，也可能来自大房间入口，原始 `utility` 难以区分二者。

你已经验证过 `visit_count`、`graph_dist_to_current`、`utility_over_dist` 能缩短测试集轨迹长度，这说明模型受益于：

- 历史冗余信息
- 真实图路径成本
- 即时收益/代价比

本阶段新增的两个特征用于补足 `utility` 的质量判断：

- `expected_unknown_gain`：节点附近预计可带来的未知空间观测收益。
- `frontier_cluster_size`：节点关联 frontier 所属未知边界簇的规模。

一句话概括：

```text
已有三特征让模型知道“去这里贵不贵、是否重复、当前收益/距离比如何”；
新增两特征让模型知道“这个收益背后可能有多大空间、是不是值得赌”。
```

## 3. 非目标

本阶段明确不做以下事情：

- 不把 `expected_unknown_gain` 或 `frontier_cluster_size` 作为辅助预测目标。
- 不增加 `gain_head`、`cluster_head` 等模型分支。
- 不修改 `PolicyNet` 或 `QNet` 的 loss。
- 不把 affordance score 加进 pointer attention logits。
- 不修改 reward function。
- 不修改 `Graph_generator.generate_uniform_points()` 的均匀采样方式。
- 不引入 dynamic connectivity graph、skeleton graph 或 graph rarefaction。
- 不引入 RGB、object category、LLM、semantic map 等高级语义。

这些内容可作为后续 V2/V3，但不属于本文件的第一版实现范围。

## 4. 新增特征定义

### 4.1 expected_unknown_gain

#### 含义

对第 `i` 个节点，估计机器人移动到该节点后，在传感器范围内可能新观测到的 unknown 区域规模。

第一版建议定义为：

```text
expected_unknown_gain_i =
  节点 i 传感范围内、从节点 i 到该 unknown cell 视线无碰撞的 unknown cell 数量
```

这里的 unknown cell 指当前 `robot_belief == 127` 的栅格。

该特征不是 ground-truth 未来信息，而是基于当前 partial belief 的可见未知空间估计。它在推理时也可计算，不依赖完整地图真值。

#### 直觉

如果一个节点可见很多 frontier，但 frontier 背后其实很窄，那么 `utility` 可能偏高但实际收益有限。

如果一个节点 frontier 数量中等，但传感范围内有大片 unknown 边界可被打开，那么 `expected_unknown_gain` 可以帮助策略更积极地向该区域移动。

#### 推荐计算方式

对每个节点 `coords`：

1. 在 `robot_belief` 中取以 `coords` 为中心、半径 `sensor_range` 内的候选 unknown cells。
2. 对候选 unknown cell 做距离筛选。
3. 对筛选后的 cell 检查从 `coords` 到该 cell 的 Bresenham line 是否穿过 occupied cell。
4. 统计可见 unknown cell 数。

伪代码：

```python
def compute_expected_unknown_gain(coords, robot_belief, sensor_range):
    unknown_cells = cells where robot_belief == 127 within sensor_range of coords
    visible_unknown_count = 0

    for cell in unknown_cells:
        if line_of_sight_not_blocked_by_occupied(coords, cell, robot_belief):
            visible_unknown_count += 1

    return visible_unknown_count
```

#### 碰撞/视线规则建议

第一版建议：line of sight 只把 occupied cell 视为阻挡。

原因：

- 目标本身就是 unknown cell，如果把 unknown 也当阻挡，则从 free node 看到 unknown 区域时会很快被第一个 unknown 截断，特征会更接近 frontier count，而不是 unknown gain。
- occupied wall 是明确的几何遮挡，应当阻挡。

但这点需要确认，见文末问题 Q1。

#### 归一化建议

建议归一化到稳定范围：

```text
expected_unknown_gain_norm =
  expected_unknown_gain / max_expected_unknown_gain
```

其中 `max_expected_unknown_gain` 推荐使用传感器圆形范围内的理论 cell 数：

```text
max_expected_unknown_gain = pi * sensor_range^2
```

如果按原始高分辨率地图 cell 计数，`sensor_range = 80` 时，该上界约为 `20106`。

也可以使用经验尺度，例如除以 `10000` 或当前 episode 内最大值。第一版更建议使用固定理论上界，避免不同 observation 之间归一化尺度变化。

最终建议：

```text
expected_unknown_gain_norm = clip(expected_unknown_gain / (pi * sensor_range^2), 0, 1)
```

### 4.2 frontier_cluster_size

#### 含义

对第 `i` 个节点，统计它可见 frontier 所属 frontier cluster 的规模。

第一版建议定义为：

```text
frontier_cluster_size_i =
  节点 i 可观测 frontier 中，最大 frontier cluster 的大小
```

使用 `max` 而不是 `sum` 的原因：

- `max` 更像“这个节点是否面向一个大未知入口”。
- `sum` 容易因为多个零碎小 frontier 累加而偏高。
- `max` 对噪声和多区域混合更稳定，适合作为第一版。

#### 直觉

大的 frontier cluster 往往意味着更大的未知边界，可能对应大房间、长走廊或新的区域入口。

小 cluster 可能只是墙角、死胡同边缘或地图噪声。

因此该特征可帮助模型区分：

```text
utility 相同，但一个节点面向大 frontier cluster，另一个节点只面向零碎 frontier。
```

#### 推荐计算方式

在每次 `Env.find_frontier()` 得到 `frontiers` 后：

1. 将 frontier 点映射到 downsampled grid 或原始坐标 grid。
2. 基于 8-neighborhood connected components 对 frontier 做聚类。
3. 得到每个 frontier 点所属 cluster id 与 cluster size。
4. 对每个节点，复用或重新计算其 observable frontiers。
5. 取该节点可见 frontier 对应 cluster size 的最大值。

伪代码：

```python
def compute_frontier_cluster_sizes(frontiers, resolution):
    cluster_id_by_frontier = connected_components(frontiers, neighbor=8)
    cluster_size_by_id = count frontier points in each cluster
    return cluster_id_by_frontier, cluster_size_by_id

def compute_node_frontier_cluster_size(node_observable_frontiers, cluster_lookup):
    if no observable frontier:
        return 0

    sizes = []
    for frontier in node_observable_frontiers:
        cluster_id = cluster_lookup[frontier]
        sizes.append(cluster_size_by_id[cluster_id])

    return max(sizes)
```

#### 和 `node_utility` 的关系

当前 `node_utility` 来自 `Node.observable_frontiers` 的数量。

第一版建议尽量复用 `Node.observable_frontiers`，避免为每个节点重复做 frontier 可见性判断。

理想实现路径：

- 在 `graph_generator.py` 中，节点 utility 更新完成后，同时根据 `Node.observable_frontiers` 计算 `frontier_cluster_size`。
- 将结果保存为 `self.node_frontier_cluster_size`。
- `Env` 暴露该数组。
- `Worker.get_observations()` 和 `TestWorker.get_observations()` 拼接该数组。

#### 归一化建议

由于 frontier 是在 downsampled belief 上提取后乘以 `resolution` 得到的点，cluster size 的数值尺度通常远小于 raw cell count。

建议第一版使用固定尺度归一化：

```text
frontier_cluster_size_norm = clip(frontier_cluster_size / 50, 0, 1)
```

理由：

- 原始 ARiADNE 已将 `node_utility` 除以 `50`。
- `frontier_cluster_size` 与 frontier 点数同一量纲，先沿用 `50` 方便对比。

但如果实际 cluster size 经常明显超过 50，则可改为 `100` 或使用 `log1p`：

```text
frontier_cluster_size_norm = log1p(frontier_cluster_size) / log1p(cluster_size_scale)
```

这点需要通过统计确认，见文末问题 Q5。

## 5. 建议代码影响范围

本文件只描述后续实现方案，不在当前步骤修改代码。

预计后续编码涉及：

- `graph_generator.py`
  - 计算并维护 `node_expected_unknown_gain`
  - 计算并维护 `node_frontier_cluster_size`
  - 在 `generate_graph()` 和 `update_graph()` 返回或暴露新增数组

- `node.py`
  - 可选：让 `Node` 记录 `expected_unknown_gain` 与 `frontier_cluster_size`
  - 可选：提供基于 `observable_frontiers` 的 cluster size 更新方法

- `env.py`
  - 接收并保存新增节点特征数组
  - 在图更新后同步更新新增特征

- `worker.py`
  - 在训练 observation 中拼接两个新增特征
  - 保持 padding、mask、edge input 逻辑不变

- `test_worker.py`
  - 在测试 observation 中拼接同样的新增特征
  - 保证训练/测试输入维度一致

- `parameter.py`
  - 更新 `INPUT_DIM`

- `test_parameter.py`
  - 更新 `INPUT_DIM`

## 6. 推荐实现策略

### 6.1 先实现统计工具，再接入 observation

建议先实现独立函数，便于单元测试和 ablation：

```text
compute_expected_unknown_gain_for_nodes(node_coords, robot_belief, sensor_range)
compute_frontier_cluster_lookup(frontiers, resolution)
compute_frontier_cluster_size_for_nodes(nodes_list, cluster_lookup)
```

然后再接入 `Graph_generator.generate_graph()` 和 `Graph_generator.update_graph()`。

### 6.2 先保证语义正确，再优化速度

`expected_unknown_gain` 如果逐节点逐 unknown cell 做 Bresenham，可能较慢。

第一版可以先实现正确版本，然后根据 profiling 决定是否优化：

- 只在节点附近局部窗口内查 unknown。
- 对 unknown grid 做 stride/downsample。
- 使用射线方向采样替代逐 cell 检查。
- 缓存每个节点局部窗口坐标。

不要在第一版同时引入复杂优化，否则难以判断效果来自特征还是实现细节。

### 6.3 训练/测试必须同步

`worker.py` 与 `test_worker.py` 必须使用完全一致的特征顺序、归一化方式和默认值。

建议在文档和代码中固定 feature order：

```text
FEATURE_ORDER_V1 = [
  "x",
  "y",
  "utility",
  "guidepost",
  "graph_dist_to_current",
  "utility_over_dist",
  "visit_count",
  "expected_unknown_gain",
  "frontier_cluster_size",
]
```

## 7. 验收标准

后续代码实现完成后，应满足：

- `node_inputs.shape[-1] == INPUT_DIM`
- 训练和测试 `INPUT_DIM` 一致
- 所有新增特征均为有限值，不出现 `NaN` 或 `Inf`
- 无 observable frontier 的节点：
  - `frontier_cluster_size = 0`
- 无可见 unknown 的节点：
  - `expected_unknown_gain = 0`
- 两个新增特征 padding 后仍为 0
- `edge_inputs`、`edge_padding_mask`、`node_padding_mask`、`edge_mask` 语义不变
- reward、动作空间、episode length、SAC loss 不变
- 旧 checkpoint 明确不兼容，需要重新训练或按确认策略处理

## 8. 建议实验设计

建议沿用你已经验证有效的三特征模型作为直接对照，而不是只和原始 ARiADNE 对照。

```text
Baseline-0: 原始 ARiADNE 4维
Baseline-1: 4维 + visit_count + graph_dist_to_current + utility_over_dist
V1-A: Baseline-1 + expected_unknown_gain
V1-B: Baseline-1 + frontier_cluster_size
V1-C: Baseline-1 + expected_unknown_gain + frontier_cluster_size
```

核心指标：

- 测试集总轨迹长度
- 平均轨迹长度
- success rate
- explored rate
- makespan / episode step count
- 每步 planning time

建议额外记录：

- `expected_unknown_gain` 分布
- `frontier_cluster_size` 分布
- 两个特征与最终 action 选择概率的相关性
- 失败 episode 中被选中节点的两个特征均值

## 9. 后续 V2/V3 方向，不属于本阶段

### V2: 辅助预测目标

如果 V1 有稳定收益，可将两个特征从“直接输入”升级为辅助监督目标：

```text
encoder(node_inputs) -> enhanced_node_feature
gain_head(enhanced_node_feature) -> pred_expected_gain
cluster_head(enhanced_node_feature) -> pred_cluster_size
```

训练目标变为：

```text
L_total =
  L_policy
  + L_q1
  + L_q2
  + alpha_loss
  + lambda_gain * L_gain
  + lambda_cluster * L_cluster
```

这里的价值在于让 encoder 学会节点未来探索后果的表征，而不是只被动接收人工特征。

### V3: Affordance-biased attention

如果 V2 有收益，可进一步把预测出的 affordance 调制 action attention：

```text
attention_logits =
  query dot key
  + beta_gain * pred_expected_gain
  + beta_cluster * pred_cluster_size
```

这会把 affordance 从普通输入提升为 action scoring bias。

本阶段不实现 V2/V3。

## 10. 编码前必须确认的问题

### Q1. `expected_unknown_gain` 的 line-of-sight 是否只把 occupied cell 视为阻挡？

建议：只把 occupied cell 视为阻挡，unknown cell 不阻挡。

分析：如果 unknown 也阻挡，该特征会退化成“能看到第一层未知边界”，和 frontier count 更接近。只用 occupied 阻挡更能表达从该节点朝未知区域打开的潜在空间。
只把 occupied cell 视为阻挡，unknown cell 不阻挡。

### Q2. `expected_unknown_gain` 使用原始高分辨率 `robot_belief` 计算，还是使用 `downsampled_belief` 计算？

建议：第一版使用原始 `robot_belief`。

分析：原始图更接近真实传感范围和遮挡关系；缺点是计算更慢。若性能成为瓶颈，再考虑 downsample 或 stride。
使用原始 `robot_belief`
### Q3. `expected_unknown_gain` 的候选 unknown cells 是否按完整圆形 sensor range 统计？

建议：使用完整圆形 sensor range。

分析：这与当前传感器更新和 `Node.sensor_range` 语义更一致。若后续考虑真实 LiDAR 视场，可再加入 FOV 限制。
使用完整圆形 sensor range
### Q4. `expected_unknown_gain` 的归一化尺度是否使用 `pi * sensor_range^2`？

建议：使用 `pi * sensor_range^2` 并 clip 到 `[0, 1]`。

分析：固定尺度便于训练/测试一致，也避免 episode 内动态归一化导致同一数值含义变化。
使用 `pi * sensor_range^2` 并 clip 到 `[0, 1]`
### Q5. `frontier_cluster_size` 归一化是否先沿用 `/50`？

建议：先沿用 `/50` 并 clip 到 `[0, 1]`。

分析：当前 `node_utility` 已使用 `/50`。若统计发现 cluster size 大量饱和，再改为 `/100` 或 `log1p`。
先沿用 `/50` 并 clip 到 `[0, 1]`
### Q6. frontier cluster 使用 4-neighborhood 还是 8-neighborhood？

建议：使用 8-neighborhood。

分析：frontier 点来自 downsampled grid，斜向相邻通常也代表同一连续边界。8-neighborhood 更不容易把一条斜边切碎。
使用 8-neighborhood
### Q7. `frontier_cluster_size` 对一个节点使用最大可见 cluster size，还是可见 cluster size 总和？

建议：使用最大可见 cluster size。

分析：`max` 更稳定，更能表达“主入口规模”。`sum` 容易让多个小碎片累加成虚假的大收益。
使用最大可见 cluster size
### Q8. 新特征是否直接追加在现有 7 维之后，形成 9 维输入？

建议：是，直接追加，保持已有特征顺序不变。

分析：这样最利于 ablation，也最少影响你已经验证有效的三项特征。
是，直接追加，保持已有特征顺序不变
### Q9. 本阶段是否接受旧 checkpoint 不兼容并重新训练？

建议：接受不兼容，重新训练。

分析：`INPUT_DIM` 改变后 `initial_embedding` 权重形状变化。迁移旧 checkpoint 虽然可以做，但会引入额外变量，不利于第一版验证。
接受不兼容，重新训练。
### Q10. 是否需要同时在训练和测试 worker 中实现？

建议：必须同时实现。

分析：训练/测试 observation 不一致会导致模型输入语义不一致，评估结果不可解释。
必须同时实现。
### Q11. 如果计算开销明显增加，第一版是否允许使用 stride 近似 `expected_unknown_gain`？

建议：先不近似；只有 profiling 证明瓶颈明显时，再加 stride 版本并单独记录。

分析：第一版应优先验证特征语义。过早近似可能使实验无法判断失败来自特征无效还是估计粗糙。
先不近似；只有 profiling 证明瓶颈明显时，再加 stride 版本并单独记录
### Q12. 是否需要把这两个新特征写入日志或调试图？

建议：至少在 debug/测试脚本中输出分布统计，不一定画图。

分析：如果没有分布统计，很难发现特征全零、饱和、尺度不合理或训练/测试不一致。
在wandb中加入相关量的记录，不要太多，就选代表性的
