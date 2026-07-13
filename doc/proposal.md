# ARiADNE 节点级 CNN Feature Sampling 需求文档

## 1. 背景与动机

当前 ARiADNE 的策略网络主要使用节点图作为输入。`Worker.get_observations()` 从环境中取出 `node_coords`、`graph`、`node_utility` 和 `guidepost`，构造：

```text
node_inputs = [x / 640, y / 640, utility / 50, guidepost]
edge_inputs = 当前节点邻居
edge_mask = 图 attention mask
```

该表示高效，但占据栅格地图中的局部几何信息没有直接进入每个节点表示。也就是说，网络知道“节点在哪里、utility 有多少、图上怎么连”，但不知道某个节点附近是走廊、房间、障碍边界、未知区域边缘，还是 frontier 聚集区域。

之前的全局地图特征方案若只把整张地图压成一个 `map_feature`，信息没有和节点/候选动作对齐，容易变成弱全局上下文，不能稳定帮助 policy 比较当前节点的不同邻居。因此下一版建议改成 **节点级 CNN feature sampling**：CNN 保留空间 feature map，然后按每个图节点的位置采样对应的地图特征，并把该特征融合进每个节点 token。

核心动机：

- 让每个图节点拥有自己的局部地图语义。
- 让 graph encoder 在节点间传播“坐标 + utility + 局部地图结构”的联合信息。
- 让 pointer policy 和 QNet 在比较候选邻居时使用候选节点对应的地图特征，而不是共享同一个全局地图向量。

## 2. 方案概述

新增一条地图分支：

```text
downsampled_belief / semantic_map
  -> CNN backbone
  -> spatial feature map [B, C, Hf, Wf]

node_inputs[:, :, :2]
  -> recover node coords
  -> convert to feature-map sampling grid
  -> grid_sample
  -> node_map_features [B, N, C]

node_inputs
  -> node linear embedding

node embedding + node_map_features
  -> fusion layer
  -> graph encoder
  -> existing decoder / pointer / Q head
```

关键区别：地图特征不是一个全局向量，而是每个节点一个 feature。候选动作的 `neighboring_feature` 会自然包含候选节点附近的地图语义。

## 3. 输入设计

建议继续使用 `downsampled_belief`，而不是原始 `robot_belief`：

```text
robot_belief:       [480, 640]
resolution:         4
downsampled_belief: [120, 160]
```

地图输入建议作为独立张量传入模型：

```text
map_inputs: [B, C, 120, 160]
```

推荐第一版语义通道：

```text
channel 0: free      = downsampled_belief == 255
channel 1: obstacle  = downsampled_belief == 1
channel 2: unknown   = downsampled_belief == 127
channel 3: frontier  = frontier 映射回 downsampled grid
channel 4: position  = 当前机器人位置映射回 downsampled grid
```

该通道设计仍需确认，见第 9 节问题 1。

## 4. 模型结构设计

### 4.1 CNN Spatial Encoder

新增 `SpatialMapEncoder`，输出空间 feature map，不做全局池化：

```python
class SpatialMapEncoder(nn.Module):
    def __init__(self, input_channels, feature_dim):
        ...

    def forward(self, map_inputs):
        # map_inputs: [B, C, 120, 160]
        # returns:    [B, feature_dim, Hf, Wf]
```

第一版建议使用小型 CNN，并明确使用 `GroupNorm`，避免 RL 训练中 `BatchNorm` 的 batch statistics 不稳定问题。

推荐结构：

```text
Input:
  map_inputs [B, MAP_INPUT_CHANNELS, 120, 160]

Block 1:
  Conv2d(MAP_INPUT_CHANNELS, 16, kernel_size=3, stride=2, padding=1)
  GroupNorm(num_groups=4, num_channels=16)
  ReLU

Block 2:
  Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
  GroupNorm(num_groups=4, num_channels=32)
  ReLU

Block 3:
  Conv2d(32, MAP_FEATURE_DIM, kernel_size=3, stride=2, padding=1)
  GroupNorm(num_groups=8, num_channels=MAP_FEATURE_DIM)
  ReLU

Output:
  feature_map [B, MAP_FEATURE_DIM, 15, 20]
```

当 `MAP_FEATURE_DIM = 64` 时，输出为：

```text
[B, 64, 15, 20]
```

如果后续选择 `MAP_FEATURE_DIM = 32`，`GroupNorm(num_groups=8, num_channels=32)` 仍然可用；如果选择其他维度，需要保证 `MAP_FEATURE_DIM` 能被 `num_groups` 整除，或者把最后一层 `GroupNorm` 的 group 数改成兼容值。

不建议第一版使用更深 CNN 或预训练视觉 backbone。当前地图是低维语义栅格，不是自然图像；过大的 CNN 会增加训练不稳定性，也可能掩盖图网络本身的作用。

### 4.2 节点坐标到 feature map 采样坐标

当前 `node_inputs` 的前两维来自：

```python
node_coords = node_coords / 640
```

因此模型中可以从 `node_inputs[:, :, :2]` 恢复原图尺度坐标：

```python
node_xy = node_inputs[:, :, :2] * 640
```

再转成 `grid_sample` 需要的 `[-1, 1]` 坐标：

```python
x_norm = 2 * (node_xy[..., 0] / (map_width * resolution - 1)) - 1
y_norm = 2 * (node_xy[..., 1] / (map_height * resolution - 1)) - 1
grid = torch.stack([x_norm, y_norm], dim=-1)
```

然后采样：

```python
sample_grid = grid.view(B, N, 1, 2)
sampled = F.grid_sample(feature_map, sample_grid, align_corners=True)
node_map_features = sampled.squeeze(-1).permute(0, 2, 1)
```

输出：

```text
node_map_features: [B, N, feature_dim]
```

padding 节点坐标为 0，会采样左上角。后续 `node_padding_mask` 已经会屏蔽 padding 节点，但实现时仍建议对 padding 节点的 `node_map_features` 显式置 0，避免无意义特征影响残差或辅助 loss。

建议将采样逻辑封装为独立模块：

```python
class NodeMapFeatureSampler(nn.Module):
    def forward(self, feature_map, node_inputs, node_padding_mask=None):
        # feature_map:       [B, MAP_FEATURE_DIM, Hf, Wf]
        # node_inputs:       [B, N, 4]
        # node_padding_mask: [B, 1, N] or None
        # returns:           [B, N, MAP_FEATURE_DIM]
```

注意事项：

- `grid_sample` 的 grid 顺序是 `[x, y]`，不是 `[y, x]`。
- `node_inputs` 中坐标使用原代码的归一化方式 `coord / 640`，实现时必须与 worker 保持一致。
- 当前地图宽高可从 `map_inputs.shape[-2:]` 或 `feature_map.shape[-2:]` 推导，不要硬编码 `120 / 160` 到采样模块内部。
- 采样建议使用 `mode="bilinear"`，`padding_mode="border"`，`align_corners=True`。
- 对 padding 节点，将采样后的 feature 置 0。

### 4.3 节点特征融合

当前：

```python
node_feature = self.initial_embedding(node_inputs)
enhanced_node_feature = self.encoder(node_feature, ...)
```

第一版建议使用 **gated residual fusion**，降低新增地图分支扰乱原始图特征的风险。

推荐结构：

```python
node_feature = self.initial_embedding(node_inputs)          # [B, N, EMBEDDING_DIM]
node_map_features = self.sample_node_map_features(...)      # [B, N, MAP_FEATURE_DIM]

map_projection = self.map_projection(node_map_features)     # [B, N, EMBEDDING_DIM]
gate_inputs = torch.cat([node_feature, map_projection], dim=-1)
gate = torch.sigmoid(self.map_gate(gate_inputs))            # [B, N, EMBEDDING_DIM] or [B, N, 1]

fused_node_feature = node_feature + gate * map_projection
enhanced_node_feature = self.encoder(fused_node_feature, ...)
```

推荐 layer 定义：

```python
self.map_projection = nn.Sequential(
    nn.Linear(MAP_FEATURE_DIM, EMBEDDING_DIM),
    nn.ReLU(inplace=True),
    nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM),
)

self.map_gate = nn.Linear(EMBEDDING_DIM * 2, EMBEDDING_DIM)
```

gate 输出维度建议第一版使用 `[B, N, EMBEDDING_DIM]`，让模型能按 embedding 维度控制地图特征注入强度。如果希望更简单，也可以用 `[B, N, 1]`，但表达能力更弱。

为了让训练初期更接近原始图网络，可以将 `map_gate.bias` 初始化为负值，例如：

```python
nn.init.constant_(self.map_gate.bias, -2.0)
```

这样初始 gate 约为：

```text
sigmoid(-2.0) ≈ 0.12
```

模型一开始主要依赖原节点图特征，后续再学习何时增强使用地图信息。

如果不采用 gated residual，也可以使用更简单的 concat projection：

```python
node_feature = self.initial_embedding(node_inputs)
node_map_features = self.sample_node_map_features(map_inputs, node_inputs, node_padding_mask)
node_feature = self.node_map_fusion(torch.cat([node_feature, node_map_features], dim=-1))
enhanced_node_feature = self.encoder(node_feature, ...)
```

但 concat projection 会强制把地图特征混入每个节点 token，训练初期风险更高。因此本 proposal 推荐 gated residual 作为默认实现。

`PolicyNet` 和 `QNet` 应使用同一种融合逻辑，但不一定共享参数。为保持改动清晰，建议先分别在两个网络内各自持有 `SpatialMapEncoder`、`NodeMapFeatureSampler`、`map_projection` 和 `map_gate`；后续如果要做共享 encoder，再单独重构。

### 4.4 diagnostics 输出

为了支持第 8 节的 W&B 诊断，模型 forward 应能可选返回地图分支统计：

```python
diagnostics = {
    "map_feature_std": node_map_features.std().detach(),
    "fusion_gate_mean": gate.mean().detach(),
    "fusion_gate_std": gate.std().detach(),
}
```

建议接口：

```python
logp = policy(..., return_diagnostics=False)
logp, diagnostics = policy(..., return_diagnostics=True)

q_values, attention = q_net(..., return_diagnostics=False)
q_values, attention, diagnostics = q_net(..., return_diagnostics=True)
```

图片类可视化需要节点级 feature norm：

```python
node_map_feature_norm = torch.norm(node_map_features, dim=-1)
```

该值不一定每次 forward 都返回；可以在 diagnostic image 生成路径中单独请求。

## 5. 代码实现范围

### 5.1 新增模块

建议新增：

- `map_input.py`
  - `build_semantic_map_input(...)`
  - 将 `downsampled_belief`、`frontiers`、`robot_position` 转成语义图。
- `replay_schema.py`
  - 用命名常量替代 replay buffer 的硬编码 index。

### 5.2 修改 `worker.py`

`Worker.get_observations()` 新增 `map_inputs`：

```text
observations = (
    node_inputs,
    edge_inputs,
    current_index,
    node_padding_mask,
    edge_padding_mask,
    edge_mask,
    map_inputs,
)
```

训练 replay buffer 从 15 槽扩展到 17 槽：

```text
current observation: node / edge / masks / map_inputs
action
reward / done
next observation: node / edge / masks / next_map_inputs
```

### 5.3 修改 `test_worker.py`

测试 observation 与训练保持一致，也传入 `map_inputs`。

### 5.4 修改 `model.py`

新增：

- `SpatialMapEncoder`
- `NodeMapFeatureSampler`
- `node_map_fusion`

修改：

- `PolicyNet.forward(..., map_inputs)`
- `QNet.forward(..., map_inputs)`
- `PolicyNet.encode_graph(...)` 和 `QNet.encode_graph(...)` 在 graph encoder 前融合节点级地图特征。

### 5.5 修改 `driver.py`

训练 batch 中新增：

```python
map_inputs_batch
next_map_inputs_batch
```

所有 policy / q / target q 调用都需要传入对应地图输入：

```python
dp_policy(..., edge_mask_batch, map_inputs_batch)
dp_q_net1(..., edge_mask_batch, map_inputs_batch)
dp_target_q_net1(..., next_edge_mask_batch, next_map_inputs_batch)
```

### 5.6 修改 `runner.py` / `test_driver.py`

网络初始化需要增加地图相关参数，例如：

```python
PolicyNet(INPUT_DIM, EMBEDDING_DIM, MAP_INPUT_CHANNELS, MAP_FEATURE_DIM)
QNet(INPUT_DIM, EMBEDDING_DIM, MAP_INPUT_CHANNELS, MAP_FEATURE_DIM)
```

## 6. 训练与兼容性

新增地图分支后，旧 checkpoint 不能直接完整加载。建议新方案默认从零训练，避免旧权重和新模块混合带来不稳定。

如果需要复用旧图网络权重，可以后续实现 partial load：

```text
加载 shape 匹配的 graph encoder / decoder / pointer / Q head
CNN 和 fusion layer 随机初始化
```

该点仍需确认，见第 9 节问题 5。

## 7. 测试要求

需要新增 pytest 单元测试：

- `test_map_input.py`
  - 验证 free / obstacle / unknown / frontier / position 通道。
  - 验证 frontier 和 position 的坐标裁剪。
- `test_model_node_map_sampling.py`
  - 验证 `SpatialMapEncoder` 输出空间 feature map。
  - 验证节点坐标采样输出 `[B, N, C]`。
  - 验证 padding 节点 feature 被置 0。
  - 验证 `PolicyNet` forward 输出 `[B, K_SIZE]`。
  - 验证 `QNet` forward 输出 `[B, K_SIZE, 1]`。
- `test_worker_map_observation.py`
  - 验证训练和测试 observation 都包含 `map_inputs`。
  - 验证 replay buffer 槽位数量和 current/next map 保存位置。

## 8. 训练诊断与可视化要求

为了更直观判断节点级 CNN feature sampling 是否真的产生作用，需要在训练和测试路径中增加低频诊断记录。诊断分为图片类和数值类。

### 8.1 图片类指标

图片类指标每 `200` episode 保存一次，可记录到 W&B，也可同时保存到本地诊断目录。不要每 step 都保存，避免训练日志过重。

需要保存以下图片：

1. semantic map 五通道图片
   - 展示 `free / obstacle / unknown / frontier / position` 五个 channel。
   - 用于检查 semantic map 构造是否正确，尤其是 frontier 和当前位置是否映射到正确位置。
   - 如果 frontier 使用 heatmap，应显示 heatmap 的连续强度，而不是二值点。

2. 节点 map feature norm overlay
   - 在当前 `robot_belief` 或 `downsampled_belief` 上绘制节点。
   - 每个节点颜色表示该节点采样到的 `node_map_feature` 的 L2 norm。
   - 用于检查不同节点是否真的采样到了不同的地图特征。
   - 如果所有节点颜色长期接近，说明 CNN/sampling 分支可能没有提供有效区分信息。

3. action probability overlay
   - 在地图上绘制当前节点到候选邻居的边。
   - 边的颜色或粗细表示 policy 对该候选动作的概率。
   - 用于观察 policy 是否倾向于选择靠近 frontier、未知边界或更有探索价值的方向。

建议实现位置：

- 优先在 `test_worker.py` 中实现可视化，便于训练后集中检查策略行为。
- 训练过程中可复用现有 `SAVE_IMG_GAP` 逻辑，或新增 `DIAGNOSTIC_IMG_GAP = 200`。
- 图片上传 W&B 时建议使用清晰命名，例如：

```text
Diagnostics/Semantic Map
Diagnostics/Node Map Feature Norm
Diagnostics/Action Probability
```

### 8.2 数值类指标

数值类指标按现有训练 summary 节奏记录，即每 `SUMMARY_WINDOW = 32` episode 聚合一次，并在 W&B 中查看。

需要新增以下 scalar：

1. `Map/Feature Std`
   - 统计同一 batch 中节点级 map feature 的标准差。
   - 用于判断不同节点采样到的地图特征是否有区分度。
   - 如果长期接近 0，说明地图分支可能退化为近似常量特征。

2. `Fusion/Gate Mean`
   - 仅当采用 gated residual fusion 时记录。
   - 表示模型平均使用地图信息的强度。
   - 如果长期接近 0，说明模型基本忽略地图特征。
   - 如果长期接近 1，说明地图分支可能过强，需要观察是否扰乱图网络。

3. `Fusion/Gate Std`
   - 仅当采用 gated residual fusion 时记录。
   - 表示不同节点或不同样本的 gate 是否有差异。
   - 如果长期接近 0，说明 gate 没有学出针对不同节点的区分。

建议在 `model.py` 中让 policy/q 网络在 forward 时可选返回 diagnostics，例如：

```python
logp, diagnostics = policy(..., return_diagnostics=True)
q_values, attention, diagnostics = q_net(..., return_diagnostics=True)
```

训练主循环在 `driver.py` 中收集这些 diagnostics，并加入现有 `training_data` / W&B metrics 流程。若第一版未采用 gated residual，则只记录 `Map/Feature Std`，并跳过 gate 指标。

## 9. 验收标准

- `conda run -n ariadne pytest -q` 通过。
- `PolicyNet` / `QNet` 均能在 CPU 上完成一次 batch forward。
- `Worker.get_observations()` 和 `TestWorker.get_observations()` 返回格式一致。
- replay buffer 能保存 current/next `map_inputs`。
- `driver.py` 中 current state 与 next state 的所有网络调用都传入正确地图输入。
- 不修改节点图生成逻辑。
- 不改变动作空间，动作仍来自当前节点邻居。
- W&B 中可查看每 32 episode 聚合记录的 `Map/Feature Std`。
- 若采用 gated residual，W&B 中可查看每 32 episode 聚合记录的 `Fusion/Gate Mean` 和 `Fusion/Gate Std`。
- 每 200 episode 可以保存或上传一次 semantic map、节点 map feature norm overlay、action probability overlay。

## 10. 待确认问题

### 问题 1：地图语义图是否继续使用五通道？

可选方向：

- A. 使用 5 通道：free / obstacle / unknown / frontier / position。
- B. 使用 4 通道：free / obstacle / unknown / frontier。
- C. 使用更强 frontier 表达：free / obstacle / unknown / frontier heatmap / position。

我的建议：选择 C。

分析：节点级采样已经让地图特征和节点对齐，但 sparse binary frontier 经过 stride CNN 后仍可能弱化。frontier heatmap 或轻微膨胀后的 frontier channel 更容易被 CNN 捕捉。若希望先控制变量，可选 A。

需要你确认：第一版使用 A 还是 C？

回答：C，请你认真思考衡量，为 frontier heatmap 选择一个合适的扩散参数。

### 问题 2：节点地图特征维度设置多少？

可选方向：

- A. `MAP_FEATURE_DIM = 32`
- B. `MAP_FEATURE_DIM = 64`
- C. `MAP_FEATURE_DIM = 128`

我的建议：选择 B。

分析：当前 graph embedding 是 128。节点地图特征设为 64 能提供足够地图信息，同时避免 CNN 分支过强、训练不稳定。128 表达力更强但参数和过拟合风险更高。

需要你确认：`MAP_FEATURE_DIM` 取多少？

回答：B。

### 问题 3：融合方式用 concat projection 还是 gated residual？

可选方向：

- A. concat 后 linear projection。
- B. gated residual：`node_feature + gate * map_projection(map_feature)`。
- C. concat projection + residual。

我的建议：选择 B。

分析：你已经观察到地图增强版本可能不如原始模型，说明新模态可能扰乱原图网络。gated residual 可以让模型一开始更保守地使用地图信息，再逐渐学习何时依赖地图。

需要你确认：第一版是否使用 gated residual？

回答：使用 gated residual，并且请在wandb记录的训练参数中加入gate相关参数记录。

### 问题 4：是否增加 node utility 辅助预测任务？

可选方向：

- A. 不加，先只改主网络。
- B. 加，用节点地图特征预测 `node_utility`。

我的建议：第一版选择 A，第二阶段再加 B。

分析：辅助任务可能帮助 CNN 学到和探索相关的局部地图表示，但会增加 loss、driver 训练逻辑和权重调参。为了先验证节点级采样本身，第一版建议不加。

需要你确认：是否第一版加入 auxiliary loss？

回答：A。

### 问题 5：旧 checkpoint 如何处理？

可选方向：

- A. 新模型从零训练。
- B. partial load 旧图网络权重，CNN/fusion 随机初始化。

我的建议：选择 A。

分析：节点输入融合位置变化后，即使部分层 shape 匹配，特征分布也会变化。从零训练更干净，便于判断新结构是否有效。

需要你确认：是否从零训练？

回答：A。

### 问题 6：是否保留配置开关回到原始图网络？

可选方向：

- A. 不保留，直接实现新结构。
- B. 保留 `USE_MAP_INPUT` 开关。

我的建议：选择 B。

分析：你已经有原始模型作为对照，保留开关有利于 ablation 和快速回退。缺点是代码分支略多。

需要你确认：是否需要保留开关？

回答：A。

### 问题 7：地图输入在 replay buffer 中保存什么格式？

可选方向：

- A. 保存 `uint8` semantic map，训练时转 float。
- B. 保存 `float32` semantic map。
- C. 保存原始 `downsampled_belief + frontiers + position`，训练时重建 semantic map。

我的建议：选择 A。

分析：A 实现简单且内存可控。B 内存压力较大。C 更省一些存储但训练 batch 重建逻辑复杂，容易引入 CPU 开销。

需要你确认：replay buffer 是否保存 `uint8` semantic map？

回答：A。
