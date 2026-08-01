# 动作级 `edge_dist_norm_j` 迁移需求 Proposal

## 1. 本次范围

本 proposal 面向后续代码实现，用于把本地分支 `tx-dist-pointer-memory-i` 中的动作级距离特征 `edge_dist_norm_j` 迁移到当前分支 `tx-cnn-dist`。

本次只迁移动作级特征管线：

```text
edge_dist_norm_j = edge_length(current_node, candidate_node_j) / GRAPH_DISTANCE_NORMALIZER
```

明确不做：

- 不迁移 pointer-level distance-aware attention。
- 不新增 `pointer_distance_scale_raw`、`USE_DISTANCE_AWARE_POINTER` 或 pointer logit bias。
- 不修改 `SingleHeadAttention` 的打分公式。
- 不迁移 decoder current-to-all distance-aware attention。
- 不迁移 encoder edge-length attention bias。
- 不默认迁移 trajectory memory、branch action features、visit count、backtracking metrics 等目标分支中的其它实验内容。

## 2. 当前代码基线

当前工作分支为 `tx-cnn-dist`，核心特征是 CNN map-inputs：

- `parameter.py` 当前 `INPUT_DIM = 4`，节点输入为 `[x, y, utility, guidepost]`。
- `parameter.py` 当前有 `MAP_INPUT_CHANNELS = 5`、`MAP_FEATURE_DIM = 64`、`MAP_GATE_BIAS_INIT`、`MAP_LR`。
- `worker.py` / `test_worker.py` 当前 observation tuple 为：

```text
node_inputs,
edge_inputs,
current_index,
node_padding_mask,
edge_padding_mask,
edge_mask,
map_inputs
```

- `replay_schema.py` 当前有 17 个 replay slots，其中 `MAP_INPUTS = 6`，`NEXT_MAP_INPUTS = 16`。
- `model.py` 当前 `PolicyNet` 和 `QNet` 都保留 map encoder、node map sampler、node map fusion 和 diagnostics。
- 当前 `PolicyNet.output_policy()` 只用候选邻居节点 embedding 进入 pointer。
- 当前 `QNet.output_q_values()` 拼接：

```text
enhanced_current_node_feature,
current_node_feature,
neighboring_feature
```

然后通过 `action_embedding = nn.Linear(embedding_dim * 3, embedding_dim)` 输出动作 Q 值。

迁移必须保留上述 CNN map-inputs 主线，不应把当前代码回退成目标分支中不带 map 输入的模型。

## 3. 目标分支可复用内容

`tx-dist-pointer-memory-i` 中与动作级 `edge_dist_norm_j` 直接相关的内容包括：

- `parameter.py`
  - `USE_ACTION_FEATURES`
  - `USE_ACTION_FEATURE_EDGE_DIST`
  - `ACTION_FEATURE_DIM`
  - `GRAPH_DISTANCE_NORMALIZER = 640`
  - `DISTANCE_ATTENTION_MAX_NORM = 2.0`
  - `PADDING_NODE_INDEX = -1`
- `node_features.py`
  - `ACTION_FEATURE_EDGE_DIST = 'edge_dist_norm'`
  - `get_active_action_feature_names()`
  - `get_action_feature_dim()`
  - `get_action_feature_indices()`
  - `build_action_inputs()`
  - `get_edge_distance()`
- `observations.py`
  - 把 numpy action feature 转成 torch tensor，形状为 `[1, K_SIZE, ACTION_FEATURE_DIM]`。
- `model.py`
  - `PolicyNet` 使用 `action_input_embedding(action_inputs)`，再和候选邻居 embedding 融合。
  - `QNet` 使用 `action_input_embedding(action_inputs)`，再拼入 action value head。
- `tests/test_node_features.py`
  - 验证 edge distance 归一化、padding 为 0、clip 生效。
- `tests/test_model_action_inputs.py`
  - 验证 Policy/QNet 能消费 action inputs，并保持 padding/stay mask。

目标分支还包含 pointer distance bias，但该部分不属于本次迁移范围。

## 4. 需要迁移的行为定义

### 4.1 特征含义

对当前节点 `i` 的第 `j` 个候选动作节点 `next_j`：

```text
edge_dist_norm_j = edge_length(i, next_j) / GRAPH_DISTANCE_NORMALIZER
```

其中：

- 优先使用 graph 中现有边对象的 `edge.length`。
- 如果边对象不存在，fallback 为两个节点坐标的欧氏距离。
- `GRAPH_DISTANCE_NORMALIZER` 建议沿用 `640`，因为当前坐标也按 `640` 归一化。
- padding slot 的 `edge_dist_norm_j` 必须为 `0.0`。
- stay/current slot 的距离自然为 `0.0`，但仍由现有 stay mask 屏蔽，不能依赖距离值屏蔽。

### 4.2 特征形状

建议保持目标分支的 tensor 形状：

```text
numpy:  action_inputs.shape == [K_SIZE, ACTION_FEATURE_DIM]
torch:  action_inputs.shape == [batch_size, K_SIZE, ACTION_FEATURE_DIM]
```

若只启用 `edge_dist_norm_j`：

```text
ACTION_FEATURE_DIM = 1
edge_dist_norm column index = 0
```

### 4.3 模型消费方式

本次不是显式 attention bias。动作距离只作为 learnable action feature 输入模型：

PolicyNet：

```text
neighboring_feature_j = enhanced_node_feature[next_j]
action_feature_j = action_input_embedding(action_inputs_j)
neighboring_feature_j = neighbor_action_fusion([neighboring_feature_j, action_feature_j])
logp = pointer(enhanced_current_node_feature, neighboring_feature, current_mask)
```

QNet：

```text
action_feature_j = action_input_embedding(action_inputs_j)
q_input_j = [
    enhanced_current_node_feature,
    current_node_feature,
    neighboring_feature_j,
    action_feature_j,
]
q_value_j = q_values_layer(action_embedding(q_input_j))
```

模型不应在 pointer logits 上额外加：

```text
-softplus(scale) * edge_dist_norm_j
```

该项属于下一步 pointer-level distance-aware attention。

## 5. 建议的当前分支适配方案

### 5.1 参数

建议在 `parameter.py` 增加最小动作特征配置：

```python
USE_ACTION_FEATURES = True
USE_ACTION_FEATURE_EDGE_DIST = True
ACTION_FEATURE_DIM = int(USE_ACTION_FEATURES) * int(USE_ACTION_FEATURE_EDGE_DIST)
GRAPH_DISTANCE_NORMALIZER = 640
DISTANCE_ATTENTION_MAX_NORM = 2.0
PADDING_NODE_INDEX = -1
```

注意：

- `DISTANCE_ATTENTION_MAX_NORM` 名字来自目标分支，但本次它只用于 `edge_dist_norm_j` clipping，不用于 attention bias。
- 如觉得命名会误导，可改名为 `EDGE_DIST_MAX_NORM`，但这会偏离目标分支命名。
- 如果默认启用 `ACTION_FEATURE_DIM = 1`，旧 checkpoint 与新模型结构不兼容，应重新训练。

### 5.2 特征构造 helper

不要整文件照搬目标分支 `node_features.py`，因为目标分支包含 trajectory memory、branch features、visit count 和 graph shortest path tree，当前分支没有这些默认状态。

建议新建一个当前分支专用的轻量 helper，二选一：

```text
方案 A：新增 action_features.py，仅负责 edge_dist_norm_j。
方案 B：新增 node_features.py，但只放 edge_dist_norm_j 需要的最小内容。
```

建议选 A，理由是当前迁移目标是动作级 edge distance，不应给后续 Codex 造成“要迁移整套 DTM/node_features”的暗示。

核心 helper 建议提供：

```python
ACTION_FEATURE_EDGE_DIST = "edge_dist_norm"

def get_action_feature_dim():
    ...

def get_action_feature_indices():
    ...

def build_edge_dist_action_inputs(
    env,
    current_index,
    edge_inputs_np,
    edge_padding_mask_np,
    normalizer,
    max_norm,
):
    ...

def get_edge_distance(env, current_index, next_index, normalizer, max_norm=None):
    ...
```

`get_edge_distance()` 逻辑建议沿用目标分支：

1. 从 `env.graph` 中取 `graph_edges[str(current_index)][str(next_index)].length`。
2. 如果找不到边对象，则用 `np.linalg.norm(env.node_coords[current_index] - env.node_coords[next_index])`。
3. 除以 normalizer。
4. 如果 `max_norm is not None`，clip 到 `[0, max_norm]`。
5. 返回 float。

### 5.3 Observation tuple

当前 tuple 中没有 `action_inputs`，后续必须同时修改训练和测试路径。

推荐模型 forward 层用显式 keyword 方式保持兼容：

```python
policy(
    node_inputs,
    edge_inputs,
    current_index,
    node_padding_mask,
    edge_padding_mask,
    edge_mask,
    map_inputs,
    action_inputs=action_inputs,
)
```

或者改成目标分支顺序：

```python
policy(
    node_inputs,
    edge_inputs,
    action_inputs,
    current_index,
    node_padding_mask,
    edge_padding_mask,
    edge_mask,
    map_inputs,
)
```

建议选 keyword 方式，理由是当前代码中 `PolicyNet`/`QNet` 调用点较多，保留原位置参数顺序能降低误改风险。

### 5.4 Replay schema

当前 `replay_schema.py` 已有 map slots，不能套用目标分支的硬编码 17-slot 顺序。

有两种可选 schema：

方案 A：append 新字段，减少现有 index churn。

```python
ACTION_INPUTS = 17
NEXT_ACTION_INPUTS = 18
BUFFER_SIZE = 19
```

方案 B：重排为 current observation 和 next observation 分组更整齐。

```text
node, edge, action_inputs, current_index, masks, edge_mask, map_inputs,
action, reward, done,
next_node, next_edge, next_action_inputs, next_current_index, next_masks, next_edge_mask, next_map_inputs
```

建议选 A，理由是当前训练代码和 tests 已经引用 `MAP_INPUTS = 6`、`NEXT_MAP_INPUTS = 16`；append 能保留现有 slot 含义，降低迁移风险。

### 5.5 Padding sentinel

目标分支使用 `PADDING_NODE_INDEX = -1`，并在模型 gather 前执行：

```python
safe_edge_inputs = edge_inputs.clamp(min=0)
```

当前分支用 `0` 同时表示 padding 和节点 0，这会导致一个边界问题：

- 当当前节点不是 0 且有效邻居包含节点 0 时，`edge_padding_mask = edge_inputs == 0` 会把节点 0 误判为 padding。

这个问题并非 `edge_dist_norm_j` 独有，但迁移动作特征时会更明显，因为 padding/action feature 必须严格对齐。

建议一并切换到 `PADDING_NODE_INDEX = -1`：

- `worker.py` 和 `test_worker.py` padding edge slot 使用 `-1`。
- `edge_padding_mask` 使用 `edge_inputs == PADDING_NODE_INDEX`。
- `PolicyNet` 和 `QNet` 在 `torch.gather()` 前对 `edge_inputs.clamp(min=0)`。
- stay/current slot 仍单独执行 `current_mask[:, :, 0] = 1`。

如果暂不切换 padding sentinel，也能做最小迁移，但必须接受当前节点 0 作为有效邻居时可能被错误 mask 的行为。

### 5.6 Worker/TestWorker

训练路径 `worker.py` 需要：

1. 继续构造 `map_inputs`，不要删除 CNN map-inputs。
2. 在 edge_inputs 和 edge_padding_mask 构造完成后，构造 `action_inputs`。
3. 返回 observation 时包含 `action_inputs`。
4. `save_observations()` 保存 `ACTION_INPUTS`。
5. `save_next_observations()` 保存 `NEXT_ACTION_INPUTS`。

测试路径 `test_worker.py` 需要同样构造 `action_inputs`，并在 `select_node()` 调用 policy 时传入。

训练和测试 observation tuple 必须字段一致；区别只应该是训练路径有 node padding，测试路径 `node_padding_mask = None`。

### 5.7 Model

`PolicyNet.__init__` 建议新增 `action_input_dim=0`，但放在参数列表尾部并用 keyword 传入，避免破坏当前 positional 调用：

```python
class PolicyNet(nn.Module):
    def __init__(
        self,
        input_dim,
        embedding_dim,
        map_input_channels=5,
        map_feature_dim=64,
        map_resolution=4,
        gate_bias_init=-2.0,
        action_input_dim=0,
    ):
        ...
```

`QNet.__init__` 同理。

PolicyNet 初始化：

```python
if self.action_input_dim > 0:
    self.action_input_embedding = nn.Linear(action_input_dim, embedding_dim)
    self.neighbor_action_fusion = nn.Linear(embedding_dim * 2, embedding_dim)
```

QNet 初始化：

```python
if self.action_input_dim > 0:
    self.action_input_embedding = nn.Linear(action_input_dim, embedding_dim)
    self.action_embedding = nn.Linear(embedding_dim * 4, embedding_dim)
else:
    self.action_embedding = nn.Linear(embedding_dim * 3, embedding_dim)
```

`encode_graph()`、map encoder、map fusion、diagnostics 不应改变。

### 5.8 Driver/Runner

`driver.py` 和 `runner.py` 初始化模型时应显式传：

```python
PolicyNet(
    INPUT_DIM,
    EMBEDDING_DIM,
    MAP_INPUT_CHANNELS,
    MAP_FEATURE_DIM,
    map_resolution=4,
    gate_bias_init=MAP_GATE_BIAS_INIT,
    action_input_dim=ACTION_FEATURE_DIM,
)
```

`QNet` 同理。

训练 batch 解包应新增：

```python
action_inputs_batch = torch.stack(rollouts[ACTION_INPUTS]).to(device)
next_action_inputs_batch = torch.stack(rollouts[NEXT_ACTION_INPUTS]).to(device)
```

调用 policy/q/target_q 时传入对应 current/next action inputs。

`map_lr_param_groups()` 当前只处理 `map_encoder` 和 `node_map_fusion` 的学习率。动作特征 embedding 默认走 backbone LR 即可，不建议单独分组，除非后续实验确认需要。

## 6. 建议测试

后续实现完成后，建议至少增加/更新以下测试：

1. `test_action_features.py`
   - fake env 中 graph edge length 为 0、1、较大值。
   - 验证 `edge_dist_norm_j = edge.length / 640`。
   - 验证 padding slot 为 0。
   - 验证 `DISTANCE_ATTENTION_MAX_NORM` clipping 生效。
   - 验证缺失 edge 时 fallback 到欧氏距离。

2. `tests/test_model_node_map_sampling.py`
   - 增加 `action_inputs` + `action_input_dim=1` 的 PolicyNet forward。
   - 增加 `action_inputs` + `action_input_dim=1` 的 QNet forward。
   - 保持 map diagnostics 存在。
   - 验证 edge padding mask 不被 in-place 修改。

3. `tests/test_worker_map_observation.py`
   - observation tuple 包含 `action_inputs`。
   - replay buffer 新 slots 有 current 和 next action inputs。
   - `map_inputs` slot 仍然存在且 shape 不变。

4. `test_worker.py` 或测试专用 smoke
   - 测试路径 observation 也包含 `action_inputs`。
   - 测试路径不做 node padding 时 action input shape 仍为 `[1, K_SIZE, 1]`。

建议测试命令：

```bash
conda run -n ariadne pytest tests/test_worker_map_observation.py tests/test_model_node_map_sampling.py
conda run -n ariadne pytest tests/test_action_features.py
```

## 7. 风险与约束

- 新增 action feature 会改变模型结构，旧 checkpoint 默认不兼容。
- 若启用 `PADDING_NODE_INDEX = -1`，需要同步修改所有 gather 前的 edge index 使用方式。
- 不应复制目标分支整个 `node_features.py`，否则会引入当前分支没有明确需求的 trajectory memory、branch features 和 shortest path tree 依赖。
- 不应删除或弱化当前 CNN map-inputs 分支。
- 不应把 `edge_dist_norm_j` 同时作为 pointer logit bias 使用；这是下一阶段需求。
- `ACTION_FEATURE_DIM` 必须和 `action_inputs.shape[-1]` 一致，否则训练 worker 和模型会在 batch forward 时失败。
- 如果 `USE_ACTION_FEATURES=False`，应允许 `ACTION_FEATURE_DIM=0` 且模型回退到当前行为。

## 8. Definition of Done

后续代码实现完成后，应满足：

- 当前训练路径和测试路径都能构造 `action_inputs`。
- `action_inputs[..., 0]` 表示 `edge_dist_norm_j`。
- padding slot 的 action feature 为 0。
- stay/current slot 仍由 mask 屏蔽。
- PolicyNet 和 QNet 都可以在保留 `map_inputs` 的同时消费 `action_inputs`。
- 没有迁移 pointer-level distance-aware attention。
- replay buffer current/next action input 字段完整保存。
- 相关单测在 `conda run -n ariadne pytest ...` 下通过。

## 9. 需要确认的问题

以下问题需要你确认。每个问题后面的建议和分析只供参考，不代表已确认需求。

### Q1. `edge_dist_norm_j` 是否同时输入 PolicyNet 和 QNet？

可选：

```text
A. 同时输入 PolicyNet 和 QNet
B. 只输入 QNet
C. 只输入 PolicyNet
```

建议：选 A。

分析：目标分支的动作级 feature 管线是 actor/critic 同时消费。这样 policy 在形成候选动作表示时看到距离，critic 在估值时也看到动作代价，表达更一致。它不同于 pointer-level logit bias，不会直接在 softmax 前惩罚远边。

回答：A。

### Q2. 是否默认启用该特征？

可选：

```text
A. 默认启用：USE_ACTION_FEATURES=True，USE_ACTION_FEATURE_EDGE_DIST=True
B. 只接入基础设施，默认关闭
```

建议：选 A。

分析：你的目标是迁移该分支中关于 `edge_dist_norm_j` 的内容，默认启用更接近目标分支行为。若担心 checkpoint 兼容或想先做无行为变化合入，可选 B。

回答：A。

### Q3. padding 是否从 0 切换为 `PADDING_NODE_INDEX = -1`？

可选：

```text
A. 切换为 -1，并在模型 gather 前 clamp(min=0)
B. 保持当前 0 padding
```

建议：选 A。

分析：当前 0 padding 会把真实节点 0 和 padding 混在一起。动作特征迁移依赖 edge slot 与 mask 严格对齐，切换为 -1 更稳，也与目标分支一致。但这会扩大修改面，需要同步更新 worker、test_worker、model 和测试。

回答：A。

### Q4. replay schema 新字段放哪里？

可选：

```text
A. append 到末尾：ACTION_INPUTS=17，NEXT_ACTION_INPUTS=18
B. 重排 replay_schema，使 action_inputs 紧跟 edge_inputs
```

建议：选 A。

分析：append 对当前 `MAP_INPUTS=6`、`NEXT_MAP_INPUTS=16` 的影响最小。B 更整齐，也更接近目标分支，但需要改动更多 index，风险更高。

回答：A。

### Q5. helper 文件命名和边界怎么定？

可选：

```text
A. 新增 action_features.py，只实现 edge_dist_norm_j
B. 新增 node_features.py，但只放最小动作特征逻辑
C. 直接把逻辑写在 worker.py / test_worker.py
```

建议：选 A。

分析：A 最能表达本次只迁移动作级特征，也方便测试。B 可能让后续实现误以为要迁移目标分支整套 node/action feature 系统。C 改动少但会导致训练/测试重复逻辑。

回答：A。

### Q6. 是否沿用 `DISTANCE_ATTENTION_MAX_NORM = 2.0` 做 clipping？

可选：

```text
A. 沿用 2.0 clipping
B. 不 clip
C. 改名为 EDGE_DIST_MAX_NORM 再 clip
```

建议：选 A 或 C。

分析：目标分支对 `edge_dist_norm_j` 做 clip，能避免异常边长造成过大输入。若担心名字含有 `ATTENTION` 会和本次非 pointer-bias 范围混淆，建议选 C。

回答：C。

### Q7. observation/model forward 参数顺序怎么处理？

可选：

```text
A. 保持当前 positional 顺序，action_inputs 用 keyword 传入
B. 改成目标分支顺序，把 action_inputs 放在 current_index 前
```

建议：选 A。

分析：当前代码有 CNN map-inputs 和 diagnostics，调用点已经围绕当前顺序展开。用 keyword 加入 `action_inputs` 能减少误传参数的风险。B 与目标分支一致，但在当前分支上更容易引入位置参数错误。

回答：A。

### Q8. 是否同步新增距离相关评估指标？

可选：

```text
A. 本次不新增指标，只迁移特征
B. 同步新增 average_step_distance / long_edge_action_ratio
```

建议：选 A。

分析：目标分支有 step distance 相关指标，但它不是 `edge_dist_norm_j` 特征本身。若本次目标是纯迁移动作级输入，建议先保持评估面不变；后续可以单独加指标验证行为变化。

回答：A。

### Q9. 新实验目录名 `FOLDER_NAME` 用什么？

可选：

```text
A. cnn_edge_dist_action
B. cnn_map_edge_dist
C. 沿用当前 FOLDER_NAME
```

建议：选 A 或 B。

分析：启用 action feature 后模型结构变了，不建议沿用当前已训练目录，避免误加载旧 checkpoint 或覆盖 baseline。A 更明确强调是 action-level edge distance；B 更强调仍是 CNN map-inputs 主线。

回答：cnn_edge_dist

