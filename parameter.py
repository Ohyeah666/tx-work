REPLAY_SIZE = 10000
MINIMUM_BUFFER_SIZE = 2000
BATCH_SIZE = 128

# Node input features are [x, y, utility, guidepost] plus the optional features below.
BASE_NODE_FEATURE_DIM = 4
# 调整下面四个参数以启用或禁用相应的节点特征，INPUT_DIM 会自动计算
USE_NODE_FEATURE_GRAPH_DIST_TO_CURRENT = False
USE_NODE_FEATURE_UTILITY_OVER_DIST = False
USE_NODE_FEATURE_VISIT_COUNT = False
USE_NODE_FEATURE_TRAJECTORY_MEMORY = True  # 是否把轨迹记忆 memory_i 加入节点输入
INPUT_DIM = (
    BASE_NODE_FEATURE_DIM
    + int(USE_NODE_FEATURE_GRAPH_DIST_TO_CURRENT)
    + int(USE_NODE_FEATURE_UTILITY_OVER_DIST)
    + int(USE_NODE_FEATURE_VISIT_COUNT)
    + int(USE_NODE_FEATURE_TRAJECTORY_MEMORY)
)
USE_ACTION_FEATURES = False  # 是否启用动作级 DTM 特征
# 调整下面六个参数以启用或禁用相应的动作特征，ACTION_FEATURE_DIM 会自动计算
USE_ACTION_FEATURE_EDGE_DIST = False  # edge_dist_norm_j
USE_ACTION_FEATURE_IMMEDIATE_REVERSE = False  # is_immediate_reverse_j
USE_ACTION_FEATURE_NEXT_NODE_MEMORY = False  # next_node_memory_j
# 下面三个是 branch 相关的特征
USE_ACTION_FEATURE_BRANCH_UTILITY = False  # branch_utility_norm_j
USE_ACTION_FEATURE_BRANCH_GAIN = False  # branch_gain_norm_j
USE_ACTION_FEATURE_BRANCH_MEMORY = False  # branch_memory_j

USE_TRAJECTORY_MEMORY = True  # 是否计算 max-decay 轨迹记忆场
USE_DIRECTIONAL_BRANCH_FEATURES = False  # 是否允许计算 first-hop 方向分支动作特征
ACTION_FEATURE_DIM = int(USE_ACTION_FEATURES) * (
    int(USE_ACTION_FEATURE_EDGE_DIST)
    + int(USE_ACTION_FEATURE_IMMEDIATE_REVERSE)
    + int(USE_ACTION_FEATURE_NEXT_NODE_MEMORY)
    + int(USE_DIRECTIONAL_BRANCH_FEATURES and USE_ACTION_FEATURE_BRANCH_UTILITY)
    + int(USE_DIRECTIONAL_BRANCH_FEATURES and USE_ACTION_FEATURE_BRANCH_GAIN)
    + int(USE_DIRECTIONAL_BRANCH_FEATURES and USE_ACTION_FEATURE_BRANCH_MEMORY)
)
TRAJECTORY_MEMORY_GAMMA = 0.95  # 轨迹记忆时间衰减系数
TRAJECTORY_MEMORY_SIGMA = 80  # 轨迹记忆图距离扩散尺度
TRAJECTORY_MEMORY_WINDOW = 64  # 轨迹记忆使用的最近步数
BRANCH_GAIN_EPS = 1e-6  # 分支收益除法防零项
GRAPH_DISTANCE_NORMALIZER = 640  # 图距离和坐标归一化尺度
PADDING_NODE_INDEX = -1  # edge padding 使用的无效节点编号
BACKTRACK_MEMORY_THRESHOLD = 0.5  # 判定进入历史区域的 memory 阈值
BACKTRACK_FUTURE_WINDOW = 6  # 回头有效性评估的未来步数
BACKTRACK_GAIN_THRESHOLD = 1.0  # 判定回头后有有效探索收益的阈值
LOCAL_OSCILLATION_WINDOWS = (4, 6)  # 局部震荡统计窗口

EMBEDDING_DIM = 128
NODE_PADDING_SIZE = 360  # the number of nodes will be padded to this value
K_SIZE = 20  # the number of neighboring nodes

USE_GPU = False  # do you want to collect training data using GPUs
USE_GPU_GLOBAL = True  # do you want to train the network using GPUs
NUM_GPU = 1
NUM_META_AGENT = 32
LR = 1e-5
GAMMA = 1
DECAY_STEP = 256  # not use
SUMMARY_WINDOW = 32
FOLDER_NAME = 'node_memory_only2'  # the name of the folder to save models and logs, should be set according to the active node feature configuration
model_path = f'model/{FOLDER_NAME}'
train_path = f'train/{FOLDER_NAME}'
gifs_path = f'gifs/{FOLDER_NAME}'
LOAD_MODEL = False  # do you want to load the model trained before
SAVE_IMG_GAP = 100
ARCHIVE_CHECKPOINT_START_EPISODE = 12000
ARCHIVE_CHECKPOINT_GAP = 800
ARCHIVE_CHECKPOINT_FREQUENT_START_EPISODE = 16000
ARCHIVE_CHECKPOINT_FREQUENT_GAP = 160
# 总共训练17600轮，从12000episode开始，每800轮存档一次，16000轮后每320轮存档一次，最终在17600轮结束时存档一次
TOTAL_TRAIN_EPISODE = 17600
USE_WANDB = True  # do you want to log training metrics to Weights & Biases
WANDB_PROJECT = 'ARiADNE'
WANDB_ENTITY = None
WANDB_MODE = 'online'  # use 'offline' when training without network access
