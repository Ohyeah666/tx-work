REPLAY_SIZE = 10000
MINIMUM_BUFFER_SIZE = 2000
BATCH_SIZE = 128
INPUT_DIM = 9
# Basin 动作特征维度；Basin-5 对应 5 个 action-level 特征。
ACTION_FEATURE_DIM = 5
# 是否让 QNet 也使用 basin 特征；False 为 policy-only 消融，True 为 Policy+QNet。
USE_BASIN_IN_Q = True
EMBEDDING_DIM = 128
NODE_PADDING_SIZE = 360  # the number of nodes will be padded to this value
K_SIZE = 20  # the number of neighboring nodes
# basin sum 类特征的固定归一化尺度，通常不需要改。
BASIN_UTILITY_SUM_NORMALIZER = 50 * NODE_PADDING_SIZE
BASIN_EXPECTED_UNKNOWN_GAIN_SUM_NORMALIZER = NODE_PADDING_SIZE
# expected_unknown_gain 计算模式；local 使用缓存并只重算机器人附近节点，full 每步全量重算。
EXPECTED_UNKNOWN_GAIN_UPDATE_MODE = 'local'
# local 模式的重算半径 = 该系数 * sensor_range；2.0 覆盖本步 belief 变化可能影响的节点。
EXPECTED_UNKNOWN_GAIN_LOCAL_RADIUS_FACTOR = 2.0
# 计时 profiler 默认关闭；调性能时改 True，不改变训练数据和模型结构。
ENABLE_TIMING_PROFILER = False
# profiler 每多少次 graph 更新打印一次，0 表示不打印。
TIMING_PROFILER_PRINT_EVERY = 0

USE_GPU = False  # do you want to collect training data using GPUs
USE_GPU_GLOBAL = True  # do you want to train the network using GPUs
NUM_GPU = 1
NUM_META_AGENT = 32
LR = 1e-5
GAMMA = 1
DECAY_STEP = 256  # not use
SUMMARY_WINDOW = 32
FOLDER_NAME = 'ae_clean_basin5_node_9'
model_path = f'model/{FOLDER_NAME}'
train_path = f'train/{FOLDER_NAME}'
gifs_path = f'gifs/{FOLDER_NAME}'
LOAD_MODEL = False  # do you want to load the model trained before
SAVE_IMG_GAP = 100
ARCHIVE_CHECKPOINT_START_EPISODE = 11200
ARCHIVE_CHECKPOINT_GAP = 640
ARCHIVE_CHECKPOINT_FREQUENT_START_EPISODE = 16000
ARCHIVE_CHECKPOINT_FREQUENT_GAP = 64
USE_WANDB = True  # do you want to log training metrics to Weights & Biases
WANDB_PROJECT = 'ARiADNE'
WANDB_ENTITY = None
WANDB_MODE = 'online'  # use 'offline' when training without network access
