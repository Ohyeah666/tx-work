REPLAY_SIZE = 10000
MINIMUM_BUFFER_SIZE = 2000
BATCH_SIZE = 128
INPUT_DIM = 4
EMBEDDING_DIM = 128
# 是否使用地图输入
USE_MAP_INPUTS = False
MAP_INPUT_CHANNELS = 5
MAP_FEATURE_DIM = 64
FRONTIER_HEATMAP_SIGMA = 3.0
MAP_GATE_BIAS_INIT = -2.0       # -2.0 时初始 gate 约 sigmoid(-2)=0.119。-2.0/-1.5/-1.0
DIAGNOSTIC_IMG_GAP = 200
# 关于 edge_dist 的参数
USE_ACTION_FEATURES = True
USE_ACTION_FEATURE_EDGE_DIST = True  # edge_dist_norm_j
ACTION_FEATURE_DIM = int(USE_ACTION_FEATURES) * int(USE_ACTION_FEATURE_EDGE_DIST)
GRAPH_DISTANCE_NORMALIZER = 640
EDGE_DIST_MAX_NORM = 2.0
PADDING_NODE_INDEX = -1

NODE_PADDING_SIZE = 360  # the number of nodes will be padded to this value
K_SIZE = 20  # the number of neighboring nodes

USE_GPU = False  # do you want to collect training data using GPUs
USE_GPU_GLOBAL = True  # do you want to train the network using GPUs
NUM_GPU = 1
NUM_META_AGENT = 32
LR = 1e-5
MAP_LR = 1e-5           # map_inputs 分支的学习率
GAMMA = 1
DECAY_STEP = 256  # not use
SUMMARY_WINDOW = 32
FOLDER_NAME = 'only_edge_dist'
model_path = f'model/{FOLDER_NAME}'
train_path = f'train/{FOLDER_NAME}'
gifs_path = f'gifs/{FOLDER_NAME}'
LOAD_MODEL = False  # map-enhanced model is trained from scratch by default
SAVE_IMG_GAP = 100
ARCHIVE_CHECKPOINT_START_EPISODE = 14400
ARCHIVE_CHECKPOINT_GAP = 400
# Total 17600 episodes; archive every 800 from 12000, then every 320 after 16000.
ARCHIVE_CHECKPOINT_FREQUENT_START_EPISODE = 16000
ARCHIVE_CHECKPOINT_FREQUENT_GAP = 160
TOTAL_TRAIN_EPISODE = 17600
USE_WANDB = True  # do you want to log training metrics to Weights & Biases
WANDB_PROJECT = 'ARiADNE'
WANDB_ENTITY = None
WANDB_MODE = 'online'  # use 'offline' when training without network access
