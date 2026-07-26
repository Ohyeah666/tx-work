REPLAY_SIZE = 10000
MINIMUM_BUFFER_SIZE = 2000
BATCH_SIZE = 128
INPUT_DIM = 4
EMBEDDING_DIM = 128
MAP_INPUT_CHANNELS = 2  # 地图输入只保留 unknown + frontier heatmap 两个探索引导通道
MAP_FEATURE_DIM = 64
FRONTIER_HEATMAP_SIGMA = 3.0
MAP_GATE_BIAS_INIT = -2.0       # -2.0 时初始 gate 约 sigmoid(-2)=0.119。-2.0/-1.5/-1.0
DIAGNOSTIC_IMG_GAP = 200

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
FOLDER_NAME = 'cnn_map_inputs_2ch_unknown_frontier'
model_path = f'model/{FOLDER_NAME}'
train_path = f'train/{FOLDER_NAME}'
gifs_path = f'gifs/{FOLDER_NAME}'
LOAD_MODEL = False  # map-enhanced model is trained from scratch by default
SAVE_IMG_GAP = 100
ARCHIVE_CHECKPOINT_START_EPISODE = 12000
ARCHIVE_CHECKPOINT_GAP = 800
# Total 17600 episodes; archive every 800 from 12000, then every 320 after 16000.
ARCHIVE_CHECKPOINT_FREQUENT_START_EPISODE = 16000
ARCHIVE_CHECKPOINT_FREQUENT_GAP = 160
TOTAL_TRAIN_EPISODE = 17600
USE_WANDB = True  # do you want to log training metrics to Weights & Biases
WANDB_PROJECT = 'ARiADNE'
WANDB_ENTITY = None
WANDB_MODE = 'online'  # use 'offline' when training without network access
