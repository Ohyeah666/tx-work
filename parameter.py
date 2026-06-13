REPLAY_SIZE = 10000
MINIMUM_BUFFER_SIZE = 2000
BATCH_SIZE = 128

# Node input features are [x, y, utility, guidepost] plus the optional features below.
BASE_NODE_FEATURE_DIM = 4
# 调整下面三个参数以启用或禁用相应的节点特征，INPUT_DIM 会自动计算
USE_NODE_FEATURE_GRAPH_DIST_TO_CURRENT = False
USE_NODE_FEATURE_UTILITY_OVER_DIST = True
USE_NODE_FEATURE_VISIT_COUNT = False
INPUT_DIM = (
    BASE_NODE_FEATURE_DIM
    + int(USE_NODE_FEATURE_GRAPH_DIST_TO_CURRENT)
    + int(USE_NODE_FEATURE_UTILITY_OVER_DIST)
    + int(USE_NODE_FEATURE_VISIT_COUNT)
)
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
FOLDER_NAME = 'only_utility_over_dist'  # the name of the folder to save models and logs, should be set according to the active node feature configuration
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
