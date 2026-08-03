INPUT_DIM = 4
EMBEDDING_DIM = 128
USE_MAP_INPUTS = True
MAP_INPUT_CHANNELS = 5
MAP_FEATURE_DIM = 64
FRONTIER_HEATMAP_SIGMA = 3.0
MAP_GATE_BIAS_INIT = -2.0
DIAGNOSTIC_IMG_GAP = 200
USE_ACTION_FEATURES = False
USE_ACTION_FEATURE_EDGE_DIST = False
ACTION_FEATURE_DIM = int(USE_ACTION_FEATURES) * int(USE_ACTION_FEATURE_EDGE_DIST)
GRAPH_DISTANCE_NORMALIZER = 640
EDGE_DIST_MAX_NORM = 2.0
PADDING_NODE_INDEX = -1
K_SIZE = 20  # the number of neighbors

USE_GPU = False  # do you want to use GPUS?
NUM_GPU = 0  # the number of GPUs
NUM_META_AGENT = 16  # the number of processes
FOLDER_NAME = 'only_cnn_stride4_dilation'
model_path = f'model/{FOLDER_NAME}/run_20260802_120244__elapsed_25h16m33s'


TEST_SET_NAME = 'test'  # choose from 'complex', 'medium', 'easy', and 'test'
gifs_path = f'gifs/test'
trajectory_path = f'results/trajectory'
length_path = f'results/length'

NUM_TEST = 100
NUM_RUN = 1
SAVE_GIFS = False  # do you want to save GIFs
SAVE_TRAJECTORY = False  # do you want to save per-step metrics
SAVE_LENGTH = False  # do you want to save per-episode metrics
