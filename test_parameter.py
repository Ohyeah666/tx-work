INPUT_DIM = 4
EMBEDDING_DIM = 128
MAP_INPUT_CHANNELS = 2  # 地图输入只保留 unknown + frontier heatmap 两个探索引导通道
MAP_FEATURE_DIM = 64
FRONTIER_HEATMAP_SIGMA = 3.0
MAP_GATE_BIAS_INIT = -2.0
DIAGNOSTIC_IMG_GAP = 200
K_SIZE = 20  # the number of neighbors

USE_GPU = False  # do you want to use GPUS?
NUM_GPU = 0  # the number of GPUs
NUM_META_AGENT = 16  # the number of processes
FOLDER_NAME = 'cnn_map_inputs_2ch_unknown_frontier'
model_path = f'model/{FOLDER_NAME}/run_20260726_162733__elapsed_25h58m00s'


TEST_SET_NAME = 'test'  # choose from 'complex', 'medium', 'easy', and 'test'
gifs_path = f'gifs/test'
trajectory_path = f'results/trajectory'
length_path = f'results/length'

NUM_TEST = 100
NUM_RUN = 1
SAVE_GIFS = False  # do you want to save GIFs
SAVE_TRAJECTORY = False  # do you want to save per-step metrics
SAVE_LENGTH = False  # do you want to save per-episode metrics
