INPUT_DIM = 9
ACTION_FEATURE_DIM = 5
USE_BASIN_IN_Q = False
EMBEDDING_DIM = 128
K_SIZE = 20  # the number of neighbors
BASIN_UTILITY_SUM_NORMALIZER = 50 * 360
BASIN_EXPECTED_UNKNOWN_GAIN_SUM_NORMALIZER = 360
EXPECTED_UNKNOWN_GAIN_UPDATE_MODE = 'local'
EXPECTED_UNKNOWN_GAIN_LOCAL_RADIUS_FACTOR = 2.0
# 256 大概两分钟；128 更快，一分半大概，但是精度差很多；
# 越大时间越长，精度越高；0 表示全量计算，理论上最精确，但非常慢。
EXPECTED_UNKNOWN_GAIN_RAY_SAMPLE_COUNT = 256
ENABLE_TIMING_PROFILER = False
TIMING_PROFILER_PRINT_EVERY = 0

USE_GPU = False  # do you want to use GPUS?
NUM_GPU = 0  # the number of GPUs
NUM_META_AGENT = 16  # the number of processes
FOLDER_NAME = 'ae_clean_fast_basin5_node9'
model_path = f'model/{FOLDER_NAME}/run_20260531_111403__elapsed_25h29m12s'


TEST_SET_NAME = 'test'  # choose from 'complex', 'medium', 'easy', and 'test'
gifs_path = f'gifs/test'
trajectory_path = f'results/trajectory'
length_path = f'results/length'

NUM_TEST = 100
NUM_RUN = 1
SAVE_GIFS = False  # do you want to save GIFs
SAVE_TRAJECTORY = False  # do you want to save per-step metrics
SAVE_LENGTH = False  # do you want to save per-episode metrics
