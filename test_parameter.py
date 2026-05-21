INPUT_DIM = 7
EMBEDDING_DIM = 128
K_SIZE = 20  # the number of neighbors

USE_GPU = False  # do you want to use GPUS?
NUM_GPU = 0  # the number of GPUs
NUM_META_AGENT = 16  # the number of processes
FOLDER_NAME = 'ae_clean'
model_path = f'model/{FOLDER_NAME}/run_20260520_151118__elapsed_24h05m29s'

TEST_SET_NAME = 'medium'  # choose from 'complex', 'medium', 'easy', and 'test'
gifs_path = f'gifs/test'
trajectory_path = f'results/trajectory'
length_path = f'results/length'

NUM_TEST = 100
NUM_RUN = 1
SAVE_GIFS = True  # do you want to save GIFs
SAVE_TRAJECTORY = False  # do you want to save per-step metrics
SAVE_LENGTH = False  # do you want to save per-episode metrics
