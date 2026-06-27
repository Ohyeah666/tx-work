from parameter import (
    EMBEDDING_DIM,
    FOLDER_NAME,
    INPUT_DIM,
    ACTION_FEATURE_DIM,
    PADDING_NODE_INDEX,
    USE_ACTION_FEATURES,
    USE_ACTION_FEATURE_BRANCH_GAIN,
    USE_ACTION_FEATURE_BRANCH_MEMORY,
    USE_ACTION_FEATURE_BRANCH_UTILITY,
    USE_ACTION_FEATURE_EDGE_DIST,
    USE_ACTION_FEATURE_IMMEDIATE_REVERSE,
    USE_ACTION_FEATURE_NEXT_NODE_MEMORY,
    USE_DIRECTIONAL_BRANCH_FEATURES,
    USE_NODE_FEATURE_GRAPH_DIST_TO_CURRENT,
    USE_NODE_FEATURE_UTILITY_OVER_DIST,
    USE_NODE_FEATURE_VISIT_COUNT,
    USE_NODE_FEATURE_TRAJECTORY_MEMORY,
)

K_SIZE = 20  # the number of neighbors

USE_GPU = False  # do you want to use GPUS?
NUM_GPU = 0  # the number of GPUs
NUM_META_AGENT = 16  # the number of processes
# Set this to a run trained with the active node/action feature configuration.
model_path = f'model/{FOLDER_NAME}/run_20260618_225029__elapsed_25h12m57s'

TEST_SET_NAME = 'test'  # choose from 'complex', 'medium', 'easy', and 'test'
gifs_path = f'gifs/test'
trajectory_path = f'results/trajectory'
length_path = f'results/length'

NUM_TEST = 100
NUM_RUN = 1
SAVE_GIFS = True  # do you want to save GIFs
SAVE_TRAJECTORY = False  # do you want to save per-step metrics
SAVE_LENGTH = False  # do you want to save per-episode metrics
