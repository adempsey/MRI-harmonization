import os

### ____Data Parameters____

# Training input image locations
TRAINING_DATA_PATH          = os.path.join('data','train')

# Testing input image locations
TESTING_DATA_PATH           = os.path.join('data','test')

# Training label image locations
TARGET_DATA_PATH            = os.path.join('data','label')

# Directory containing transforms from images to atlas space
TRANSFORMATION_DATA_PATH    = os.path.join('data','transforms')

# Directory containing subject atlases
ATLAS_PATH                  = os.path.join('data','atlases')

# Location of the weights to use during a testing run
WEIGHT_PATH                 = os.path.join('model',
                                           'pretrained_30000',
                                           'model.npz')

# Directory to save parameter snapshots during testing
SAVE_PATH                   = os.path.join('model')

# Directory to output images during testing
OUTPUT_PATH                 = os.path.join('data','output')

### ____Environment Parameters____

# How many actions the agent takes in an episode
EPISODE_LEN = 10

# Reward Discount Factor
GAMMA = 0.95

# Number of actions available to the agent
# Action behavior is defined in state.py
N_ACTIONS = 9

# Number of actions that move the pixel values. e.g., when MOVE_RANGE=3, there are three actions: pixel_value+=1, +=0, -=1.
MOVE_RANGE = 9

### ____Training Parameters____
LEARNING_RATE    = 0.001
TRAIN_BATCH_SIZE = 16

# Width, height, and depth of training images
CROP_SIZE = 15

# Maximum image intensity
MAX_INTENSITY = (2**15)-1

# Number of episodes to train for
N_EPISODES = 30000

# Save a copy of the network parameters after this many episodes
SNAPSHOT_EPISODES  = 100

### ____Hardware Parameters____
GPU_ID = 0
