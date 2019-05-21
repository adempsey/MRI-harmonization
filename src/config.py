import os

TRAINING_DATA_PATH          = os.path.join('data','train')
TESTING_DATA_PATH           = os.path.join('data','test')
TARGET_DATA_PATH            = os.path.join('data','label')
TRANSFORMATION_DATA_PATH    = os.path.join('data','transforms')
ATLAS_PATH                  = os.path.join('data','atlases')
SAVE_PATH                   = os.path.join('model',
                                           'pretrained_30000',
                                           'model.npz')
OUTPUT_PATH                 = os.path.join('data','output')

#_/_/_/ training parameters _/_/_/
LEARNING_RATE    = 0.001
TRAIN_BATCH_SIZE = 16
TEST_BATCH_SIZE  = 1 #must be 1
N_EPISODES           = 30000
EPISODE_LEN = 10
SNAPSHOT_EPISODES  = 100
TEST_EPISODES = 100
GAMMA = 0.95 # discount factor
MAX_INTENSITY = (2**15)-1

#noise setting
MEAN = 0
SIGMA = 80
N_ACTIONS = 9
MOVE_RANGE = 9 #number of actions that move the pixel values. e.g., when MOVE_RANGE=3, there are three actions: pixel_value+=1, +=0, -=1.
CROP_SIZE = 15

GPU_ID = 0
