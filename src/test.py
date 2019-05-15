from mini_batch_loader import *
from chainer import serializers
from network import *
from chainer import cuda, optimizers, Variable
import sys
import math
import time
import chainerrl
import State
import os
from pixelwise_a3c import *
import nrrd

#_/_/_/ paths _/_/_/
TRAINING_DATA_PATH          = os.path.join('..','adni3','train2')
TESTING_DATA_PATH           = os.path.join('..','adni3','test_action_sample')
IMAGE_DIR_PATH              = "../"
SAVE_PATH            = "./model/denoise_myfcn_3d_"

#_/_/_/ training parameters _/_/_/
LEARNING_RATE    = 0.001
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE  = 1 #must be 1
N_EPISODES           = 1000#30000
EPISODE_LEN = 10#5
GAMMA = 0.95 # discount factor
MAX_INTENSITY = (2**15)-1#32767

#noise setting
MEAN = 0
SIGMA = 80

N_ACTIONS = 9#9
MOVE_RANGE = 9 #number of actions that move the pixel values. e.g., when MOVE_RANGE=3, there are three actions: pixel_value+=1, +=0, -=1.
CROP_SIZE = 15#70

GPU_ID = 0

def test(loader, agent, fout):
    sum_psnr     = 0
    sum_reward = 0
    test_data_size = MiniBatchLoader.count_paths(TESTING_DATA_PATH)
    current_state = State.State((TEST_BATCH_SIZE,1,CROP_SIZE,CROP_SIZE), MOVE_RANGE)
    for i in range(0, 1, TEST_BATCH_SIZE):
        raw_x, raw_y, ogMax = loader.load_testing_data(np.array(range(i, i+TEST_BATCH_SIZE)))
        raw_n = np.random.normal(MEAN,SIGMA,raw_x.shape).astype(raw_x.dtype)/MAX_INTENSITY
        current_state.reset(raw_x,raw_n)
        reward = np.zeros(raw_x.shape, raw_x.dtype)*MAX_INTENSITY


        for t in range(0, EPISODE_LEN):
            previous_image = current_state.image.copy()
            action, inner_state = agent.act(current_state.tensor)
            current_state.step(action, inner_state)

        agent.stop_episode()

        I = np.maximum(0,raw_x)
        I = np.minimum(1,I)
        p = np.maximum(0,current_state.image)
        p = np.minimum(1,p)
        p = p.squeeze()
        I = I.squeeze()
        I = (I/I.max())*ogMax
        p = (p/p.max())*ogMax
        nrrd.write('./resultimage/input_%d.nrrd'%i,I)
        nrrd.write('./resultimage/output_%d.nrrd'%i,p)

    sys.stdout.flush()


def main(fout):
    #_/_/_/ load dataset _/_/_/
    mini_batch_loader = MiniBatchLoader(
        TRAINING_DATA_PATH,
        TESTING_DATA_PATH,
        IMAGE_DIR_PATH,
        CROP_SIZE)

    chainer.cuda.get_device_from_id(GPU_ID).use()

    current_state = State.State((TRAIN_BATCH_SIZE,1,CROP_SIZE,CROP_SIZE), MOVE_RANGE)

    # load network
    model = Network(N_ACTIONS)

    #_/_/_/ setup _/_/_/
    optimizer = chainer.optimizers.Adam(alpha=LEARNING_RATE)
    optimizer.setup(model)

    agent = PixelWiseA3C_InnerState_ConvR(model, optimizer, EPISODE_LEN, GAMMA)
    chainer.serializers.load_npz('./model/denoise_myfcn_3d_30000/model.npz', agent.model)
    agent.act_deterministically = True
    agent.model.to_gpu()

    #_/_/_/ testing _/_/_/
    test(mini_batch_loader, agent, fout)

if __name__ == '__main__':
    try:
        fout = open('testlog.txt', "w")
        start = time.time()
        main(fout)
        end = time.time()
        print("{s}[s]".format(s=end - start))
        print("{s}[m]".format(s=(end - start)/60))
        print("{s}[h]".format(s=(end - start)/60/60))
        fout.write("{s}[s]\n".format(s=end - start))
        fout.write("{s}[m]\n".format(s=(end - start)/60))
        fout.write("{s}[h]\n".format(s=(end - start)/60/60))
        fout.close()
    except Exception as error:
        print(error.message)
