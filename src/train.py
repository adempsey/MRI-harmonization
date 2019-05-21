from mini_batch_loader import *
from chainer import serializers
from network import *
from chainer import cuda, optimizers, Variable
import sys
import math
import time
import chainerrl
import state
import os
from pixelwise_a3c import *
from config import *

def main(fout):
    #_/_/_/ load dataset _/_/_/
    mini_batch_loader = MiniBatchLoader(
        TRAINING_DATA_PATH,
        TESTING_DATA_PATH,
        CROP_SIZE)

    chainer.cuda.get_device_from_id(GPU_ID).use()

    current_state = state.State((TRAIN_BATCH_SIZE,1,CROP_SIZE,CROP_SIZE), MOVE_RANGE)

    # load myfcn model
    model = Network(N_ACTIONS)

    #_/_/_/ setup _/_/_/
    optimizer = chainer.optimizers.Adam(alpha=LEARNING_RATE)
    optimizer.setup(model)

    agent = PixelWiseA3C_InnerState_ConvR(model, optimizer, EPISODE_LEN, GAMMA)
    agent.model.to_gpu()

    #_/_/_/ training _/_/_/

    train_data_size = MiniBatchLoader.count_paths(TRAINING_DATA_PATH)
    indices = np.random.permutation(train_data_size)
    i = 0
    for episode in range(1, N_EPISODES+1):
        # display current state
        print("episode %d" % episode)
        fout.write("episode %d\n" % episode)
        sys.stdout.flush()
        # load images
        r = indices[i:i+TRAIN_BATCH_SIZE]
        raw_x, raw_y = mini_batch_loader.load_training_data(r)
        # initialize the current state and reward
        current_state.reset(raw_x)
        reward = np.zeros(raw_x.shape, raw_x.dtype)
        sum_reward = 0

        for t in range(0, EPISODE_LEN):
            previous_image = current_state.image.copy()
            action, inner_state = agent.act_and_train(current_state.tensor, reward)
            current_state.step(action, inner_state)
            reward = np.square(raw_y - previous_image)*MAX_INTENSITY - np.square(raw_y - current_state.image)*MAX_INTENSITY

            sum_reward += np.mean(reward)*np.power(GAMMA,t)
        raw_x *= (2**15)-1
        raw_y *= (2**15)-1
        output = current_state.image * (2**15)-1

        agent.stop_episode_and_train(current_state.tensor, reward, True)
        print("train total reward {a}".format(a=sum_reward*MAX_INTENSITY))
        fout.write("train total reward {a}\n".format(a=sum_reward))
        sys.stdout.flush()

        if episode % SNAPSHOT_EPISODES == 0:
            agent.save(SAVE_PATH+str(episode))

        if i+TRAIN_BATCH_SIZE >= train_data_size:
            i = 0
            indices = np.random.permutation(train_data_size)
        else:
            i += TRAIN_BATCH_SIZE

        if i+2*TRAIN_BATCH_SIZE >= train_data_size:
            i = train_data_size - TRAIN_BATCH_SIZE

        optimizer.alpha = LEARNING_RATE*((1-episode/N_EPISODES)**0.9)

def trainModel():
    try:
        np.seterr(divide='raise', invalid='raise')
        fout = open('log.txt', "w")
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
