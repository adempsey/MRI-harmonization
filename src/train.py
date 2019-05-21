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

"""
- Trains the network according to parameters supplied in config
@param loader - instance of mini_batch_loader to load test data
@param agent - initialized agent to perform harmonization task
@param optimizer - initialized optimization object to use for training
"""
def train(loader, agent, optimizer):

    # Set up agent state space
    current_state = state.State((TRAIN_BATCH_SIZE,1,CROP_SIZE,CROP_SIZE),
                                 MOVE_RANGE)

    # Shuffle order of training data
    train_data_size = MiniBatchLoader.count_paths(TRAINING_DATA_PATH)
    indices = np.random.permutation(train_data_size)

    i = 0
    for episode in range(1, N_EPISODES+1):
        print("episode %d" % episode)
        sys.stdout.flush()

        # Load batch of images
        r = indices[i:i+TRAIN_BATCH_SIZE]
        raw_x, raw_y = loader.load_training_data(r)

        # Initialize current state and reward
        current_state.reset(raw_x)
        reward = np.zeros(raw_x.shape, raw_x.dtype)
        sum_reward = 0

        # Iterate through episode for each image in the batch
        for t in range(0, EPISODE_LEN):
            # Store copy of previous image for reference
            previous_image = current_state.image.copy()

            # Sample, execute action, and track resulting reward
            action, inner_state = agent.act_and_train(current_state.tensor,
                                                      reward)
            current_state.step(action, inner_state)

            # Calculate current reward
            reward = np.square(raw_y - previous_image)*MAX_INTENSITY - np.square(raw_y - current_state.image)*MAX_INTENSITY

            # Track cumulative sum of rewards
            sum_reward += np.mean(reward)*np.power(GAMMA,t)

        # Update weights
        agent.stop_episode_and_train(current_state.tensor, reward, True)
        print("train total reward {a}".format(a=sum_reward*MAX_INTENSITY))
        sys.stdout.flush()

        # Save current weights to disk
        if episode % SNAPSHOT_EPISODES == 0:
            agent.save(SAVE_PATH+str(episode))

        # Reshuffle training image order or get next batch
        if i+TRAIN_BATCH_SIZE >= train_data_size:
            i = 0
            indices = np.random.permutation(train_data_size)
        else:
            i += TRAIN_BATCH_SIZE

        # Account for batch size overflow
        if i+2*TRAIN_BATCH_SIZE >= train_data_size:
            i = train_data_size - TRAIN_BATCH_SIZE

        # Update learning rate
        optimizer.alpha = LEARNING_RATE*((1-episode/N_EPISODES)**0.9)

"""
- Set up agent, batch loader, optimizer, and GPU links, and begin training
"""
def main():

    # Initialize data loader
    loader = MiniBatchLoader(
        TRAINING_DATA_PATH,
        TESTING_DATA_PATH,
        CROP_SIZE
    )

    # Set up GPU
    chainer.cuda.get_device_from_id(GPU_ID).use()

    # Load network
    model = Network(N_ACTIONS)

    # Initialize optimizer
    optimizer = chainer.optimizers.Adam(alpha=LEARNING_RATE)
    optimizer.setup(model)

    # Initialize agent
    agent = PixelWiseA3C_InnerState_ConvR(model,
                                          optimizer,
                                          EPISODE_LEN,
                                          GAMMA)
    agent.model.to_gpu()

    # Start training
    train(loader, agent, optimizer)

"""
- Entry point for running a training session
"""
def trainModel():
    try:
        np.seterr(divide='raise', invalid='raise')
        fout = open('log.txt', "w")
        start = time.time()
        main()
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
