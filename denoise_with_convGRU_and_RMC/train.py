from mini_batch_loader import *
from chainer import serializers
from MyFCN import *
from chainer import cuda, optimizers, Variable
import sys
import math
import time
import chainerrl
import State
import os
from pixelwise_a3c import *

#_/_/_/ paths _/_/_/
TRAINING_DATA_PATH          = os.path.join('..','adni3','train2')#"../training_BSD68.txt"
TESTING_DATA_PATH           = os.path.join('..','adni3','test2')
# TESTING_DATA_PATH           = "../testing.txt"
IMAGE_DIR_PATH              = "../"
SAVE_PATH            = "./model/denoise_myfcn_3d_"

#_/_/_/ training parameters _/_/_/
LEARNING_RATE    = 0.001
TRAIN_BATCH_SIZE = 16#32#64
TEST_BATCH_SIZE  = 1 #must be 1
N_EPISODES           = 30000
EPISODE_LEN = 10#30#5
SNAPSHOT_EPISODES  = 100
TEST_EPISODES = 100
GAMMA = 0.95 # discount factor
MAX_INTENSITY = (2**15)-1

#noise setting
MEAN = 0
SIGMA = 80
N_ACTIONS = 9
MOVE_RANGE = 9 #number of actions that move the pixel values. e.g., when MOVE_RANGE=3, there are three actions: pixel_value+=1, +=0, -=1.
CROP_SIZE = 15#70

GPU_ID = 0

# def test(loader, agent, fout):
#     return
#     sum_psnr     = 0
#     sum_reward = 0
#     test_data_size = 1#MiniBatchLoader.count_paths(TESTING_DATA_PATH)
#     current_state = State.State((TEST_BATCH_SIZE,1,CROP_SIZE,CROP_SIZE), MOVE_RANGE)
#     for i in range(0, test_data_size, TEST_BATCH_SIZE):
#         raw_x, raw_y = loader.load_testing_data(np.array(range(i, i+TEST_BATCH_SIZE)))
#         raw_n = np.random.normal(MEAN,SIGMA,raw_x.shape).astype(raw_x.dtype)/MAX_INTENSITY
#         current_state.reset(raw_x,raw_n)
#         # reward = np.zeros(raw_x.shape, raw_x.dtype)*MAX_INTENSITY
#         for t in range(0, EPISODE_LEN):
#             print(t, EPISODE_LEN)
#             previous_image = current_state.image.copy()
#             action, inner_state = agent.act(current_state.tensor)
#             current_state.step(action, inner_state)
#             # reward = np.square(raw_y - previous_image)*MAX_INTENSITY - np.square(raw_y - current_state.image)*MAX_INTENSITY
#             # sum_reward += np.mean(reward)*np.power(GAMMA,t)
#
#         agent.stop_episode()
#
#         I = np.maximum(0,raw_x)
#         I = np.minimum(1,I)
#         p = np.maximum(0,current_state.image)
#         p = np.minimum(1,p)
#         I = (I*MAX_INTENSITY+0.5).astype(np.uint32)
#         p = (p*MAX_INTENSITY+0.5).astype(np.uint32)
#         # sum_psnr += cv2.PSNR(p, I)
#
#     # print("test total reward {a}, PSNR {b}".format(a=sum_reward*MAX_INTENSITY/test_data_size, b=sum_psnr/test_data_size))
#     # fout.write("test total reward {a}, PSNR {b}\n".format(a=sum_reward*MAX_INTENSITY/test_data_size, b=sum_psnr/test_data_size))
#     # sys.stdout.flush()


def main(fout):
    #_/_/_/ load dataset _/_/_/
    mini_batch_loader = MiniBatchLoader(
        TRAINING_DATA_PATH,
        TESTING_DATA_PATH,
        IMAGE_DIR_PATH,
        CROP_SIZE)

    chainer.cuda.get_device_from_id(GPU_ID).use()

    current_state = State.State((TRAIN_BATCH_SIZE,1,CROP_SIZE,CROP_SIZE), MOVE_RANGE)

    # load myfcn model
    model = MyFcn(N_ACTIONS)

    #_/_/_/ setup _/_/_/
    optimizer = chainer.optimizers.Adam(alpha=LEARNING_RATE)
    optimizer.setup(model)

    agent = PixelWiseA3C_InnerState_ConvR(model, optimizer, EPISODE_LEN, GAMMA)
    agent.model.to_gpu()

    #_/_/_/ training _/_/_/

    train_data_size = MiniBatchLoader.count_paths(TRAINING_DATA_PATH)
    indices = np.random.permutation(train_data_size)
    i = 0
    rewardtrack = []
    for episode in range(1, N_EPISODES+1):
        # display current state
        print("episode %d" % episode)
        fout.write("episode %d\n" % episode)
        sys.stdout.flush()
        # load images
        r = indices[i:i+TRAIN_BATCH_SIZE]
        raw_x, raw_y = mini_batch_loader.load_training_data(r)
        # generate noise
        raw_n = np.random.normal(MEAN,SIGMA,raw_x.shape).astype(raw_x.dtype)/MAX_INTENSITY
        # initialize the current state and reward
        current_state.reset(raw_x,raw_n)
        reward = np.zeros(raw_x.shape, raw_x.dtype)
        sum_reward = 0

        for t in range(0, EPISODE_LEN):
            previous_image = current_state.image.copy()
            action, inner_state = agent.act_and_train(current_state.tensor, reward)
            current_state.step(action, inner_state)
            reward = np.square(raw_y - previous_image)*MAX_INTENSITY - np.square(raw_y - current_state.image)*MAX_INTENSITY
            # print(reward.min(),reward.max())
            # reward = np.square(raw_x - previous_image)*255 - np.square(raw_x - current_state.image)*255
            # print("raw_x",raw_x.min(),raw_x.max(),np.mean(raw_x))
            # print("raw_y",raw_y.min(),raw_y.max(),np.mean(raw_y))
            # print("*")
            sum_reward += np.mean(reward)*np.power(GAMMA,t)
        # print(current_state.tensor.shape)
        # print(current_state.image.shape)
        # print(raw_x.shape)
        # print(raw_y.shape)

        # I = np.maximum(0,raw_x[0,:,:,:,:])
        # I = np.minimum(1,I)
        # N = np.maximum(0,raw_y[0,:,:,:,:])
        # N = np.minimum(1,N)
        # p = np.maximum(0,current_state.image[0,:,:,:,:])
        # p = np.minimum(1,p)
        # # print("p[0].max()",p[0].max()*)
        # I = (I[0]*MAX_INTENSITY+0.5).astype(np.uint32)
        # N = (N[0]*MAX_INTENSITY+0.5).astype(np.uint32)
        # p = (p[0]*MAX_INTENSITY+0.5).astype(np.uint32)
        # p = np.transpose(p,(1,2,3,0))
        # I = np.transpose(I,(1,2,3,0))
        # N = np.transpose(N,(1,2,3,0))
        # print(I.max(),I.min(),I.shape,MAX_INTENSITY)
        # print(p.max(),p.min(),p.shape,MAX_INTENSITY)
        raw_x *= (2**15)-1
        raw_y *= (2**15)-1
        output = current_state.image * (2**15)-1
        # print(np.max(raw_x),np.max(raw_y), np.max(output))
        # if episode % 100 == 0:
        #     nrrd.write('./trainoutput/%d_input.nrrd'%episode,raw_x)
        #     nrrd.write('./trainoutput/%d_target.nrrd'%episode,raw_y)
        #     nrrd.write('./trainoutput/%d_output.nrrd'%episode,output)
        # cv2.imwrite('./resultimage/'+str(i)+'_input.png',I)
        # cv2.imwrite('./resultimage/'+str(i)+'_output.png',p)
        # cv2.imwrite('./resultimage/'+str(i)+'_label.png',N)




        agent.stop_episode_and_train(current_state.tensor, reward, True)
        print("train total reward {a}".format(a=sum_reward*MAX_INTENSITY))
        fout.write("train total reward {a}\n".format(a=sum_reward))
        sys.stdout.flush()
        rewardtrack.append(sum_reward*MAX_INTENSITY)

        if episode % 50 == 0:
            s = np.array(rewardtrack)
            print("avg: %f, std_dev: %f" % (np.mean(s), np.var(s)))
            rewardtrack = []

        # if episode % TEST_EPISODES == 0:
        #     #_/_/_/ testing _/_/_/
        #     test(mini_batch_loader, agent, fout)

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



if __name__ == '__main__':
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
