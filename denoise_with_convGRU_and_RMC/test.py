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
import nrrd

#_/_/_/ paths _/_/_/
TRAINING_DATA_PATH          = os.path.join('..','adni3','train2')
# TRAINING_DATA_PATH          = "../training_BSD68.txt"
TESTING_DATA_PATH           = os.path.join('..','adni3','test3')
# TESTING_DATA_PATH           = "../testing_1.txt"
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
    # for i in range(0, test_data_size, TEST_BATCH_SIZE):
    for i in range(0, 5, TEST_BATCH_SIZE):
        raw_x, raw_y, ogMax = loader.load_testing_data(np.array(range(i, i+TEST_BATCH_SIZE)))
        # print(raw_y.max(),raw_x.max())
        raw_n = np.random.normal(MEAN,SIGMA,raw_x.shape).astype(raw_x.dtype)/MAX_INTENSITY
        current_state.reset(raw_x,raw_n)
        reward = np.zeros(raw_x.shape, raw_x.dtype)*MAX_INTENSITY


        for t in range(0, EPISODE_LEN):
            previous_image = current_state.image.copy()
            action, inner_state = agent.act(current_state.tensor)
            # actionMap = np.stack((action,)*3, axis=-1).squeeze()

            # def actionToColor(a):
            #     if a[0] == 0: # Big decrement
            #         return np.array([255, 0, 0]) #FF0000
            #     if a[0] == 1: # Little decrement
            #         return np.array([255, 0, 255]) #FF00FF
            #     if a[0] == 2: # Nothing
            #         return np.array([0, 0, 0]) #000000
            #     if a[0] == 3: # Little increment
            #         return np.array([0, 255, 255]) #00FFFF
            #     if a[0] == 4: # Big increment
            #         return np.array([0, 0, 255]) #0000FF
            #     if a[0] == 5: # Sharpen
            #         return np.array([0, 255, 0]) #00FF00
            #     if a[0] == 6: # Blur
            #         return np.array([255, 40, 0]) #FFAA00
                # if a[0] == 7:
                #     return np.array([128, 0, 128])
            #
            # actionMap = np.apply_along_axis(actionToColor, 2, actionMap)
            # print(current_state.image.max())
            current_state.step(action, inner_state)


            # reward = np.square(raw_y - previous_image)*MAX_INTENSITY - np.square(raw_y - current_state.image)*MAX_INTENSITY
            # sum_reward += np.mean(reward)*np.power(GAMMA,t)

            # p = np.maximum(0,current_state.image)
            # p = np.minimum(1,p)
            # p = (p[0]*MAX_INTENSITY+0.5).astype(np.uint32)
            # p = np.transpose(p,(1,2,3,0))
            # nrrd.write('./resultimage/output_%s.nrrd' % str(t),p)
            # p = (p[0]*255+0.5).astype(np.uint8)
            # p = np.transpose(p,(1,2,0))
            # cv2.imwrite('./resultimage/'+str(i)+'_'+str(t)+'_output.png',p)
            # np.save('./resultimage/actions',action)
            # cv2.imwrite('./resultimage/'+str(i)+'_'+str(t)+'_action.png',actionMap)

        agent.stop_episode()

        I = np.maximum(0,raw_x)
        I = np.minimum(1,I)
        # N = np.maximum(0,raw_y)
        # N = np.minimum(1,N)
        #
        p = np.maximum(0,current_state.image)
        p = np.minimum(1,p)
        p = p.squeeze()
        I = I.squeeze()
        # I *= (2.**15)-1.
        I = (I/I.max())*ogMax
        p = (p/p.max())*ogMax
        # print(p.shape,I.shape)
        # print("p[0].max()",p[0].max()*)
        # I = (I[0]*MAX_INTENSITY+0.5).astype(np.uint32)
        # N = (N[0]*MAX_INTENSITY+0.5).astype(np.uint32)
        # p = (p[0]*MAX_INTENSITY+0.5).astype(np.uint32)
        # p = np.transpose(p,(1,2,0))
        # p = np.flip(p,1)
        # I = np.flip(I,1)
        # I = np.transpose(I,(1,2,0))
        # p = np.transpose(p,(1,2,3,0))
        # I = np.transpose(I,(1,2,3,0))
        # N = np.transpose(N,(1,2,3,0))
        # print(I.max(),I.min(),I.shape,MAX_INTENSITY)
        # print(p.max(),p.min(),p.shape,MAX_INTENSITY)
        nrrd.write('./resultimage/input_%d.nrrd'%i,I)
        nrrd.write('./resultimage/output_%d.nrrd'%i,p)
        # cv2.imwrite('./resultimage/'+str(i)+'_input.png',I)
        # cv2.imwrite('./resultimage/'+str(i)+'_output.png',p)
        # cv2.imwrite('./resultimage/'+str(i)+'_label.png',N)

        # sum_psnr += cv2.PSNR(p, I)

    # print("test total reward {a}, PSNR {b}".format(a=sum_reward*MAX_INTENSITY/test_data_size, b=sum_psnr/test_data_size))
    # fout.write("test total reward {a}, PSNR {b}\n".format(a=sum_reward*MAX_INTENSITY/test_data_size, b=sum_psnr/test_data_size))
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

    # load myfcn model
    model = MyFcn(N_ACTIONS)

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
