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
import nibabel as nib
from config import *

"""
- Normalizes the provided image to intensities between 0 and the given max
  and creates a suitable output format
@param img - the image to normalize
@param maxIntensity - the highest intensity integer for the output image
@param imgAffine - the affine matrix of the image (pre-harmonization)
@return - a normalized image in nifti format
"""
def normalizedImage(img, maxIntensity, imgAffine):
    outputImg = np.maximum(0,img)
    outputImg = np.minimum(1,outputImg)
    outputImg = outputImg.squeeze()
    outputImg = (outputImg/outputImg.max())*maxIntensity
    outputImg = nib.Nifti1Image(outputImg.astype(np.float32), imgAffine)

    return outputImg

"""
- Harmonizes all images in the TESTING_DATA_PATH directory and writes the
  results to OUTPUT_PATH
@param loader - instance of mini_batch_loader to load test data
@param agent - initialized agent to perform harmonization task
"""
def test(loader, agent):

    # Set up agent state space
    current_state = state.State((1,1,CROP_SIZE,CROP_SIZE),
                                MOVE_RANGE)

    # Obtain image data from loader
    raw_x, maxIntensity, imgName, imgAffine = loader.load_testing_data()

    # Reset state values to input image intensities
    current_state.reset(raw_x)

    # Iterate through episode steps
    for t in range(0, EPISODE_LEN):
        # Sample and execute action
        action, inner_state = agent.act(current_state.tensor)
        current_state.step(action, inner_state)

    # Halt iteration
    agent.stop_episode()

    # Normalize image and write to disk
    outputImg = normalizedImage(current_state.image, maxIntensity, imgAffine)
    nib.save(outputImg,os.path.join(OUTPUT_PATH,'%s_output.nii' % imgName))

    sys.stdout.flush()

"""
- Set up agent, batch loader, optimizer, and GPU links, and begin testing
"""
def main():

    # Initialize data loader
    mini_batch_loader = MiniBatchLoader(
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
    agent.act_deterministically = True

    # Load weights
    chainer.serializers.load_npz(WEIGHT_PATH, agent.model)
    agent.model.to_gpu()

    # Start testing
    test(mini_batch_loader, agent)

"""
- Entry point for running a test iteration. Harmonizes all images in the
TESTING_DATA_PATH directory and writes results to OUTPUT_PATH
"""
def testModel():
    try:
        fout = open('testlog.txt', "w")
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
