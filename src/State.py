import numpy as np
import sys

class State():
    """
    @param size - tuple of image dimensions (batchsize, x, y, z)
    @param move_range - the number of actions for directly modifying voxel
    intensities (as defined in config/MOVE_RANGE)
    """
    def __init__(self, size, move_range):
        self.image = np.zeros(size,dtype=np.float32)
        self.move_range = move_range

    """
    Set up initial state at the beginning of the episode
    @param x - The image to operate on
    """
    def reset(self, x):
        self.image = x
        size = self.image.shape
        prev_state = np.zeros((size[0],16,size[2],size[3],size[4]),dtype=np.float32)
        self.tensor = np.concatenate((self.image, prev_state), axis=1)

    """
    Perform execution of an action on an image
    @param act - a cube with the same dimension as the image composed of
    scalar values corresponding to actions
    @inner_state - GRU hidden state
    """
    def step(self, act, inner_state):
        move = act.astype(np.float32)

        move = np.where(move == 8,  0.01, move)
        move = np.where(move == 7,  -0.01, move)
        move = np.where(move == 5,  0.000001, move)
        move = np.where(move == 6,  -0.000001, move)
        move = np.where(move == 4,  0.0001, move)
        move = np.where(move == 3,  0.00001, move)
        move = np.where(move == 1, -0.00001, move)
        move = np.where(move == 0, -0.0001, move)
        move = np.where(move == 2,  0, move)

        moved_image = self.image + move[:,np.newaxis,:,:]

        self.image = moved_image

        self.tensor[:,:self.image.shape[1],:,:,:] = self.image
        self.tensor[:,-16:,:,:,:] = inner_state
