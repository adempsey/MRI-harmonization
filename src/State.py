import numpy as np
import sys

class State():
    def __init__(self, size, move_range):
        self.image = np.zeros(size,dtype=np.float32)
        self.move_range = move_range

    def reset(self, x):
        self.image = x
        size = self.image.shape
        prev_state = np.zeros((size[0],16,size[2],size[3],size[4]),dtype=np.float32)
        self.tensor = np.concatenate((self.image, prev_state), axis=1)

    def set(self, x):
        self.image = x
        self.tensor[:,:self.image.shape[1],:,:,:] = self.image

    def step(self, act, inner_state):
        move = act.astype(np.float32)
        # moveLabels = move.copy()
        #
        # moveLabels = np.where(move == 5,  "+1", moveLabels)
        # moveLabels = np.where(move == 6,  "-1", moveLabels)
        # moveLabels = np.where(move == 7,  "-1000", moveLabels)
        # moveLabels = np.where(move == 8,  "+1000", moveLabels)
        # moveLabels = np.where(move == 4,  "+100", moveLabels)
        # moveLabels = np.where(move == 3,  "+10", moveLabels)
        # moveLabels = np.where(move == 1, "-10", moveLabels)
        # moveLabels = np.where(move == 0, "-100", moveLabels)
        # moveLabels = np.where(move == 2,  "0", moveLabels)
        #
        # uni, counts = np.unique(moveLabels, return_counts=True)
        # print(dict(zip(uni,counts)))

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
