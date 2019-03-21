import os
import numpy as np
import cv2
from glob import glob
import nibabel as nib
import random
from scipy import io

MAX_INTENSITY = (2**15)-1

class MiniBatchLoader(object):

    def __init__(self, train_path, test_path, image_dir_path, crop_size):

        # load data paths
        self.training_path_infos = self.read_paths(train_path)
        self.testing_path_infos = self.read_paths(test_path)

        self.crop_size = crop_size

    # test ok
    @staticmethod
    def path_label_generator(txt_path, src_path):
        for line in open(txt_path):
            line = line.strip()
            src_full_path = os.path.join(src_path, line)
            if os.path.isfile(src_full_path):
                yield src_full_path

    # test ok
    @staticmethod
    def count_paths(path):
        return len([n for n in os.listdir(path) if os.path.splitext(n)[1] == '.nii'])
        # c = 0
        # for _ in open(path):
        #     c += 1
        # return c

    # test ok
    @staticmethod
    def read_paths(txt_path):
        return glob(os.path.join(txt_path,"*.nii"))
        # cs = []
        # for pair in MiniBatchLoader.path_label_generator(txt_path, src_path):
        #     cs.append(pair)
        # return cs

    def load_training_data(self, indices):
        return self.load_data(self.training_path_infos, indices, augment=True)

    def load_testing_data(self, indices):
        return self.load_data(self.testing_path_infos, indices)
        # return self.load_data(self.testing_path_infos, indices)

    def antsmat2mat(self, mat):
        finalMat = np.zeros((4,4))

        m_Matrix = mat['AffineTransform_float_3_3']
        m_Center = mat['fixed']
        m_Translation = m_Matrix[-3:]

        m_Matrix = np.reshape(m_Matrix[:9],(3,3))

        finalMat[0:3,0:3] = m_Matrix
        finalMat[3,3] = 1.

        offset = np.zeros(3)
        for i in range(0,3):
            offset[i] = m_Translation[i] + m_Center[i]
            for j in range(0,3):
                offset[i] -= m_Matrix[i,j] * m_Center[j]
        finalMat[0:3,3] = offset

        return finalMat

    # test ok
    def load_data(self, path_infos, indices, augment=False):
        mini_batch_size = len(indices)
        in_channels = 1
        def labelPathFromPath(path):
            fName = os.path.basename(path)
            subject = fName[3:13]
            labels = glob(os.path.join(os.path.dirname(path),'..','label2','ss_%s*.nii' % subject))
            labelPath = random.choice(labels)

            return labelPath

        def transformPathsFromPath(path):
            fName = os.path.basename(path)
            subject = fName[3:13]
            full = fName[:-4]

            affPath = glob(os.path.join(os.path.dirname(path),'..','transforms',subject,'antsBTP%s*.mat' % full))[0]
            warpPath = glob(os.path.join(os.path.dirname(path),'..','transforms',subject,'antsBTP%s*[0-9]Warp.nii' % full))[0]
            invWarpPath = glob(os.path.join(os.path.dirname(path),'..','transforms',subject,'antsBTP%s*InverseWarp.nii' % full))[0]

            return affPath, warpPath, invWarpPath

        if augment:
            xs = np.zeros((mini_batch_size, in_channels, self.crop_size, self.crop_size, self.crop_size)).astype(np.float32)
            ys = np.zeros((mini_batch_size, in_channels, self.crop_size, self.crop_size, self.crop_size)).astype(np.float32)

            for i, index in enumerate(indices):
                path = path_infos[index]
                labelPath = labelPathFromPath(path)
                affPath, warpPath, invWarpPath = transformPathsFromPath(path)
                labelAffPath, labelWarpPath, labelInvWarpPath = transformPathsFromPath(labelPath)

                img = np.array(nib.load(path).dataobj)
                labelImg = np.array(nib.load(labelPath).dataobj)

                imgAff = np.linalg.inv(self.antsmat2mat(io.loadmat(affPath)))
                labelAff = self.antsmat2mat(io.loadmat(labelAffPath))
                imgInvWarp = np.array(nib.load(invWarpPath).dataobj)
                labelWarp = np.array(nib.load(labelWarpPath).dataobj)

                if img is None or labelImg is None:
                    raise RuntimeError("invalid image: {i}".format(i=path))
                x, y, z = img.shape
                if np.random.rand() > 0.5:
                    img = np.fliplr(img)
                    labelImg = np.fliplr(labelImg)

                # if np.random.rand() > 0.5:
                #     angle = 10*np.random.rand()
                #     if np.random.rand() > 0.5:
                #         angle *= -1
                #     M = cv2.getRotationMatrix2D((w/2,h/2),angle,1)
                #     img = cv2.warpAffine(img,M,(w,h))
                #     labelImg = cv2.warpAffine(labelImg,M,(w,h))

                rand_range_x = x-self.crop_size
                rand_range_y = y-self.crop_size
                rand_range_z = z-self.crop_size
                x_offset = np.random.randint(rand_range_x)
                y_offset = np.random.randint(rand_range_y)
                z_offset = np.random.randint(rand_range_z)
                img = img[x_offset:x_offset+self.crop_size, y_offset:y_offset+self.crop_size,z_offset:z_offset+self.crop_size]
                labelImg = labelImg[x_offset:x_offset+self.crop_size, y_offset:y_offset+self.crop_size,z_offset:z_offset+self.crop_size]

                if img.max() > 0:
                    img = img.astype(np.float32)
                    labelImg = labelImg.astype(np.float32)
                    img = (img / img.max()) * labelImg.max()

                xs[i, 0, :, :, :] = (img/MAX_INTENSITY).astype(np.float32)
                ys[i, 0, :, :, :] = (labelImg/MAX_INTENSITY).astype(np.float32)

                return xs, ys, imgAff, labelAff, imgInvWarp, labelWarp

        elif mini_batch_size == 1:
            for i, index in enumerate(indices):
                path = path_infos[index]
                # labelPath = labelPathFromPath(path)
                labelPath =  os.path.join('..','adni3','label','antsBTPtemplate0ss_002_S_0559_2007-10-18_17_06_13.03WarpedToTemplate.nii')#labelPathFromPath(path)

                # img = cv2.imread(path,0)
                img = np.array(nib.load(path).dataobj)
                labelImg = np.array(nib.load(labelPath).dataobj)
                # labelImg = cv2.imread(labelPath,0)
                if img is None or labelImg is None:
                    raise RuntimeError("invalid image: {i}".format(i=path))

            if img.max() > 0:
                img = img.astype(np.float32)
                labelImg = labelImg.astype(np.float32)
                img = (img / img.max()) * labelImg.max()

            x, y, z = img.shape
            xs = np.zeros((mini_batch_size, in_channels, x, y, z)).astype(np.float32)
            ys = np.zeros((mini_batch_size, in_channels, x, y, z)).astype(np.float32)
            xs[0, 0, :, :, :] = (img/MAX_INTENSITY).astype(np.float32)
            ys[0, 0, :, :, :] = (labelImg/MAX_INTENSITY).astype(np.float32)

            return xs, ys

        else:
            raise RuntimeError("mini batch size must be 1 when testing")

        # return xs, ys
