import os
import numpy as np
import cv2
from glob import glob
import nibabel as nib

MAX_INTENSITY = (2**15)-1

class MiniBatchLoader(object):

    def __init__(self, train_path, test_path, image_dir_path, crop_size):

        # load data paths
        self.training_path_infos = self.read_paths(train_path, image_dir_path)
        self.testing_path_infos = self.read_paths(test_path, image_dir_path)

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
        return len([n for n in os.listdir(path)])
        # c = 0
        # for _ in open(path):
        #     c += 1
        # return c

    # test ok
    @staticmethod
    def read_paths(txt_path, src_path):
        return glob(os.path.join(txt_path,"*.nii"))
        # cs = []
        # for pair in MiniBatchLoader.path_label_generator(txt_path, src_path):
        #     cs.append(pair)
        # return cs

    def load_training_data(self, indices):
        return self.load_data(self.training_path_infos, indices, augment=True)

    def load_testing_data(self, indices):
        return self.load_data(self.training_path_infos, indices)
        # return self.load_data(self.testing_path_infos, indices)

    # test ok
    def load_data(self, path_infos, indices, augment=False):
        mini_batch_size = len(indices)
        in_channels = 1
        def labelPathFromPath(path):
            fName = os.path.basename(path)
            segments = fName.split('_')
            n = segments[0]
            labelPath = glob(os.path.join('..','adni','label','%s*' % n))[0]

            return labelPath

        if augment:
            xs = np.zeros((mini_batch_size, in_channels, self.crop_size, self.crop_size, self.crop_size)).astype(np.float32)
            ys = np.zeros((mini_batch_size, in_channels, self.crop_size, self.crop_size, self.crop_size)).astype(np.float32)

            for i, index in enumerate(indices):
                path = path_infos[index]
                # labelPath = labelPathFromPath(path)
                labelPath = os.path.join('..','adni3','label','antsBTPtemplate0ss_002_S_0559_2007-10-18_17_06_13.03WarpedToTemplate.nii')

                # img = cv2.imread(path,0)
                img = np.array(nib.load(path).dataobj)
                labelImg = np.array(nib.load(labelPath).dataobj)
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

        elif mini_batch_size == 1:
            for i, index in enumerate(indices):
                path = path_infos[index]
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

        else:
            raise RuntimeError("mini batch size must be 1 when testing")

        return xs, ys
