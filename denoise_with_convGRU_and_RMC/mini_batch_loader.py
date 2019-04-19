import os
import numpy as np
import cv2
from glob import glob
import nibabel as nib
import random
from scipy import io
import itk
import nrrd

MAX_INTENSITY = float((2**15)-1)

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

    def voxel_to_world(self, coord, img):
        return img.TransformIndexToPhysicalPoint(coord)

    def world_to_voxel(self, coord, img):
        return img.TransformPhysicalPointToIndex(coord)

    def transformPoint(self, point, invAff, aff, invWarp, warp, imgX, imgY, atlas):
        # Convert src voxel coordinate indices to world coordinates
        # point = [70,91,158]
        src_coord_world = self.voxel_to_world(point, imgX)

        # Transform src world coordinate into atlas coord
        src_coord_world_h = np.append(src_coord_world, [1])
        atl_coord_world_prewarp = np.matmul(invAff, src_coord_world_h)

        # Get voxel coords of pre-warped atlas coord
        atl_coord_vox_prewarp = self.world_to_voxel(atl_coord_world_prewarp[:3],atlas)

        # Get inverse deformation vector at pre-warped atlas coord
        invwarp_x = invWarp[atl_coord_vox_prewarp[0],atl_coord_vox_prewarp[1],atl_coord_vox_prewarp[2],:,0]
        invwarp_y = invWarp[atl_coord_vox_prewarp[0],atl_coord_vox_prewarp[1],atl_coord_vox_prewarp[2],:,1]
        invwarp_z = invWarp[atl_coord_vox_prewarp[0],atl_coord_vox_prewarp[1],atl_coord_vox_prewarp[2],:,2]

        inv_warp_vec = np.array([invwarp_x[0], invwarp_y[0], invwarp_z[0]])

        # Add deformation vector to pre-warped world atlas coordinates
        atl_coord_world = atl_coord_world_prewarp[:3] + inv_warp_vec

        # Get voxel coords of atlas coord
        atl_coord_vox = self.world_to_voxel(atl_coord_world, atlas)

        # Get deformation vector at atlas coord
        warp_x = warp[atl_coord_vox[0],atl_coord_vox[1],atl_coord_vox[2],:,0]
        warp_y = warp[atl_coord_vox[0],atl_coord_vox[1],atl_coord_vox[2],:,1]
        warp_z = warp[atl_coord_vox[0],atl_coord_vox[1],atl_coord_vox[2],:,2]

        warp_vec = np.array([warp_x[0], warp_y[0], warp_z[0]])

        # Add deformation vector to world atlas coordinates
        atl_coord_world_warped = atl_coord_world + warp_vec

        # Transform atl world coordinate into dst coord
        atl_coord_world_warped_h = np.append(atl_coord_world_warped, [1])
        dst_coord_world = np.matmul(aff, atl_coord_world_warped_h)

        dst_coord_vox = self.world_to_voxel(dst_coord_world[:3], imgY)

        return np.array(dst_coord_vox)


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

        def atlasPathFromPath(path):
            fName = os.path.basename(path)
            subject = fName[3:13]
            return os.path.join(os.path.dirname(path),'..','atlases',subject,'antsBTPtemplate0.nii')

        if augment:
            xs = np.zeros((mini_batch_size, in_channels, self.crop_size, self.crop_size, self.crop_size)).astype(np.float32)
            ys = np.zeros((mini_batch_size, in_channels, self.crop_size, self.crop_size, self.crop_size)).astype(np.float32)

            for i, index in enumerate(indices):
                # path = path_infos[index]
                path = os.path.join('..','adni3','train2','ss_002_S_1018_2006-11-29_10_09_05.0.nii')
                # path = os.path.join('..','adni3','train2','ss_002_S_1261_2009-02-05_10_32_22.0.nii')
                labelPath = labelPathFromPath(path)
                # labelPath = os.path.join('..','adni3','label2','ss_002_S_1261_2007-02-27_13_40_08.0.nii')
                affPath, warpPath, invWarpPath = transformPathsFromPath(path)
                labelAffPath, labelWarpPath, labelInvWarpPath = transformPathsFromPath(labelPath)
                atlasPath = atlasPathFromPath(path)

                img = np.array(nib.load(path).dataobj)
                labelImg = np.array(nib.load(labelPath).dataobj)

                imgAff = np.linalg.inv(self.antsmat2mat(io.loadmat(affPath)))
                labelAff = self.antsmat2mat(io.loadmat(labelAffPath))
                imgInvWarp = np.array(nib.load(invWarpPath).dataobj)
                labelWarp = np.array(nib.load(labelWarpPath).dataobj)

                imgITK = itk.imread(path)
                labelITK = itk.imread(labelPath)
                atlasITK = itk.imread(atlasPath)

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

                xRange = 20
                yRange = 70
                zRange = 70

                rand_range_x = x-self.crop_size-(xRange*2)
                rand_range_y = y-self.crop_size-(yRange*2)
                rand_range_z = z-self.crop_size-(zRange*2)
                x_offset = np.random.randint(rand_range_x)+xRange
                y_offset = np.random.randint(rand_range_y)+yRange
                z_offset = np.random.randint(rand_range_z)+zRange

                # print(i,x_offset,y_offset,z_offset)

                img = img[x_offset:x_offset+self.crop_size, y_offset:y_offset+self.crop_size,z_offset:z_offset+self.crop_size]
                labelImg = labelImg[x_offset:x_offset+self.crop_size, y_offset:y_offset+self.crop_size,z_offset:z_offset+self.crop_size]

                # labelImgWarped = np.zeros((self.crop_size,self.crop_size,self.crop_size))
                # for i in range(self.crop_size):
                #     for j in range(self.crop_size):
                #         for k in range(self.crop_size):
                #             originPoint = [x_offset+i,y_offset+j,z_offset+k]
                #             refPoint = self.transformPoint(originPoint, imgAff, labelAff, imgInvWarp, labelWarp, imgITK, labelITK, atlasITK)
                #             # print(path, labelPath, originPoint, refPoint)
                #             labelImgWarped[i,j,k] = labelImg[refPoint[0],refPoint[1],refPoint[2]]
                #
                # # labelImg = labelImg[x_offset:x_offset+self.crop_size, y_offset:y_offset+self.crop_size,z_offset:z_offset+self.crop_size]
                # # print(x_offset,y_offset,z_offset, img.min(), img.max())
                # labelImg = labelImgWarped

                if img.max() > 0 and labelImg.max() > 0:
                    img = img.astype(np.float32)
                    labelImg = labelImg.astype(np.float32)
                    img = (img / img.max())# * labelImg.max()
                    labelImg = (labelImg / labelImg.max())
                else:
                    img = np.zeros(img.shape)
                    labelImg = np.zeros(labelImg.shape)

                # nrrd.write('./uh/original.nrrd',img)
                # nrrd.write('./uh/warpy.nrrd',labelImg)

                # print(x_offset,y_offset,z_offset, img.min(),img.max(), labelImg.min(), labelImg.max())

                xs[i, 0, :, :, :] = img.astype(np.float32)#(img/MAX_INTENSITY).astype(np.float32)
                ys[i, 0, :, :, :] = labelImg.astype(np.float32)#(labelImg/MAX_INTENSITY).astype(np.float32)
                return xs, ys#, imgAff, labelAff, imgInvWarp, labelWarp, imgITK, labelITK, atlasITK

        elif mini_batch_size == 1:
            for i, index in enumerate(indices):
                path = path_infos[index]
                # labelPath = labelPathFromPath(path)

                img = np.array(nib.load(path).dataobj)
                # labelImg = np.array(nib.load(labelPath).dataobj)
                if img is None:# or labelImg is None:
                    raise RuntimeError("invalid image: {i}".format(i=path))

            if img.max() > 0:
                img = img.astype(np.float32)
                # labelImg = labelImg.astype(np.float32)
                img = (img / img.max())
                # labelImg = (labelImg / labelImg.max())

            x, y, z = img.shape
            xs = np.zeros((mini_batch_size, in_channels, x, y, z)).astype(np.float32)
            ys = np.zeros((mini_batch_size, in_channels, x, y, z)).astype(np.float32)
            xs[0, 0, :, :, :] = (img/MAX_INTENSITY).astype(np.float32)
            # ys[0, 0, :, :, :] = (labelImg/MAX_INTENSITY).astype(np.float32)
            ys = None

            return xs, ys

        else:
            raise RuntimeError("mini batch size must be 1 when testing")

        # return xs, ys
