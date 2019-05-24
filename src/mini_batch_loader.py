import os
import numpy as np
from glob import glob
import nibabel as nib
import random
from scipy import io
import itk
import nrrd
from config import *

class MiniBatchLoader(object):

    """
    @param train_path - the path of the directory containing training images
    @param test_path - the path of the directory containing test data
    """
    def __init__(self, train_path, test_path, crop_size):
        self.training_path_infos = self.read_paths(train_path)
        self.testing_path_infos = self.read_paths(test_path)

        self.crop_size = crop_size

    """
    - Counts the number of files in a directory (for determining number of
    training images, etc.)
    @param path - the file path to search
    @return - integer representing number of files in directory at path
    """
    @staticmethod
    def count_paths(path):
        return len([n for n in os.listdir(path) if os.path.splitext(n)[1] == '.nii'])

    """
    - Returns a list of nifti files in the directory at txt_path
    @param txt_path - the file path to search
    @return - list of strings representing nifti files at directory
    """
    @staticmethod
    def read_paths(txt_path):
        return glob(os.path.join(txt_path,"*.nii"))

    """
    - Load training images at the provided indices
    @param indices - numpy array of integers indicating which input images
    to load (bound from 0 to total number of input images)
    @return - tuple of numpy matrices, (xs, ys), where xs is the set of input
    images and ys is the set of target images
    """
    def load_training_data(self, indices):
        return self.load_data(self.training_path_infos, indices, train=True)

    """
    - Load all testing images
    @return - tuple, (xs, maxIntensity, imgFilename, imgAffine), where xs is
    the set of test images, maxIntensity
    """
    def load_testing_data(self):
        return self.load_data(self.testing_path_infos, np.array([0]))

    """
    - Convert an ANTs-style transform into a numpy matrix
     (see https://github.com/ANTsX/ANTs/wiki/ITK-affine-transform-conversion)
    @param mat - An ANTs matrix object imported from an affine mat file
    @return - a numpy matrix representing the transform
    """
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

    """
    - Convert a voxel coordinate to world coordinates according to the provided
    image matrix
    @param coord - the voxel coordinate to transform
    @param img - the image containing the voxel coordinate
    @return - a 3D coordinate in world units
    """
    def voxel_to_world(self, coord, img):
        return img.TransformIndexToPhysicalPoint(coord)

    """
    - Convert a world coordinate to voxel coordinates according to the provided
    image matrix
    @param coord - the world coordinate to transform
    @param img - the image containing the world coordinate
    @return - a 3D coordinate in voxel units
    """
    def world_to_voxel(self, coord, img):
        return img.TransformPhysicalPointToIndex(coord)

    """
    - Convert a world coordinate to voxel coordinates according to the provided
    image matrix
    @param coord - the world coordinate to transform
    @param img - the image containing the world coordinate
    @return - a 3D coordinate in voxel units
    """
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

    """
    - Randomly select a target image from the same subject as the image
    at the path parameter
    @param path - The path of the input image to search with
    @return - The path of a corresponding target image
    """
    def labelPathFromPath(self, path):
        fName = os.path.basename(path)

        # FIXME: this does not generalize to any filename structure
        # Currently requires format ss_002_S_xxxx_... format, where
        # subject is the 002_S_xxxx portion
        subject = fName[3:13]
        labels = glob(os.path.join(TARGET_DATA_PATH,'*%s*.nii' % subject))
        labelPath = random.choice(labels)

        return labelPath

    """
    - Locate the filepaths for the atlas transforms associated with an image
    @param path - The filepath of the image to search with
    @return - affPath, the path to the affine transform; warpPath, the path
    to the warp field; invWarpPath, the path to the inverse warp field
    """
    def transformPathsFromPath(self, path):
        fName = os.path.basename(path)

        # FIXME: this does not generalize to any filename structure
        # Currently requires format ss_002_S_xxxx_... format, where
        # subject is the 002_S_xxxx portion
        subject = fName[3:13]
        full = os.path.splitext(fName)[0]

        affPath = glob(os.path.join(TRANSFORMATION_DATA_PATH,subject,'*%s*.mat' % full))[0]
        warpPath = glob(os.path.join(TRANSFORMATION_DATA_PATH,subject,'*%s*[!Inverse]Warp.nii' % full))[0]
        invWarpPath = glob(os.path.join(TRANSFORMATION_DATA_PATH,subject,'*%s*InverseWarp.nii' % full))[0]

        return affPath, warpPath, invWarpPath

    """
    - Locate the atlas image corresponding to an image from a particular subject
    @param path - The filepath of the image to search with
    @return - The path to the atlas image
    """
    def atlasPathFromPath(self, path):
        fName = os.path.basename(path)
        subject = fName[3:13]
        return glob(os.path.join(ATLAS_PATH,subject,'*'))[0]

    """
    - Load image data for training or testing. If training, images are cropped
    to a random 15x15x15 patch.
    @param path_infos - A list of filepaths to input images
    @param indices - Which indices in the filepath list to consider for the
    current batch
    @param train - Whether or not the data is to be loaded for training or
    testing
    @return -
        if testing: the image, the maximum intensity of the image, the image
        filename, and the image's affine matrix
        if training: the batch of images, the batch of target images
    """
    def load_data(self, path_infos, indices, train=False):
        mini_batch_size = len(indices)
        in_channels = 1

        if train == True:
            xs = np.zeros((mini_batch_size,
                           in_channels,
                           self.crop_size,
                           self.crop_size,
                           self.crop_size)).astype(np.float32)
            ys = np.zeros((mini_batch_size,
                           in_channels,
                           self.crop_size,
                           self.crop_size,
                           self.crop_size)).astype(np.float32)

            for i, index in enumerate(indices):
                path = path_infos[index]
                labelPath = self.labelPathFromPath(path)
                affPath, warpPath, invWarpPath = self.transformPathsFromPath(path)
                labelAffPath, labelWarpPath, labelInvWarpPath = self.transformPathsFromPath(labelPath)
                atlasPath = self.atlasPathFromPath(path)

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

                # Restrict random cropping to a central location in the image
                # to avoid training on too much blank space
                xRange = 20
                yRange = 70
                zRange = 70

                rand_range_x = x-self.crop_size-(xRange*2)
                rand_range_y = y-self.crop_size-(yRange*2)
                rand_range_z = z-self.crop_size-(zRange*2)
                x_offset = np.random.randint(rand_range_x)+xRange
                y_offset = np.random.randint(rand_range_y)+yRange
                z_offset = np.random.randint(rand_range_z)+zRange

                img = img[x_offset:x_offset+self.crop_size, y_offset:y_offset+self.crop_size,z_offset:z_offset+self.crop_size]

                # Get a transformed patch of the label image to match
                # the training image
                labelImgWarped = np.zeros((self.crop_size,self.crop_size,self.crop_size))
                for i in range(self.crop_size):
                    for j in range(self.crop_size):
                        for k in range(self.crop_size):
                            originPoint = [x_offset+i,y_offset+j,z_offset+k]
                            refPoint = self.transformPoint(originPoint, imgAff, labelAff, imgInvWarp, labelWarp, imgITK, labelITK, atlasITK)
                            labelImgWarped[i,j,k] = labelImg[refPoint[0],refPoint[1],refPoint[2]]

                labelImg = labelImgWarped

                # Normalize images
                if img.max() > 0 and labelImg.max() > 0:
                    img = img.astype(np.float32)
                    labelImg = labelImg.astype(np.float32)
                    img = (img / img.max())
                    labelImg = (labelImg / labelImg.max())
                else:
                    img = np.zeros(img.shape)
                    labelImg = np.zeros(labelImg.shape)

                xs[i, 0, :, :, :] = img.astype(np.float32)
                ys[i, 0, :, :, :] = labelImg.astype(np.float32)
                return xs, ys

        else:
            for i, index in enumerate(indices):
                path = path_infos[index]

                imgNib = nib.load(path)
                imgAffine = imgNib.affine
                img = np.array(imgNib.dataobj)
                if img is None:
                    raise RuntimeError("invalid image: {i}".format(i=path))

            img = img.astype(np.float32)
            maxIntensity = img.max()

            # Normalize image
            x, y, z = img.shape
            xs = np.zeros((mini_batch_size, in_channels, x, y, z)).astype(np.float32)
            xs[0, 0, :, :, :] = (img/img.max()).astype(np.float32)

            imgFileName = os.path.splitext(os.path.basename(path))[0]
            return xs, maxIntensity, imgFileName, imgAffine
