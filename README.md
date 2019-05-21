# Harmonization of Structural MR Images Through Voxelwise Deep Reinforcement Learning
This is the implementation of the code used in Andrew Dempsey's master's thesis at NYU Tandon School of Engineering.

## Setup
The model environment can be set up using Docker. We recommend using [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) to enable training with GPUs.

Build the image by running:
```
nvidia-docker build -t python:harm .
```

Enter the Docker environment by running:
```
nvidia-docker run -it --rm -v <absolute_path_to_your_data_directory>:/app/data python:harm
```

## Usage

### Test with pretrained model
A set of pre-trained weights are provided in `src/model`. To run the model
with these weights (or any generated weights from another training session),
run
```
python main.py
```
ensuring that `WEIGHT_PATH` in `config.py` points to the `npz` file you wish
to load.

### Training
To train the model, run the following:
```
python main.py -t
```
Weights will be saved to the directory specified in the `SAVE_PATH` in `config.py`. Weights are saved after the network runs through the number of
iterations specified in `SNAPSHOT_EPISODES`.

## Data
You should create a `data` directory with the following subdirectories:
- A directory containing input images to use for training
- A directory containing label images to training
- A directory containing input images to use for testing
- A directory, organized by subject, containing affine matrices in `.mat` format, and warp and inverse warp field files in `.nii` format
- A directory containing atlas images for each subject
- An empty directory to output images during testing

A sample structure might look like this:
```
.
+--atlases
|  +--subject_1
|  |  +--subject_1_atlas.nii
|  ...
+--input
|  |  +--input_image_1.nii
|  |  +--input_image_2.nii
|  |  ...
+--label
|  |  +--label_image_1.nii
|  |  +--label_image_2.nii
|  |  ...
+--test
|  |  +--test_image_1.nii
|  |  +--test_image_2.nii
|  |  ...
+--transforms
|  +--subject_1
|  |  +--input_image_1_affine.mat
|  |  +--input_image_1_warp.nii
|  |  +--input_image_1_inverse_warp.nii
|  |  +--label_image_1_affine.mat
|  |  +--label_image_1_Warp.nii
|  |  +--label_image_1_InverseWarp.nii
|  |  ...
|  ...
```

Transform warp field files should end in `Warp` and `InverseWarp` and be in nifti format. Affine matrices should be in `.mat` format. All transform files
should have matching names as their corresponding images up to the `Warp` or
`InverseWarp` suffixes.

## Abstract
Longitudinal analysis of structural magnetic resonance images is often inhibited by changes in scanner hardware, which can introduce differences in contrast, noise, and resolution. The field of image harmonization aims to control for this variability by adjusting images to a consistent baseline while preserving overall structure. Recent advancements in image-based deep reinforcement learning have produced models where independent, pixelwise agents can be trained in parallel to achieve an overall goal. Here, we apply a voxelwise reinforcement learning approach to the harmonization task. During training, each agent learns a policy for selecting elementary transformations to adapt images to a harmonized environment. We train our network to learn the relationship between two scanners, and show that this method can help achieve more consistent performance in image segmentation tasks by quantitatively measuring volume difference and Dice similarity of segmentation volumes.

## References
This work is based on research conducted in [(Furuta, 2019)](https://github.com/rfuruta/pixelRL).
