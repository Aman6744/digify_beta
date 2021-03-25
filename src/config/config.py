import os
import string
from glob import glob

# define paths
raw_path = os.path.join("..", "data", "raw")
source_path = os.path.join("..", "data", "dataset_hdf5", "dataset.hdf5")
output_path = os.path.join("..", "output")
target_path = os.path.join(output_path, "checkpoint_weights_bentham_mixed.hdf5")
json_file = os.path.join(output_path, "initial_params.json")
ensemble_checkpoint_weights = sorted(glob(os.path.join(output_path, "*.hdf5")))

# define input size, number max of chars per line and list of valid chars
target_image_size = (1024, 128, 1)
maxTextLength = 128
charset = string.printable[:95]
buf_size = 10000
epochs = 1000
batch_size = 32
prefetch_size = 10

#define augmentation parameters
rotation_range = 1.5 
scale_range = 0.05
height_shift_range = 0.025
width_shift_range = 0.05
erode_range = 5
dilate_range = 3

#preprocess parameters
gamma = 0.3
