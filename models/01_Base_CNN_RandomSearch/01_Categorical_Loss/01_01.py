import tensorflow as tf
from sklearn.model_selection import ParameterGrid
import os
import argparse
from collections import Counter

import sys
sys.path.append(os.path.expandvars('$HOME/src'))
from cnn_base import load_images_and_labels, train_val_test, downsample_oversized_classes, save_data, train_model

# Arguments
parser = argparse.ArgumentParser(add_help=True, formatter_class=argparse.RawDescriptionHelpFormatter, description="base_CNN")
parser.add_argument("-a", "--slurm_array", help="slurm array number", type=int, default=None)
parser.add_argument("-p", "--per_array", help="models per array", type=int, default=None)
args = parser.parse_args()
slurm_array, per_array = args.slurm_array, args.per_array
model_number = "0004"

# Define train:val:test data
images, labels, filepaths = load_images_and_labels(os.path.expandvars('$HOME/data/GAN_images_1/'))

train_images, train_labels, train_filepaths, val_images, val_labels, val_filepaths, test_images, test_labels, test_filepaths = train_val_test(images, labels, filepaths, train_ratio = 0.8, val_ratio = 0.1)

# Normalise train class sizes
train_images, train_labels, train_filepaths = downsample_oversized_classes(train_images, train_labels, train_filepaths)

# Normalise pixel values
train_images = train_images / 256.0

print(Counter(train_labels))

# Select parameters for model training
params = {
    'num_filters': [8, 16, 32, 64, 128],
    'filter_size': [3],
    'learning_rate': [0.01, 0.001],
    'epochs': [50],
    'k': [5],
    'num_layers': [2, 3, 4],
    'pooling_size': [2],
    'activation_function': ['relu'],
    'batch_size': [64],
    'reg': [None, "L1", "L2"],
    'opt': ["Adam", "Momentum", "RMSProp"],
    'dropout': [0]
}
param_grid = list(ParameterGrid(params))

start_index = (slurm_array-1) * per_array
end_index = slurm_array * per_array
selected_param_sets = param_grid[start_index:end_index]

for index, selected_params in enumerate(selected_param_sets):
    train_model(slurm_array, index, selected_params, train_images, train_labels)

save_data(".", model_number, train_images, train_labels, train_filepaths, val_images, val_labels, val_filepaths, test_images, test_labels, test_filepaths)