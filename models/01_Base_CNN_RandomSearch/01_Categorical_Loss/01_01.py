"""Base CNN grid search (categorical loss), sharded across a SLURM array.

Trains a slice of the hyperparameter grid per array task using the current
engine in src/cnn_base.py + src/base_dataset.py. Each trained model writes its
outputs to ./array_task{slurm_array}/model{index}_{avg_val_acc}/.
"""

import argparse
import sys
from pathlib import Path

from sklearn.model_selection import ParameterGrid

# Make the repo's src/ importable. Uses the script location (works locally and
# on the HPC, where the rsync'd tree has no .git to search for).
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT / "src"))

import base_dataset  # noqa: E402
import cnn_base  # noqa: E402

DEFAULT_DATASET = Path.home() / "data" / "datasets" / "base_64" / "base_64.npy"

parser = argparse.ArgumentParser(description="Base CNN grid search (categorical loss)")
parser.add_argument(
    "-a", "--slurm_array", type=int, required=True, help="SLURM array task number (1-based)"
)
parser.add_argument(
    "-p", "--per_array", type=int, required=True, help="number of models trained per array task"
)
parser.add_argument(
    "-d",
    "--dataset",
    type=str,
    default=str(DEFAULT_DATASET),
    help="path to the prebuilt base_64 .npy dataset",
)
args = parser.parse_args()
slurm_array, per_array = args.slurm_array, args.per_array

# Load the prebuilt base dataset. Pool train+val for k-fold CV; the test split
# is held out and passed through so train_model reports test metrics too.
dataset = base_dataset.load_dataset(args.dataset)
combined_training_dataset = base_dataset.combine_train_val(dataset)
print(f"Combined train+val size: {len(combined_training_dataset['labels'])}")

# Hyperparameter grid (5 * 2 * 3 * 3 * 3 = 270 combinations).
params = {
    "num_filters": [8, 16, 32, 64, 128],
    "filter_size": [3],
    "learning_rate": [0.01, 0.001],
    "epochs": [50],
    "k": [5],
    "num_layers": [2, 3, 4],
    "pooling_size": [2],
    "activation_function": ["relu"],
    "batch_size": [64],
    "reg": [None, "L1", "L2"],
    "opt": ["Adam", "Momentum", "RMSProp"],
    "dropout": [0],
}
param_grid = list(ParameterGrid(params))

# Shard the grid across the SLURM array (with per_array=6, array 1-45 covers all 270).
start_index = (slurm_array - 1) * per_array
end_index = slurm_array * per_array
selected_param_sets = param_grid[start_index:end_index]

for index, selected_params in enumerate(selected_param_sets):
    cnn_base.train_model(
        slurm_array,
        index,
        selected_params,
        combined_training_dataset["images"],
        combined_training_dataset["labels"],
        dataset["test_images"],
        dataset["test_labels"],
    )
