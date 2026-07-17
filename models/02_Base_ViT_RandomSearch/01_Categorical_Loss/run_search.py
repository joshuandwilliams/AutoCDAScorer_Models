"""Base ViT grid/random search (categorical loss), sharded across a SLURM array.

The exact same harness as the Base CNN search (``01_Base_CNN_RandomSearch``) -- only
two things change:
  (b) the params table -> PARAM_SPACE below (ViT hyperparameters),
  (c) the model        -> the ViTModelBuilder passed to RandomSearch.
The dataset (a) is still the Base 64 dataset, and the CV harness, self-scheduling
grid, shared global top-N and metrics are all unchanged. That is the whole point of
the ModelBuilder abstraction: swap CNN -> ViT without touching the search.
"""

import argparse
import sys
from pathlib import Path

# Make the repo's src/ importable (works locally and on the HPC, no .git needed).
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT / "src"))

import base_dataset  # noqa: E402
from model_builders import ViTModelBuilder  # noqa: E402
from random_search import RandomSearch  # noqa: E402

# The base dataset travels with the repo (data/ is synced to the HPC), so default
# to the repo-relative copy. Override with --dataset to point elsewhere.
DEFAULT_DATASET = REPO_ROOT / "data" / "datasets" / "base_64" / "base_64.npy"

# Hyperparameter space sampled by the search. `k` (CV folds) is a search setting, not
# a model hyperparameter, so it lives on RandomSearch, not here.
#
# A genuine, broad architecture search (NOT a narrow sweep around assumed-good values).
# Every combination below is valid on 64x64 crops (each patch_size tiles 64, and every
# projection_dim is divisible by every num_heads), so the full grid is
#   3 (patch) x 3 (proj) x 2 (heads) x 3 (depth) x 3 (lr) x 3 (dropout) x 2 (reg) x 2 (opt)
#   = 1,944 configs.
# This is much heavier than the CNN sweep per config (attention + up to 50 epochs x
# 5-fold CV), and patch_size=4 in particular gives a 256-token sequence, so this grid is
# meant to run on the GPU queue -- see run_search.slurm.sh (jic-gpu, --gres, --nv).
PARAM_SPACE = {
    "patch_size": [4, 8, 16],
    "projection_dim": [32, 64, 128],
    "num_heads": [4, 8],
    "num_transformer_layers": [2, 4, 6],
    "learning_rate": [0.01, 0.001, 0.0001],
    "epochs": [50],
    "mlp_ratio": [2],
    "activation_function": ["gelu"],
    "dropout": [0.0, 0.1, 0.3],
    "reg": [None, "L2"],
    "reg_strength": [0.0001],
    "opt": ["Adam", "AdamW"],
    "batch_size": [64],
}


def main():
    parser = argparse.ArgumentParser(description="Base ViT grid/random search (categorical loss)")
    parser.add_argument("-a", "--slurm_array", type=int, default=1, help="SLURM array task number")
    parser.add_argument(
        "-m",
        "--mode",
        choices=["grid", "random"],
        default="grid",
        help="grid = self-scheduled full grid search (default); random = sample n_models",
    )
    parser.add_argument(
        "-n", "--n_models", type=int, default=4, help="models to sample per task (random mode only)"
    )
    parser.add_argument("-k", "--folds", type=int, default=5, help="k for k-fold cross-validation")
    parser.add_argument(
        "-c",
        "--cv_seed",
        type=int,
        default=42,
        help="CV fold-split seed; keep the SAME across array tasks for a fair global ranking",
    )
    parser.add_argument(
        "-t", "--top_n", type=int, default=100, help="keep plots/models for the top N by avg_vaf"
    )
    parser.add_argument(
        "-d", "--dataset", type=str, default=str(DEFAULT_DATASET), help="path to base_64 .npy"
    )
    args = parser.parse_args()

    # Load the prebuilt base dataset; pool train+val for CV, hold out the test split.
    dataset = base_dataset.load_dataset(args.dataset)
    combined = base_dataset.combine_train_val(dataset)
    print(f"Combined train+val size: {len(combined['labels'])}")

    # Seed and id by array task: each task explores a different slice and contributes
    # to one shared global top-N in the current dir (01_Categorical_Loss).
    search = RandomSearch(
        ViTModelBuilder(),
        PARAM_SPACE,
        k=args.folds,
        seed=args.slurm_array,
        cv_seed=args.cv_seed,
        run_id=args.slurm_array,
    )
    common = dict(
        train_images=combined["images"],
        train_labels=combined["labels"],
        test_images=dataset["test_images"],
        test_labels=dataset["test_labels"],
        global_dir=".",
        top_n=args.top_n,
    )
    if args.mode == "grid":
        search.run_grid(**common)
    else:
        search.run(n_models=args.n_models, **common)


if __name__ == "__main__":
    main()
