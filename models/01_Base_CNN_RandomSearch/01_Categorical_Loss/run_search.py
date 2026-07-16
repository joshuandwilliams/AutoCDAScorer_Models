"""Base CNN random search (categorical loss), sharded across a SLURM array.

Modular by design: the search harness is decoupled from
  (a) the dataset      -> --dataset points at any base_dataset .npy,
  (b) the params table -> PARAM_SPACE below,
  (c) the model        -> the ModelBuilder passed to RandomSearch.
Swap CNNModelBuilder for a ViT builder, or the base dataset for a GAN one, without
changing the search itself.
"""

import argparse
import sys
from pathlib import Path

# Make the repo's src/ importable (works locally and on the HPC, no .git needed).
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT / "src"))

import base_dataset  # noqa: E402
from model_builders import CNNModelBuilder  # noqa: E402
from random_search import RandomSearch  # noqa: E402

# The base dataset travels with the repo (data/ is synced to the HPC), so default
# to the repo-relative copy. Override with --dataset to point elsewhere.
DEFAULT_DATASET = REPO_ROOT / "data" / "datasets" / "base_64" / "base_64.npy"

# Hyperparameter space sampled by the random search. `k` (CV folds) is a search
# setting, not a model hyperparameter, so it lives on RandomSearch, not here.
PARAM_SPACE = {
    "num_filters": [8, 16, 32, 64],
    "filter_size": [3, 5],
    "learning_rate": [0.01, 0.001, 0.0001],
    "epochs": [50],
    "num_layers": [2, 3, 4],
    "pooling_size": [2],
    "activation_function": ["relu", "elu"],
    "batch_size": [64],
    "reg": [None, "L1", "L2"],
    "reg_strength": [0.0001, 0.001, 0.01],
    "opt": ["Adam", "Momentum"],
    "dropout": [0.0, 0.2, 0.3, 0.5],
}


def main():
    parser = argparse.ArgumentParser(description="Base CNN random search (categorical loss)")
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
        CNNModelBuilder(),
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
