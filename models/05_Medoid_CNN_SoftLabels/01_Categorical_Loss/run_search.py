"""Medoid CNN grid search trained on SOFT labels, sharded across a SLURM array.

Identical to models/03_Medoid_CNN_RandomSearch in every respect -- same medoid_64
dataset, same splits, same PARAM_SPACE, same k=5 folds, same cv_seed -- except for the
training target. That makes it a controlled comparison of the target alone.

The target
----------
Every CDA carries three independent human scores. Every search so far trained on a
one-hot of their MEDIAN, which asserts the raters were unanimous. They usually were not:
of the 3,052 crops in this dataset only 1,033 have all three agreeing, 1,630 have two of
three, and 389 have all three differing. Mean peak mass of the rater distribution is
0.737, so a one-hot target overstates rater certainty on most of the training set.

The soft target is the rater distribution itself: a CDA scored 3, 4, 4 becomes
[0, 0, 0, 1/3, 2/3, 0, 0]. This is NOT label smoothing -- smoothing spreads a chosen
epsilon uniformly over classes nobody voted for, whereas this puts mass exactly where the
raters put it, so both the amount and the direction of softening are measured.

What is deliberately NOT changed
--------------------------------
Validation stays on the hard median label, so ``avg_vaf`` means what it means in every
other search and this run is directly comparable with models/03. Fold assignment also
still uses the hard label, so stratification is unchanged. ``src/random_search.py`` scores
train accuracy against the hard label too, because the argmax of a soft target is the
modal rater score, which differs from the median for 12.75% of crops -- without that,
``avg_taf`` and ``avg_divergence`` would silently change meaning.

The comparison to run afterwards is models/03 vs this, on avg_vaf, test accuracy, and
especially near-miss and QWK: if soft targets help, the mechanism is that adjacent-score
confusions become cheaper to get slightly wrong, so the ordinal metrics should move more
than exact accuracy does.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

import base_dataset  # noqa: E402
import soft_labels  # noqa: E402
from model_builders import CNNModelBuilder  # noqa: E402
from random_search import RandomSearch  # noqa: E402

DEFAULT_DATASET = REPO_ROOT / "data" / "datasets" / "medoid_64" / "medoid_64.npy"
DEFAULT_SPLIT_CSV = REPO_ROOT / "data" / "datasets" / "medoid_64" / "medoid_64.csv"
DEFAULT_RATERS = REPO_ROOT / "data" / "combined_cda_data_median.csv"

# Byte-identical to models/03_Medoid_CNN_RandomSearch so the only difference between the
# two runs is the training target.
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


def build_aligned_soft_targets(combined, split_csv, rater_csv, num_classes):
    """Soft targets in the exact row order of ``combined['filepaths']``.

    Aligned by FILENAME rather than by assuming the manifest's row order matches the
    array order. The manifest happens to be written train-then-val, which is the order
    ``combine_train_val`` concatenates in, but relying on that would make the targets
    silently wrong if either changed.
    """
    targets, merged = soft_labels.build_soft_targets(split_csv, rater_csv, num_classes)
    by_name = {
        Path(str(fn)).name: targets[i] for i, fn in enumerate(merged["filename"].to_numpy())
    }
    names = [Path(str(p)).name for p in combined["filepaths"]]
    missing = [n for n in names if n not in by_name]
    if missing:
        raise ValueError(f"{len(missing)} training crops have no soft target, e.g. {missing[:3]}")
    aligned = np.stack([by_name[n] for n in names]).astype("float32")

    hard = np.asarray(combined["labels"]).astype(int)
    med_agree = float((aligned.argmax(1) == hard).mean())
    print(f"soft targets aligned: {aligned.shape}, rows sum to 1: "
          f"{np.allclose(aligned.sum(1), 1, atol=1e-5)}")
    print(f"  argmax(soft) == median label for {med_agree:.4f} of the CV pool")
    for k, v in soft_labels.summarise(aligned).items():
        print(f"  {k:24s} {v}")
    return aligned


def main():
    parser = argparse.ArgumentParser(description="Medoid CNN grid search on soft labels")
    parser.add_argument("-a", "--slurm_array", type=int, default=1)
    parser.add_argument("-m", "--mode", choices=["grid", "random"], default="grid")
    parser.add_argument("-n", "--n_models", type=int, default=4)
    parser.add_argument("-k", "--folds", type=int, default=5)
    parser.add_argument("-c", "--cv_seed", type=int, default=42,
                        help="keep identical to models/03 so the folds match exactly")
    parser.add_argument("-t", "--top_n", type=int, default=100)
    parser.add_argument("-d", "--dataset", type=str, default=str(DEFAULT_DATASET))
    parser.add_argument("--split_csv", type=str, default=str(DEFAULT_SPLIT_CSV))
    parser.add_argument("--raters", type=str, default=str(DEFAULT_RATERS))
    parser.add_argument("-x", "--max_configs", type=int, default=40,
                        help="exit cleanly after this many configs; 0 = unlimited")
    parser.add_argument("-r", "--max_rss_mb", type=int, default=24000,
                        help="also exit cleanly at this RSS; keep below the job's --mem")
    parser.add_argument("--hard", action="store_true",
                        help="ablation: train on one-hot medians, reproducing models/03")
    args = parser.parse_args()

    dataset = base_dataset.load_dataset(args.dataset)
    combined = base_dataset.combine_train_val(dataset)
    class_labels = sorted(np.unique(np.asarray(combined["labels"]).astype(int)).tolist())
    print(f"Dataset: {args.dataset}")
    print(f"Combined train+val size: {len(combined['labels'])}")

    if args.hard:
        soft = None
        print("\n--hard given: training on one-hot medians (models/03 behaviour)")
    else:
        soft = build_aligned_soft_targets(
            combined, args.split_csv, args.raters, len(class_labels)
        )

    search = RandomSearch(
        CNNModelBuilder(), PARAM_SPACE, k=args.folds, seed=args.slurm_array,
        cv_seed=args.cv_seed, run_id=args.slurm_array,
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
        search.run_grid(
            max_configs=args.max_configs or None,
            max_rss_mb=args.max_rss_mb or None,
            soft_targets=soft,
            **common,
        )
    else:
        search.run(n_models=args.n_models, **common)


if __name__ == "__main__":
    main()
