"""Medoid ViT grid search (categorical loss), sharded across a SLURM array.

Pointed at the IoU-medoid dataset (medoid_64), which supersedes the centre-of-overlap
base_64: both contain the same 6,364 CDAs with the same splits and the same labels, and
differ only in which of the three scorers' bounding boxes was used to crop. The medoid
box is the most-agreed-with of the three, so it is the better crop, not merely a
different one.

Modular by design: the search harness is decoupled from
  (a) the dataset      -> --dataset points at any base_dataset .npy,
  (b) the params table -> PARAM_SPACE below,
  (c) the model        -> the ModelBuilder passed to RandomSearch.

Grid changes relative to models/02_Base_ViT_RandomSearch
-------------------------------------------------------
The single-config test run (models/02_.../testrun, 5-fold CV on an A100) measured:

    config  patch  proj  depth   avg_vaf   peak GPU
    p16        16    64      4    0.5527      699 MB
    p4          4    64      4    0.5950     1359 MB
    p8          8    64      4    0.6086      712 MB
    worst       4   128      6    0.6212     1929 MB

Three things follow, and each drives one change below.

1. ``projection_dim`` is extended to 256. The best config tested sat on the old grid's
   ceiling of 128, and 256 also gives the ViT the same 8x width range over 4 values that
   the CNN sweep already had (num_filters 8..64) -- the old [32, 64, 128] was a narrower
   capacity sweep than its own CNN counterpart. ``num_transformer_layers`` stays at
   [2, 4, 6] and ``patch_size`` at [4, 8] by explicit decision.

2. Peak GPU was 1,929 MB on an 80 GB card, and host RSS peaked at 4.68 GB against 39 GB
   requested. The earlier out-of-memory failure was caused by the per-fold graph leak in
   src/random_search.py (now fixed), not by model size. ``batch_size`` therefore returns
   to 64 -- which is what this grid was originally committed with in 09e8523, before the
   OOM mitigation dropped it to 32 -- and matches the CNN sweep, keeping the two
   comparable.

``mlp_ratio`` is additionally unfrozen to [2, 4]. It was pinned at 2, but 4 is the
standard ViT feed-forward expansion, and it is a capacity axis -- leaving it fixed while
asking whether more capacity helps would beg the question.

Grid size: 2 patch x 4 proj x 2 heads x 4 depth x 3 lr x 3 dropout x 2 reg x 2 opt
           x 2 mlp_ratio = 4,608 configs, every one of them valid (each projection_dim
           is divisible by each num_heads, and both patch sizes tile 64 exactly).

That is 2.4x the old 1,944-config grid, and its configs are larger on average, so it is
a substantially bigger job -- the largest model is 6.4 M parameters against 835 k before.
run_memory_test.py phase B measures that config's cost directly, which is what should
size --time before this is submitted. If the total proves unaffordable, the cheapest
axis to drop is ``reg``: ``opt`` already contrasts Adam against AdamW, and AdamW's
decoupled weight decay covers much of what the L2 kernel regulariser is there to test,
especially with ``reg_strength`` pinned to a single value. Removing it halves the grid
to 2,304 without losing an independent question.
"""

import argparse
import sys
from pathlib import Path

# Make the repo's src/ importable (works locally and on the HPC, no .git needed).
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT / "src"))

import tensorflow as tf  # noqa: E402

import base_dataset  # noqa: E402
from model_builders import ViTModelBuilder  # noqa: E402
from random_search import RandomSearch  # noqa: E402

# Grow GPU memory on demand instead of pre-allocating the whole device, so the many
# ViT models a self-scheduling task builds don't fragment VRAM. Must run before any
# GPU op (it does -- nothing above touches the GPU).
for _gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError:
        pass

# The dataset travels with the repo (data/ is synced to the HPC), so default to the
# repo-relative copy. Override with --dataset to point elsewhere.
DEFAULT_DATASET = REPO_ROOT / "data" / "datasets" / "medoid_64" / "medoid_64.npy"

# Hyperparameter space. `k` (CV folds) is a search setting, not a model hyperparameter,
# so it lives on RandomSearch, not here. See the module docstring for why each axis has
# the values it does.
_SHARED = {
    "patch_size": [4, 8],
    # Extended to 256: 128 was the old ceiling and the best config sat on it.
    "projection_dim": [32, 64, 128, 256],
    "num_heads": [4, 8],
    "num_transformer_layers": [2, 4, 6],
    "learning_rate": [0.01, 0.001, 0.0001],
    "epochs": [50],
    # Unfrozen: 4 is the standard ViT feed-forward expansion and this is a capacity axis.
    "mlp_ratio": [2, 4],
    # relu, matching the CNN sweep. Note this axis governs less of a ViT than of a CNN:
    # the CNN applies its activation at every Conv2D, whereas here it applies only to the
    # first Dense of each MLP block -- attention and the block's second Dense stay linear.
    "activation_function": ["relu"],
    "dropout": [0.0, 0.1, 0.3],
    # Adam only. AdamW was dropped because tf.keras.optimizers.AdamW carries
    # weight_decay=0.004 by default under Keras 3.15.0 (Adam has None), which is 40x the
    # largest reg_strength swept below. Half the grid would have been silently decayed
    # through a path the reg/reg_strength columns do not record, confounding the
    # optimiser axis with the regularisation axis in any downstream analysis.
    "opt": ["Adam"],
    # Back to 64 (from the OOM-era 32): peak GPU was 1.9 GB of an 80 GB card, so the
    # earlier failure was the per-fold graph leak, not the batch. 64 also matches the
    # CNN sweep, keeping the two model classes comparable.
    "batch_size": [64],
}

# Two sub-grids rather than one dict, because _make_regularizer ignores reg_strength
# entirely when reg is None. Sweeping both strengths against reg=None would produce pairs
# of configs identical in everything the model sees, and the harness would train each
# separately. The Base CNN sweep did exactly that -- 1,920 of its 8,640 configs (22%)
# were functional duplicates, roughly 250 of its 1,121 CPU-hours spent retraining the
# same models. sklearn's ParameterGrid takes a list of dicts and unions them, so this
# costs nothing and removes 864 redundant configs here.
#
# Strengths are 1e-4 and 1e-3, the two the CNN results support. In that sweep 1e-2 put
# zero L2 models into the top 100 (against 14 at 1e-4 and 12 at 1e-3) and collapsed L1
# outright, mean avg_vaf 0.3126 versus 0.4838 at 1e-4.
PARAM_SPACE = [
    {**_SHARED, "reg": [None], "reg_strength": [0.0001]},
    {**_SHARED, "reg": ["L2"], "reg_strength": [0.0001, 0.001]},
]


def main():
    parser = argparse.ArgumentParser(description="Medoid ViT grid/random search (categorical loss)")
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
        "-d", "--dataset", type=str, default=str(DEFAULT_DATASET), help="path to a medoid_64 .npy"
    )
    parser.add_argument(
        "-x",
        "--max_configs",
        type=int,
        default=20,
        help=(
            "exit cleanly after this many configs so a queued array task takes over "
            "(TF leaks ~2.5 GB of host RSS per config; see run_search.slurm.sh). "
            "0 = unlimited."
        ),
    )
    parser.add_argument(
        "-r",
        "--max_rss_mb",
        type=int,
        default=80000,
        help=(
            "also exit cleanly if this task's RSS reaches this many MB. Backstop for "
            "--max_configs being mis-sized; keep it below the job's --mem. 0 = no ceiling."
        ),
    )
    args = parser.parse_args()

    # RandomSearch._sample_valid_configs indexes param_space by key, so it needs a single
    # dict. PARAM_SPACE here is a list of sub-grids (see above), which ParameterGrid
    # handles but the sampler does not. Fail clearly instead of raising a TypeError deep
    # in the sampler.
    if args.mode == "random" and isinstance(PARAM_SPACE, list):
        parser.error(
            "--mode random is not supported for this search: PARAM_SPACE is a list of "
            "sub-grids (used to stop reg=None being duplicated across reg_strength "
            "values). Use the default --mode grid."
        )

    # Load the prebuilt dataset; pool train+val for CV, hold out the test split.
    dataset = base_dataset.load_dataset(args.dataset)
    combined = base_dataset.combine_train_val(dataset)
    print(f"Dataset: {args.dataset}")
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
        search.run_grid(
            max_configs=args.max_configs or None,
            max_rss_mb=args.max_rss_mb or None,
            **common,
        )
    else:
        search.run(n_models=args.n_models, **common)


if __name__ == "__main__":
    main()
