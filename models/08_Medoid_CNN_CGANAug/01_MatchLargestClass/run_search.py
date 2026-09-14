"""Medoid CNN grid search on cGAN-generated crops.

Same PARAM_SPACE, folds and cv_seed as models/03_Medoid_CNN_RandomSearch, so the only
difference is the training data. Two runs share this script and differ only in
``--per_class``:

  01_MatchLargestClass  every class topped up to the size of the largest (score 0), which
                        receives no generated crops at all. Paired exactly with
                        models/07's run of the same name, so the two differ only in whether
                        the added crops are geometric transforms or GAN samples.
  02_Target5000         every class topped up to 5,000, so every class including score 0 is
                        mostly generated. Paired with models/07/02_Target5000.

Where the crops come from
-------------------------
Both sample from one pre-built pool, `data/datasets/medoid_64_cgan/medoid_64_cgan.npz`
(see analyses/01_generate_base_dataset/build_medoid_64_cgan.py), with a fixed seed, so the
smaller run's training set is a strict subset of the larger one's. The generated crops come
from config 48 of models/06_Medoid_CGAN, chosen for the reasons set out in that builder and
measured in analyses/06_cgan_quality.

What the generated crops are missing, measured rather than assumed: median lesion position,
area, circularity and internal texture match real crops, but the spread of lesion
elongation is 0.156 against 0.460, and 9.5% of generated crops carry more than 5% dark
off-leaf background against 39.5% of real ones.

CV pool
-------
The train split only, unlike models/03 which pools train and val before folding. The class
targets above are defined on the train split, and adding the val crops would break the
exact per-class sizes the two runs are built around. Test is the real, unaugmented test
split, identical to every other run.

Fold safety
-----------
Folds are grouped on `source`. Real crops group by their own index; each GAN sample is its
own group, because it is drawn from noise and so has no real image whose content could leak
across a fold boundary. The grouping machinery is shared with models/07, where it does real
work keeping a crop and its geometric copies together.

Three accuracies
----------------
`avg_vaf` stays the pooled figure so it means the same as in every other search here.
`avg_vaf_orig` and `avg_vaf_aug` report the same folds split by whether the crop is a real
photograph or an augmented copy, with `avg_taf_orig` / `avg_taf_aug` doing the same on the
training side. A model scoring far higher on the augmented half has learned the
augmentation rather than the severity, which the pooled number alone would hide.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from model_builders import CNNModelBuilder  # noqa: E402
from random_search import RandomSearch  # noqa: E402

DEFAULT_POOL = REPO_ROOT / "data/datasets/medoid_64_cgan/medoid_64_cgan.npz"
# Test comes from medoid_64.npy, NOT from the augmented pool's held-out file. Both derive
# from the same split_dataset(seed=42) call, but medoid_64 additionally downsamples every
# class to 44 test crops, whereas the full test split is unbalanced (271 of its 637 crops
# are score 0). Using the balanced one keeps test accuracy directly comparable with
# models/03 and models/05, which is the whole point of holding PARAM_SPACE and cv_seed
# fixed. It is the identical set of 308 crops those runs were scored on.
DEFAULT_TEST = REPO_ROOT / "data/datasets/medoid_64/medoid_64.npy"

# models/03_Medoid_CNN_RandomSearch's PARAM_SPACE, byte-identical. 8,640 configs.
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

MODELS_03 = REPO_ROOT / "models/03_Medoid_CNN_RandomSearch/01_Categorical_Loss"
PARAM_COLS = ["num_filters", "filter_size", "learning_rate", "epochs", "num_layers",
              "pooling_size", "activation_function", "batch_size", "reg", "reg_strength",
              "opt", "dropout"]


def paired_configs(n_configs):
    """Restrict the search to the top ``n_configs`` architectures from models/03.

    OFF by default -- ``--n_configs 0`` runs the full 8,640-config PARAM_SPACE above, which
    is byte-identical to models/03's, so these runs differ from it in the training data
    alone.

    The restricted mode exists because the comparison this run is making is about the
    DATA, not the architecture, and every config appears in models/03 as well, so pairing
    on the parameter columns gives a within-config difference rather than a comparison of
    grid maxima -- which analyses/08_accuracy_noise_floor showed is unreliable at this
    noise level. Use it if the full grid will not fit the time available.

    Cost, scaled from models/03's own recorded train_seconds (1,194 CPU-hours for the whole
    8,640-config grid on 3,052 crops): the full grid costs about 4,455 CPU-hours on the
    11,389-crop MatchLargestClass pool and 13,690 on the 35,000-crop Target5000 pool.
    Restricting to the top 100 costs 55 and 168.

    The results CSV carries the same twelve parameter columns models/03's does, so the
    pairing at analysis time is a join on those columns.
    """
    res = pd.read_csv(MODELS_03 / "random_search_results.csv")
    top = res.sort_values("avg_vaf", ascending=False).head(n_configs)
    configs, baseline = [], {}
    for _, r in top.iterrows():
        cfg = {}
        for c in PARAM_COLS:
            v = r[c]
            if c == "reg" and (isinstance(v, float) and np.isnan(v)):
                v = None
            elif c in ("num_filters", "filter_size", "epochs", "num_layers",
                       "pooling_size", "batch_size"):
                v = int(v)
            elif c in ("learning_rate", "reg_strength", "dropout"):
                v = float(v)
            cfg[c] = v
        configs.append(cfg)
        baseline[repr([cfg[k] for k in sorted(cfg)])] = float(r.avg_vaf)
    # ParameterGrid over a LIST of single-valued dicts yields exactly these configs and
    # nothing else, so run_grid's self-scheduling works unchanged.
    return [{k: [v] for k, v in c.items()} for c in configs], baseline


def subset(pool, per_class, seed, n_classes=7):
    """Deterministically take `per_class` crops of each class out of the pool.

    Real crops are taken first and in full, so a class with more real crops than the target
    is truncated rather than augmented, and a class already at or above the target receives
    no augmented crops at all. That is what makes score 0 unaugmented in the
    MatchLargestClass run. The augmented crops are then drawn without replacement using
    `seed`, so a larger target is a strict superset of a smaller one.
    """
    y, is_aug = pool["labels"].astype(int), pool["is_aug"].astype(bool)
    rng = np.random.default_rng(seed)
    keep = []
    for c in range(n_classes):
        real = np.where((y == c) & ~is_aug)[0]
        aug = np.where((y == c) & is_aug)[0]
        if len(real) >= per_class:
            keep.append(rng.permutation(real)[:per_class])
            continue
        need = per_class - len(real)
        if need > len(aug):
            raise SystemExit(f"class {c}: need {need} augmented crops, pool has {len(aug)}. "
                             f"Rebuild the pool with a larger --target.")
        # Permute the whole augmented set with the same seed and take a prefix, so the
        # selection for a smaller target nests inside the selection for a larger one.
        keep.append(np.concatenate([real, rng.permutation(aug)[:need]]))
    return np.sort(np.concatenate(keep))


def main():
    p = argparse.ArgumentParser(description="Medoid CNN grid search on augmented crops")
    p.add_argument("-a", "--slurm_array", type=int, default=1)
    p.add_argument("-m", "--mode", choices=["grid", "random"], default="grid")
    p.add_argument("-n", "--n_models", type=int, default=4)
    p.add_argument("-k", "--folds", type=int, default=5)
    p.add_argument("-c", "--cv_seed", type=int, default=42,
                   help="keep identical to models/03 so the folds match")
    p.add_argument("-t", "--top_n", type=int, default=100)
    p.add_argument("--pool", type=str, default=str(DEFAULT_POOL))
    p.add_argument("--test_from", type=str, default=str(DEFAULT_TEST))
    p.add_argument("--per_class", type=int, required=True,
                   help="training crops per class; 0 means 'match the largest class'")
    p.add_argument("--subset_seed", type=int, default=7,
                   help="fixed across both runs so the smaller set nests in the larger")
    p.add_argument("--n_configs", type=int, default=0,
                   help="0 = the full 8,640-config grid, identical to models/03. A positive "
                        "value restricts the search to that many of models/03's top configs.")
    p.add_argument("-x", "--max_configs", type=int, default=40,
                   help="exit cleanly after this many configs; 0 = unlimited")
    p.add_argument("-r", "--max_rss_mb", type=int, default=24000)
    args = p.parse_args()

    pool = np.load(args.pool, allow_pickle=True)
    y_all, is_aug_all = pool["labels"].astype(int), pool["is_aug"].astype(bool)
    real_per_class = np.bincount(y_all[~is_aug_all], minlength=7)
    per_class = args.per_class or int(real_per_class.max())
    print(f"Pool: {args.pool}")
    print(f"  real crops per class      {real_per_class.tolist()}")
    print(f"  augmented crops per class {np.bincount(y_all[is_aug_all], minlength=7).tolist()}")
    print(f"Target: {per_class} per class "
          f"({'match largest class' if not args.per_class else 'explicit'})")

    idx = subset(pool, per_class, args.subset_seed)
    X = pool["images"][idx].astype("float32") / 256.0
    y = y_all[idx]
    groups = pool["source"][idx]
    is_aug = is_aug_all[idx]
    print(f"Training pool: {len(y)} crops, per class {np.bincount(y).tolist()}")
    print(f"  real {int((~is_aug).sum())}, augmented {int(is_aug.sum())}")
    print(f"  augmented per class {np.bincount(y[is_aug], minlength=7).tolist()}")
    print(f"  distinct source crops {len(np.unique(groups))}")

    import base_dataset  # noqa: E402
    held = base_dataset.load_dataset(args.test_from)
    test_images = np.asarray(held["test_images"], dtype="float32")
    test_labels = np.asarray(held["test_labels"]).astype(int)
    print(f"Test: {len(test_labels)} real crops from {args.test_from}, "
          f"per class {np.bincount(test_labels).tolist()}")

    class_labels = sorted(np.unique(y).tolist())
    if args.n_configs:
        param_space, baseline = paired_configs(args.n_configs)
        print(f"Configs: top {len(param_space)} from models/03 by avg_vaf "
              f"(models/03 range {min(baseline.values()):.4f}-{max(baseline.values()):.4f})")
    else:
        param_space = PARAM_SPACE
        print("Configs: the full models/03 grid (8,640)")
    search = RandomSearch(CNNModelBuilder(), param_space, k=args.folds,
                          seed=args.slurm_array, cv_seed=args.cv_seed,
                          run_id=args.slurm_array)
    common = dict(train_images=X, train_labels=y, test_images=test_images,
                  test_labels=test_labels, global_dir=".", top_n=args.top_n)
    if args.mode == "grid":
        search.run_grid(max_configs=args.max_configs or None,
                        max_rss_mb=args.max_rss_mb or None,
                        groups=groups, is_aug=is_aug, **common)
    else:
        search.run(n_models=args.n_models, **common)


if __name__ == "__main__":
    main()
