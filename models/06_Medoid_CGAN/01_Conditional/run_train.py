"""Train one conditional GAN config on the medoid_64 TRAIN split.

One SLURM array task per config, as the original did: GAN training is a single long
adversarial run, not k-fold CV, so the self-scheduling RandomSearch harness does not
apply.

The grid is six binary axes = 64 configs, five of them adversarial-optimisation controls
and one capacity control. See PARAM_SPACE for why the split is that way round.

THE ONE THING THAT MATTERS MOST: this trains on ``train_images`` only. The original split
the data itself and, measured against medoid_64, 161 of the 308 held-out test crops
(52.3%) ended up in its GAN training set. Generated images therefore carried test-set
content, and every downstream classifier trained on them was contaminated. Nothing here
touches val, ensemble or test.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402
from sklearn.model_selection import ParameterGrid  # noqa: E402

import base_dataset  # noqa: E402
import cgan  # noqa: E402

for _gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError:
        pass

DEFAULT_DATASET = REPO_ROOT / "data" / "datasets" / "medoid_64" / "medoid_64.npy"
CNN_RESULTS = REPO_ROOT / "models/03_Medoid_CNN_RandomSearch/01_Categorical_Loss"

# WHAT THIS GRID SEARCHES, AND WHY IT IS MOSTLY OPTIMISATION
#
# Six binary axes = 64 configs. Five control adversarial optimisation and one controls
# capacity. That split is deliberate.
#
# A GAN has no validation loss to descend, so the only thing that can be tuned against a
# quality measure is the training dynamics. With 1,834 64x64 crops the binding constraint
# is not model size -- it is whether the two networks stay in a useful equilibrium for
# long enough to matter. Every axis below except generator_filters is a documented control
# on that equilibrium:
#
#   beta_1 0.5        Radford et al. 2016 (DCGAN, arXiv:1511.06434v2) sec. 4 "Details of
#                     adversarial training": "we found leaving the momentum term beta_1 at
#                     the suggested value of 0.9 resulted in training oscillation and
#                     instability while reducing it to 0.5 helped stabilize training".
#                     Keras defaults to 0.9, so this axis tests their finding directly.
#   lr_ratio_d        Two time-scale update rule, Heusel et al. 2017 (arXiv:1706.08500),
#                     abstract: "TTUR has an individual learning rate for both the
#                     discriminator and the generator", and "we prove that the TTUR
#                     converges under mild assumptions to a stationary local Nash
#                     equilibrium". The paper does not claim a shared rate fails to
#                     converge, only that TTUR is proved to.
#   label_smoothing   One-sided label smoothing, Salimans et al. 2016 (arXiv:1606.03498)
#                     sec. 3.4. Their stated reason for smoothing ONLY the positive
#                     target: with positive target alpha and negative target beta the
#                     optimal discriminator is (alpha*p_data + beta*p_model) /
#                     (p_data + p_model), and "the presence of p_model in the numerator is
#                     problematic because, in areas where p_data is approximately zero and
#                     p_model is large, erroneous samples from p_model have no incentive to
#                     move nearer to the data".
#   disc_dropout      Direct control on discriminator strength. No citation: this is a
#                     standard regularisation knob, not a result from a specific paper.
#   learning_rate     The scale everything else is relative to.
#
# generator_filters is the one capacity axis, kept so the search can say whether width
# mattered rather than assuming it did not.
#
# Fixed, with reasons:
#   latent_dim 100    The DCGAN value for exactly this output size -- arXiv:1511.06434v2
#                     Figure 1 describes "a 100 dimensional uniform distribution Z" driving
#                     four fractionally-strided convolutions to "a 64 x 64 pixel image".
#                     These crops are 64x64.
#   disc_filters      Held at the narrower stack so discriminator strength is varied
#                     through dropout and learning rate, which act faster and are cheaper
#                     to interpret than width.
#   batch_size 64     Halves the optimiser steps per epoch relative to 32, which is the
#                     single biggest lever on wall-clock cost, and this run is
#                     time-limited.
PARAM_SPACE = {
    # --- capacity ----------------------------------------------------------------------
    "generator_filters": [[512, 256, 128, 64], [256, 128, 64, 32]],
    # --- adversarial optimisation ------------------------------------------------------
    "learning_rate": [1e-4, 1e-5],
    "beta_1": [0.5, 0.9],
    "lr_ratio_d": [1.0, 4.0],
    "disc_dropout": [0.3, 0.5],
    "label_smoothing": [0.0, 0.1],
    # --- fixed -------------------------------------------------------------------------
    "latent_dim": [100],
    "discriminator_filters": [[64, 128, 256]],
    "batch_size": [64],
}


def feature_model_path() -> Path:
    """Top medoid CNN by avg_vaf, used as the feature space for the Frechet distance."""
    res = pd.read_csv(CNN_RESULTS / "random_search_results.csv")
    best = int(res.sort_values("avg_vaf", ascending=False).iloc[0].model_id)
    return next((CNN_RESULTS / "results").glob(f"*_{best}")) / "model.keras"


def main():
    p = argparse.ArgumentParser(description="Train one conditional GAN config")
    p.add_argument("-a", "--slurm_array", type=int, default=1,
                   help="1-based index into the 64-config grid")
    p.add_argument("-e", "--epochs", type=int, default=5000)
    p.add_argument("-d", "--dataset", type=str, default=str(DEFAULT_DATASET))
    p.add_argument("--fcd_every", type=int, default=50)
    p.add_argument("--sample_every", type=int, default=250)
    p.add_argument("--max_hours", type=float, default=3.0,
                   help="wall-clock budget per config. Early stopping only fires while "
                        "the discriminator accuracy is inside its window, so a config "
                        "whose discriminator has won runs every epoch; this bounds the "
                        "cost of the whole grid.")
    p.add_argument("--no_fcd", action="store_true",
                   help="skip the Frechet distance (fast smoke tests only)")
    args = p.parse_args()

    grid = list(ParameterGrid(PARAM_SPACE))
    if not 1 <= args.slurm_array <= len(grid):
        raise SystemExit(f"--slurm_array must be 1..{len(grid)}; got {args.slurm_array}")
    params = grid[args.slurm_array - 1]
    print(f"config {args.slurm_array}/{len(grid)}: {params}")

    ds = base_dataset.load_dataset(args.dataset)
    train_images = np.asarray(ds["train_images"], dtype="float32")
    train_labels = np.asarray(ds["train_labels"]).astype(int)
    # The Frechet distance is measured against the VAL split, as gans.py main() did, so
    # it scores generalisation rather than memorisation of the training crops.
    val_images = np.asarray(ds["val_images"], dtype="float32")
    print(f"Dataset: {args.dataset}")
    print(f"GAN training set: {len(train_labels)} crops (TRAIN SPLIT ONLY)")
    print(f"  per-class counts: {np.bincount(train_labels, minlength=7).tolist()}")

    # Fail loudly rather than silently training on more than the train split.
    n_expected = len(ds["train_labels"])
    assert len(train_images) == n_expected, "train split size changed unexpectedly"

    fcd = None if args.no_fcd else cgan.FrechetCDADistance(feature_model_path())
    if fcd is not None:
        print(f"Frechet distance feature space: medoid CNN {fcd.name}, layer "
              f"'{fcd.layer}' (last pooling layer, matching gans.py extract_fid_features;"
              f" NOT InceptionV3, so not comparable with published FID)")
        print(f"  reference activations from the VAL split ({len(val_images)} crops)")

    out = Path(f"array_task{args.slurm_array}")
    history, best = cgan.train_cgan(
        train_images, train_labels, params, epochs=args.epochs, out_dir=out,
        val_images=val_images, fcd=fcd, fcd_every=args.fcd_every,
        sample_every=args.sample_every,
        max_seconds=(args.max_hours * 3600 if args.max_hours > 0 else None),
    )

    pd.DataFrame(history).to_csv(out / f"history_{args.slurm_array}.csv", index=False)
    row = {**{k: (str(v) if isinstance(v, list) else v) for k, v in params.items()},
           "slurm_array": args.slurm_array,
           "epochs": args.epochs,
           "best_fcd": best["fcd"],
           "best_epoch": best["epoch"],
           "epochs_run": best.get("epochs_run"),
           "early_stopped_at": best.get("stopped_at"),
           "mean_epoch_seconds": float(pd.DataFrame(history).epoch_seconds.mean()),
           "final_gen_loss": history[-1]["gen_loss"],
           "final_disc_loss": history[-1]["disc_loss"]}
    pd.DataFrame([row]).to_csv(out / f"results_{args.slurm_array}.csv", index=False)
    (out / "params.json").write_text(json.dumps(row, indent=2, default=str))
    h = pd.DataFrame(history)
    print(f"\nbest Frechet CDA distance {best['fcd']:.3f} at epoch {best['epoch']}")
    print(f"epochs run: {best.get('epochs_run')} of {args.epochs}"
          + (f" (early stopped)" if best.get("stopped_at") else ""))
    print(f"mean epoch time: {h.epoch_seconds.mean():.2f} s")


if __name__ == "__main__":
    main()
