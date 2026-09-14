"""Build the cGAN-augmented medoid_64 training pool.

The counterpart to `build_medoid_64_aug.py`. Identical in structure -- same real train
split, same per-class targets, same output layout -- so the only difference between
models/07 and models/08 is where the extra crops come from: geometric transforms of real
crops in one case, samples from a conditional GAN in the other.

Which generator, and why
------------------------
Config 48 from models/06_Medoid_CGAN, the lowest Frechet CDA distance of the 64 trained
(3.659). Two checks in analyses/06_cgan_quality supported using it rather than another:

  - Its recorded score is not a lucky evaluation. `best_fcd` is a minimum over 100 noisy
    evaluations and the post-3000-epoch plateau has a standard deviation of 1.21, so the
    minimum of 100 draws would sit well below the mean by chance. Re-evaluating the saved
    checkpoint on 15 fresh latent draws gives 3.727-4.142, against a plateau mean of 4.949,
    so the epoch-selection rule picked a real improvement.
  - Mixing several top generators would not add diversity. The median Frechet distance
    between pairs of the top 8 generators is 3.68, barely above the 3.07 two independent
    samples of the SAME distribution score, while each generator sits 4.10 from real
    validation data. Pooling all 8 raises within-class spread from 0.85x real to 0.86x.
    Eight generators cover the same ground as one.

What these crops are and are not
--------------------------------
Measured in analyses/06_cgan_quality against real crops: median lesion position, area,
circularity and internal texture all match, but the spread of lesion elongation is 0.156
against 0.460 and only 9.5% of generated crops carry more than 5% dark off-leaf background
against 39.5% of real ones. A classifier trained on this pool will not see a long thin
lesion or one running off the edge of a leaf.

Grouping
--------
Geometric augmentations inherit the `source` index of the real crop they came from, so the
search harness can keep them in the same CV fold. A GAN sample has no source crop -- it is
drawn from noise -- so each gets its own group id. That is the correct treatment rather
than a workaround: there is no real image whose content it could leak across a fold
boundary.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import tensorflow as tf  # noqa: E402

import base_dataset  # noqa: E402
import cgan  # noqa: E402

CROPS = REPO / "data/cropped_images_medoid"
CGAN_RUN = REPO / "models/06_Medoid_CGAN/01_Conditional"
OUT = REPO / "data/datasets/medoid_64_cgan"
DEFAULT_CONFIG = 48


def main():
    p = argparse.ArgumentParser(description="Build the cGAN-augmented medoid training pool")
    p.add_argument("--target", type=int, default=5000, help="crops per class in the pool")
    p.add_argument(
        "--config",
        type=int,
        default=DEFAULT_CONFIG,
        help="which models/06 array task's generator to sample from",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    res = pd.read_csv(CGAN_RUN / "cgan_results.csv")
    row = res.query("slurm_array == @args.config").iloc[0]
    latent = int(row.latent_dim)
    print(
        f"generator: config {args.config}, best_fcd {row.best_fcd:.3f} at epoch "
        f"{int(row.best_epoch)}, latent_dim {latent}"
    )

    # Same split call the real dataset used, and the same skipping of the class-balancing
    # downsample: the point of both augmented runs is to keep every real crop.
    images, labels, filepaths = base_dataset.load_images_and_labels(CROPS, image_size=64)
    split = base_dataset.split_dataset(
        images=images,
        labels=labels,
        filepaths=filepaths,
        train_ratio=0.6,
        val_ratio=0.15,
        ensemble_ratio=0.15,
        seed=42,
    )
    y = np.asarray(split["train_labels"]).astype(int)
    names = np.array([Path(str(f)).name for f in split["train_filepaths"]])
    print(f"full train split: {len(y)} crops, per class {np.bincount(y).tolist()}")

    need = {c: max(0, args.target - int((y == c).sum())) for c in range(7)}
    print(f"generated crops needed per class: {[need[c] for c in range(7)]}")

    gen = tf.keras.models.load_model(
        CGAN_RUN / f"array_task{args.config}/models/generator_best.keras"
    )

    # cgan.sample draws n_per_class for EVERY class at once, so ask for the largest need
    # and then take what each class actually requires. Sampling in blocks keeps memory
    # bounded; the seed advances per block so the blocks are not identical.
    biggest = max(need.values())
    BLOCK = 500
    per_class = {c: [] for c in range(7)}
    drawn = 0
    while drawn < biggest:
        n = min(BLOCK, biggest - drawn)
        imgs, labs = cgan.sample(gen, latent, n, seed=args.seed + drawn)
        for c in range(7):
            if len(per_class[c]) < need[c]:
                take = need[c] - len(per_class[c])
                per_class[c].append(imgs[labs == c][:take])
        drawn += n
        print(f"  drawn {drawn}/{biggest} per class", flush=True)

    gen_imgs, gen_lab = [], []
    for c in range(7):
        if need[c] == 0:
            continue
        blk = np.concatenate(per_class[c])[: need[c]]
        gen_imgs.append(blk)
        gen_lab.append(np.full(len(blk), c, dtype=int))
    gen_imgs = np.concatenate(gen_imgs) if gen_imgs else np.zeros((0, 64, 64, 3), "float32")
    gen_lab = np.concatenate(gen_lab) if gen_lab else np.zeros(0, int)

    def to_u8(a):
        a = np.asarray(a)
        return (a * 255).round().astype(np.uint8) if a.max() <= 1.0 else a.astype(np.uint8)

    real = to_u8(split["train_images"])
    X = np.concatenate([real, to_u8(gen_imgs)])
    Y = np.concatenate([y, gen_lab])
    # Real crops group by their own index; each GAN sample is its own group (no source).
    source = np.concatenate([np.arange(len(y)), np.arange(len(y), len(y) + len(gen_lab))])
    is_aug = np.concatenate([np.zeros(len(y), bool), np.ones(len(gen_lab), bool)])

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / "medoid_64_cgan.npz",
        images=X,
        labels=Y,
        source=source,
        is_aug=is_aug,
        source_names=names,
        target=args.target,
        seed=args.seed,
        generator_config=args.config,
        generator_fcd=float(row.best_fcd),
    )
    print(f"\npool: {len(X)} crops, per class {np.bincount(Y).tolist()}")
    print(f"  real {int((~is_aug).sum())}, generated {int(is_aug.sum())}")
    print(f"wrote {OUT / 'medoid_64_cgan.npz'}")


if __name__ == "__main__":
    main()
