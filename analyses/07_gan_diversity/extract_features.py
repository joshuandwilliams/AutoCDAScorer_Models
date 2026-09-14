"""Extract CNN features for real and GAN-generated crops, for the diversity diagnostic.

Runs on the HPC because that is where the generated images live (~/data/GAN_images_*,
10,000 per class). It does the heavy part only -- reading images and running them through
a feature extractor -- and writes a small .npz of feature vectors. All the analysis and
plotting happens locally from that file.

Why this diagnostic exists
--------------------------
Classifiers trained on GAN images previously reached ~97% on generated images while
staying at ~60% on real ones. Two different failures produce that pattern and they need
different fixes:

  (a) the generated images are LESS DIVERSE within each class than real ones, so the
      classes are trivially separable in generated data and the task is simply easier; or
  (b) the generated images are diverse but sit somewhere else entirely in feature space,
      so what the classifier learns does not transfer.

The Frechet distance cannot tell these apart -- it compares only the first two moments of
the whole distribution and says nothing about within-class structure. These features let
both be measured directly: within-class spread, between-class separation, and the distance
from each generated crop to its nearest real one (which is the memorisation check the
Frechet distance is actively blind to, since a perfectly memorising generator scores
BETTER on it than two halves of the real data do).

Feature space is the last pooling layer of the top medoid CNN -- the same space
src/cgan.py uses for the Frechet distance, so the two are directly comparable.
"""

import argparse
import glob
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT / "src"))

import cv2 as cv  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import base_dataset  # noqa: E402
import cgan  # noqa: E402

CNN_RESULTS = REPO_ROOT / "models/03_Medoid_CNN_RandomSearch/01_Categorical_Loss"
NUM_CLASSES = 7


def feature_model():
    res = pd.read_csv(CNN_RESULTS / "random_search_results.csv")
    best = int(res.sort_values("avg_vaf", ascending=False).iloc[0].model_id)
    return cgan.FrechetCDADistance(next((CNN_RESULTS / "results").glob(f"*_{best}"))
                                   / "model.keras")


def load_generated(gan_dir: Path, n_per_class: int, seed: int = 0):
    """Sample n_per_class generated crops per class, scaled like base_dataset does."""
    rng = np.random.default_rng(seed)
    X, y, names = [], [], []
    for c in range(NUM_CLASSES):
        files = sorted(glob.glob(str(gan_dir / str(c) / "*.tif")))
        if len(files) < n_per_class:
            raise SystemExit(f"class {c} has only {len(files)} images in {gan_dir}")
        for i in rng.choice(len(files), n_per_class, replace=False):
            img = cv.imread(files[i])
            if img is None:
                raise SystemExit(f"failed to read {files[i]}")
            # base_dataset divides by 256, not 255 -- match it exactly or the features
            # are computed on a different input scale from the ones the CNN was trained on.
            X.append(cv.resize(img, (64, 64)) / 256.0)
            y.append(c)
            names.append(Path(files[i]).name)
    return np.asarray(X, dtype="float32"), np.asarray(y), np.asarray(names)


def main():
    p = argparse.ArgumentParser(description="Feature extraction for the GAN diversity check")
    p.add_argument("--gan_dir", type=str, default=str(Path.home() / "data/GAN_images_2"))
    p.add_argument("--dataset", type=str,
                   default=str(REPO_ROOT / "data/datasets/medoid_64/medoid_64.npy"))
    p.add_argument("--n_per_class", type=int, default=262,
                   help="match the real train split's per-class count so spread is not a "
                        "sample-size artefact")
    p.add_argument("--out", type=str, default=str(Path(__file__).parent / "features.npz"))
    args = p.parse_args()

    fx = feature_model()
    print(f"feature space: {fx.name}, layer '{fx.layer}'")

    ds = base_dataset.load_dataset(args.dataset)
    Xr = np.asarray(ds["train_images"], dtype="float32")
    yr = np.asarray(ds["train_labels"]).astype(int)
    fr_names = np.array([Path(str(f)).name for f in ds["train_filepaths"]])
    print(f"real train crops: {Xr.shape}, per class {np.bincount(yr).tolist()}")

    Xg, yg, gnames = load_generated(Path(args.gan_dir), args.n_per_class)
    print(f"generated crops:  {Xg.shape}, per class {np.bincount(yg).tolist()}")
    print(f"  from {args.gan_dir}")

    def feats(x):
        f = fx.model.predict(x, batch_size=128, verbose=0)
        return f.reshape(len(f), -1)

    Fr, Fg = feats(Xr), feats(Xg)
    print(f"feature dimension: {Fr.shape[1]}")

    np.savez_compressed(args.out, real=Fr, real_y=yr, real_names=fr_names,
                        gen=Fg, gen_y=yg, gen_names=gnames,
                        gan_dir=str(args.gan_dir), layer=fx.layer, model=fx.name)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
