"""Build the augmented medoid_64 training pool.

What this makes
---------------
`data/datasets/medoid_64_aug/medoid_64_aug.npz`: the TRAIN split of the medoid dataset,
topped up with geometric augmentations to `--target` crops per class (default 5,000). Both
planned runs sample their training sets out of this one pool with a fixed seed, so the
5,000-per-class run is a strict superset of the class-0-matched run and the two differ only
in how much augmentation they use.

Which crops go in
-----------------
The FULL train split, not the class-balanced one. `medoid_64.npy` downsamples every class
to 262 because that is the size of the smallest, which discards 1,627 of the 2,711 score-0
crops. Topping up by augmentation instead of throwing real crops away is the whole point of
this run, so the split is recomputed here with the same `split_dataset(seed=42)` call the
original dataset used, and the downsampling step is skipped. Val, ensemble and test are
untouched and unaugmented.

Fold safety
-----------
Every augmented crop records the index of the real crop it came from, in `source`. The
search harness folds on that group so a crop and its augmentations always land in the same
CV fold. Without it a validation fold would contain augmented copies of images whose
originals were used for training, and the validation accuracy would be meaningless.

`is_aug` marks which rows are augmented, so validation accuracy can be reported separately
on real and augmented crops.
"""

import argparse
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import augment  # noqa: E402
import base_dataset  # noqa: E402

CROPS = REPO / "data/cropped_images_medoid"
RAW = REPO / "data/raw_images_rotated"
MEDOID_COORDS = Path(
    "/Users/jowillia/Documents/GitHub/uv-agroinfiltration-dataset/"
    "analyses/01_generate_raw_data/04_representative_bounding_box_and_score/"
    "combined_cda_data_medoid.csv"
)
OUT = REPO / "data/datasets/medoid_64_aug"


def boxes():
    """Medoid bounding box per crop filename, from the rater whose box was the IoU medoid."""
    m = pd.read_csv(MEDOID_COORDS)
    m["stem"] = m.Basename.str.replace(r"\.(TIF|tif)$", "", regex=True)
    m["key"] = (
        m.stem
        + "_"
        + m.Row.astype(str)
        + "_"
        + m.Col.astype(str)
        + "_"
        + m.Pos.astype(str)
        + ".tif"
    )
    m = m.set_index("key")
    k = m.Medoid_Coords.astype(int)
    out = {}
    for key, kk in k.items():
        r = m.loc[key]
        vals = [r[f"X1_{kk}"], r[f"X2_{kk}"], r[f"Y1_{kk}"], r[f"Y2_{kk}"]]
        if not all(np.isfinite(v) for v in vals):
            continue
        out[key] = ([int(v) for v in vals], r.Basename)
    return out


def main():
    p = argparse.ArgumentParser(description="Build the augmented medoid training pool")
    p.add_argument("--target", type=int, default=5000, help="crops per class in the pool")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

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
    print(f"(medoid_64.npy downsamples this to {np.bincount(y).min()} per class)")

    bx = boxes()
    missing = [n for n in names if n not in bx]
    if missing:
        raise SystemExit(f"{len(missing)} train crops have no medoid box, e.g. {missing[:3]}")

    rng = np.random.default_rng(args.seed)
    n_classes = 7
    need = {c: max(0, args.target - int((y == c).sum())) for c in range(n_classes)}
    print(f"augmented crops needed per class: {[need[c] for c in range(n_classes)]}")

    # Draw the (source, params) plan first so it can be recorded, then execute it grouped
    # by raw image -- each raw photograph is ~60 MB and is read once for all its crops.
    plan = []
    for c in range(n_classes):
        idx = np.where(y == c)[0]
        if need[c] == 0:
            continue
        # Round-robin over the class's real crops so every one is augmented a near-equal
        # number of times, rather than sampling with replacement and leaving some unused.
        picks = np.tile(idx, int(np.ceil(need[c] / len(idx))))[: need[c]]
        rng.shuffle(picks)
        for src in picks:
            box, base = bx[names[src]]
            params, rec = augment.random_params(rng, box)
            plan.append((int(src), int(c), base, box, params, rec))
    print(f"planned {len(plan)} augmented crops")

    aug = np.zeros((len(plan), 64, 64, 3), dtype=np.uint8)
    order = np.argsort([p[2] for p in plan], kind="stable")
    cur_base, raw = None, None
    for n, pi in enumerate(order):
        _, _, base, box, params, _ = plan[pi]
        if base != cur_base:
            raw = cv.imread(str(RAW / base))
            if raw is None:
                raise SystemExit(f"missing raw image {base}")
            cur_base = base
        aug[pi] = augment.crop(raw, box, **params)
        if n % 2000 == 0:
            print(f"  {n}/{len(plan)}", flush=True)

    real = (
        (np.asarray(split["train_images"]) * 255).round().astype(np.uint8)
        if np.asarray(split["train_images"]).max() <= 1.0
        else np.asarray(split["train_images"]).astype(np.uint8)
    )

    X = np.concatenate([real, aug])
    Y = np.concatenate([y, np.array([p[1] for p in plan], dtype=int)])
    source = np.concatenate([np.arange(len(y)), np.array([p[0] for p in plan], dtype=int)])
    is_aug = np.concatenate([np.zeros(len(y), bool), np.ones(len(plan), bool)])

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / "medoid_64_aug.npz",
        images=X,
        labels=Y,
        source=source,
        is_aug=is_aug,
        source_names=names,
        target=args.target,
        seed=args.seed,
    )
    pd.DataFrame(
        [{"source_index": p[0], "label": p[1], "source_name": names[p[0]], **p[5]} for p in plan]
    ).to_csv(OUT / "augmentation_log.csv", index=False)

    # Val / ensemble / test, real only, saved in the same file so a run needs one input.
    extra = {}
    for s in ("val", "ensemble", "test"):
        xi = np.asarray(split[f"{s}_images"])
        xi = (xi * 255).round().astype(np.uint8) if xi.max() <= 1.0 else xi.astype(np.uint8)
        extra[f"{s}_images"] = xi
        extra[f"{s}_labels"] = np.asarray(split[f"{s}_labels"]).astype(int)
    np.savez_compressed(OUT / "medoid_64_aug_heldout.npz", **extra)

    print(f"\npool: {len(X)} crops, per class {np.bincount(Y).tolist()}")
    print(f"  real {int((~is_aug).sum())}, augmented {int(is_aug.sum())}")
    print(f"wrote {OUT / 'medoid_64_aug.npz'}")
    for s in ("val", "ensemble", "test"):
        print(
            f"  {s}: {len(extra[f'{s}_labels'])} "
            f"per class {np.bincount(extra[f'{s}_labels']).tolist()}"
        )


if __name__ == "__main__":
    main()
