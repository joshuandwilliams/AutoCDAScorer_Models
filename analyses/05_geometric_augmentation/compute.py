"""Measure what each geometric augmentation does to a CDA crop.

Two quantities per augmentation, both needed to decide whether it is worth applying.

1. DISPLACEMENT -- how far the crop moves in the twelve Chapter 2 features, standardised
   on the real crops and expressed in units of the real within-class spread. This says
   whether the augmentation produces something the measurable properties of the image can
   tell apart from the original at all. A displacement near zero means the augmented copy
   is, in these terms, the same image again.

2. LABEL PRESERVATION -- whether the top medoid CNN still predicts the same score for the
   augmented crop as for the original. This says whether the augmentation left the
   severity content alone. An augmentation that moves the crop a long way AND changes the
   prediction is not teaching invariance, it is relabelling the data.

The two together separate three cases:
    displacement ~ 0                      -> nothing to learn from it
    displacement large, label preserved   -> a real nuisance direction, worth training on
    displacement large, label changed     -> corrupts the training label

Crops are rebuilt from the raw leaf photographs via `augment.crop`, whose identity case
reproduces the stored medoid crops exactly, so every difference measured here comes from
the augmentation and not from resampling.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import cv2 as cv

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analyses" / "07_gan_diversity"))

# This analysis has its own augment.py (the single-augmentation sweep, with AUGMENTATIONS);
# src/ has a different augment.py (the policy that builds the datasets). Both are on the
# path, and src/ is inserted last so it would shadow the local one. Load this folder's copy
# by path so which module is meant is not left to import order.
import importlib.util as _ilu  # noqa: E402
_spec = _ilu.spec_from_file_location("sweep_augment", HERE / "augment.py")
augment = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(augment)
import chapter2_features as c2  # noqa: E402

MEDOID_COORDS = Path("/Users/jowillia/Documents/GitHub/uv-agroinfiltration-dataset/"
                     "analyses/01_generate_raw_data/04_representative_bounding_box_and_score/"
                     "combined_cda_data_medoid.csv")
CNN_RESULTS = ROOT / "models/03_Medoid_CNN_RandomSearch/01_Categorical_Loss"

# The original grid, copied verbatim from
# from_HPC/models/Model_0029_Aug_Sensitivity/random_search_array_sample.py.
ORIGINAL_GRID = {
    "shift_x": [20, -20, 0],
    "shift_y": [20, -20, 0],
    "zoom": [1.2, 0.8, 1],
    "rotation": [150, 300, 0],
    "shear_x": [0.3, -0.3, 0],
    "shear_y": [0.3, -0.3, 0],
    "flip_x": [True, False],
    "flip_y": [True, False],
    "brightness": [1.2, 0.8, 1],
}
IDENTITY = {"shift_x": 0, "shift_y": 0, "zoom": 1, "rotation": 0, "shear_x": 0,
            "shear_y": 0, "flip_x": False, "flip_y": False, "brightness": 1}


def draw_original(rng):
    """One draw from the original grid, rejecting only the all-identity combination.

    That rejection is the only filter the original applied
    (`aug_grid_filtered` in random_search_array_sample.py), so a single augmented image
    routinely received a shift, a zoom, a rotation, two shears, two flips and a brightness
    change at once.
    """
    while True:
        p = {k: v[rng.integers(len(v))] for k, v in ORIGINAL_GRID.items()}
        if p != IDENTITY:
            return p


def n_active(p):
    return sum(1 for k, v in p.items() if v != IDENTITY[k])


def load_index():
    med = pd.read_csv(MEDOID_COORDS)
    med["stem"] = med.Basename.str.replace(r"\.(TIF|tif)$", "", regex=True)
    med["key"] = (med.stem + "_" + med.Row.astype(str) + "_" + med.Col.astype(str)
                  + "_" + med.Pos.astype(str) + ".tif")
    med = med.set_index("key")
    ds = pd.read_csv(ROOT / "data/datasets/medoid_64/medoid_64.csv")
    tr = ds[ds.split == "train"].copy()
    assert tr.filename.isin(med.index).all(), "some train crops have no medoid box"
    k = med.loc[tr.filename, "Medoid_Coords"].astype(int).values
    for c, col in zip("XXYY", ["X1", "X2", "Y1", "Y2"]):
        tr[col] = [med.loc[f, f"{col}_{kk}"] for f, kk in zip(tr.filename, k)]
    tr["Basename"] = med.loc[tr.filename, "Basename"].values
    return tr


def cnn():
    import tensorflow as tf
    res = pd.read_csv(CNN_RESULTS / "random_search_results.csv")
    best = int(res.sort_values("avg_vaf", ascending=False).iloc[0].model_id)
    path = next((CNN_RESULTS / "results").glob(f"*_{best}")) / "model.keras"
    return tf.keras.models.load_model(path), path.parent.name


def main():
    rng = np.random.default_rng(0)
    tr = load_index()
    print(f"{len(tr)} train crops over {tr.Basename.nunique()} raw images")

    variants = dict(augment.AUGMENTATIONS)
    names = list(variants) + ["orig_composite"]

    # One composite draw per crop, fixed up front so it is recorded alongside the result.
    composites = [draw_original(rng) for _ in range(len(tr))]
    print(f"original grid: mean {np.mean([n_active(p) for p in composites]):.2f} "
          f"augmentations active per image")

    crops = {n: np.zeros((len(tr), 64, 64, 3), dtype=np.uint8) for n in names}
    oof = {n: np.zeros(len(tr)) for n in names}

    for base, grp in tr.groupby("Basename", sort=False):
        raw = cv.imread(str(ROOT / "data/raw_images_rotated" / base))
        if raw is None:
            raise SystemExit(f"missing raw image {base}")
        for i, r in zip(grp.index, grp.itertuples()):
            pos = tr.index.get_loc(i)
            box = [int(r.X1), int(r.X2), int(r.Y1), int(r.Y2)]
            # src/augment.py measures the box side as the mean of its width and height,
            # so the proportional-shift variants use the same definition.
            side = ((box[1] - box[0]) + (box[3] - box[2])) / 2
            for n in names:
                kw = composites[pos] if n == "orig_composite" else dict(variants[n])
                for _ax in ("x", "y"):
                    _f = kw.pop(f"shift_{_ax}_frac", None)
                    if _f is not None:
                        kw[f"shift_{_ax}"] = float(_f * side)
                crops[n][pos] = augment.crop(raw, box, **kw)
                oof[n][pos] = augment.out_of_frame(raw.shape, box, **kw)
        print(f"  {base}: {len(grp)} crops", flush=True)

    model, model_name = cnn()
    print(f"CNN for the label-preservation test: {model_name}")

    feats, preds = {}, {}
    for n in names:
        x = crops[n].astype("float32") / 256.0
        feats[n] = c2.feature_table(x)
        preds[n] = model.predict(x, batch_size=256, verbose=0).argmax(1)
        print(f"  measured {n}", flush=True)

    np.savez_compressed(
        HERE / "augmentation_effects.npz",
        names=np.array(names), labels=tr.label.values,
        filenames=tr.filename.values, model=model_name,
        feature_names=np.array(c2.FEATURE_NAMES),
        n_active=np.array([n_active(p) for p in composites]),
        **{f"feat__{n}": feats[n] for n in names},
        **{f"pred__{n}": preds[n] for n in names},
        **{f"oof__{n}": oof[n] for n in names},
        **{f"img__{n}": crops[n][::200] for n in names},
    )
    print(f"wrote {HERE / 'augmentation_effects.npz'}")


if __name__ == "__main__":
    main()
