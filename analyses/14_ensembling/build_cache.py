"""Score every saved model on the ensemble and test splits, and cache the softmax.

Five grids contribute saved weights: models/03 (CNN), models/04 (ViT), models/05 (CNN
trained on soft rater-distribution labels), models/07 (CNN trained on the geometrically
augmented pool) and models/08 (CNN trained on the cGAN-augmented pool). Every model is run
over BOTH held-out splits, because the ensembling analysis builds and selects on `ensemble`
and reports on `test`; the existing caches in analyses/10 and analyses/11 hold test only.
Softmax rather than argmax, so probability-averaging rules can be compared with voting
rules.

**One cache file per grid.** Each grid writes `pred_cache/probs_<tag>.npz` carrying a
fingerprint of the model folders it was built from, and `probs.npz` is assembled from those
without any inference. Syncing more weights for one grid then costs only that grid: before
this split, adding 83 CNNs to models/07 meant re-scoring all 400 models, and the 100 ViTs
dominate the runtime.

    conda run -n autocdascorer-analyses python build_cache.py          # only what changed
    conda run -n autocdascorer-analyses python build_cache.py geo      # one grid
    conda run -n autocdascorer-analyses python build_cache.py --all    # force everything
"""

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = HERE / "pred_cache"
OUT = CACHE / "probs.npz"

GRIDS = {
    "cnn": ROOT / "models/03_Medoid_CNN_RandomSearch/01_Categorical_Loss/results",
    "vit": ROOT / "models/04_Medoid_ViT_RandomSearch/01_Categorical_Loss/results",
    "soft": ROOT / "models/05_Medoid_CNN_SoftLabels/01_Categorical_Loss/results",
    "geo": ROOT / "models/07_Medoid_CNN_GeomAug/01_MatchLargestClass/results",
    "cgan": ROOT / "models/08_Medoid_CNN_CGANAug/01_MatchLargestClass/results",
}


def model_dirs(results: Path) -> list[Path]:
    if not results.exists():
        return []
    return sorted((p for p in results.iterdir() if (p / "model.keras").exists()),
                  key=lambda p: p.name)


def fingerprint(dirs: list[Path]) -> str:
    """The folder names encode both the config id and its avg_vaf, so a re-synced or
    re-scored grid changes this string and the cache for that grid alone is rebuilt."""
    return "|".join(d.name for d in dirs)


def score_grid(tag: str, dirs: list[Path], Xe, Xt) -> dict:
    import tensorflow as tf

    Pe = np.zeros((len(dirs), len(Xe), 7), np.float32)
    Pt = np.zeros((len(dirs), len(Xt), 7), np.float32)
    ids = []
    t0 = time.time()
    for i, d in enumerate(dirs):
        m = tf.keras.models.load_model(d / "model.keras", compile=False)
        Pe[i] = m.predict(Xe, verbose=0)
        Pt[i] = m.predict(Xt, verbose=0)
        tf.keras.backend.clear_session()
        ids.append(int(d.name.split("_", 1)[1]))
        if (i + 1) % 10 == 0 or i + 1 == len(dirs):
            print(f"  {tag} {i + 1}/{len(dirs)}  {time.time() - t0:.0f}s", flush=True)
    assert len(set(ids)) == len(ids), (
        f"duplicate model_id under {GRIDS[tag]}: a stale sync can leave two folders for "
        f"one config (different avg_vaf in the name, identical weights), which would "
        f"double-weight that member in every committee below."
    )
    return {f"{tag}_e": Pe, f"{tag}_t": Pt, f"{tag}_ids": np.array(ids)}


def main(argv: list[str]) -> None:
    sys.path.append(str(ROOT / "src"))
    import base_dataset

    # Importing model_builders runs the @register_keras_serializable decorators on the
    # ViT's custom layers (ClassToken, AddPositionEmbedding, ExtractClassToken). Without
    # it, load_model cannot deserialise any model from models/04.
    import model_builders  # noqa: F401

    force_all = "--all" in argv
    named = [a for a in argv if not a.startswith("-")]
    wanted = named or list(GRIDS)
    unknown = set(wanted) - set(GRIDS)
    if unknown:
        raise SystemExit(f"unknown grid(s) {sorted(unknown)}; choose from {list(GRIDS)}")

    ds = base_dataset.load_dataset(str(ROOT / "data/datasets/medoid_64/medoid_64.npy"))
    Xe = np.asarray(ds["ensemble_images"], dtype="float32")
    Xt = np.asarray(ds["test_images"], dtype="float32")
    shared = {
        "ye": np.asarray(ds["ensemble_labels"]).astype(int),
        "yt": np.asarray(ds["test_labels"]).astype(int),
        "files_e": np.array(
            [Path(str(f)).name for f in np.asarray(ds["ensemble_filepaths"]).tolist()]
        ),
        "files_t": np.array([Path(str(f)).name for f in np.asarray(ds["test_filepaths"]).tolist()]),
    }
    print(f"ensemble {Xe.shape}  test {Xt.shape}", flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    for tag in GRIDS:
        part = CACHE / f"probs_{tag}.npz"
        dirs = model_dirs(GRIDS[tag])
        if not dirs:
            print(f"{tag}: no saved models, skipped", flush=True)
            continue
        fp = fingerprint(dirs)
        if tag in wanted and not force_all and part.exists():
            if str(np.load(part, allow_pickle=True)["fingerprint"]) == fp:
                print(f"{tag}: {len(dirs)} models unchanged, cache kept", flush=True)
                continue
        if tag not in wanted:
            print(f"{tag}: not requested, cache kept", flush=True)
            continue
        print(f"{tag}: scoring {len(dirs)} models", flush=True)
        np.savez_compressed(part, fingerprint=fp, **score_grid(tag, dirs, Xe, Xt))

    bundle = dict(shared)
    for tag in GRIDS:
        part = CACHE / f"probs_{tag}.npz"
        if not part.exists():
            continue
        z = np.load(part, allow_pickle=True)
        for k in z.files:
            if k != "fingerprint":
                bundle[k] = z[k]
    np.savez_compressed(OUT, **bundle)
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
