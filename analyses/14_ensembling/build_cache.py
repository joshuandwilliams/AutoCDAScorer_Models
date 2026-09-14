"""Score every saved model on the ensemble and test splits, and cache the softmax.

Three grids contribute saved weights: models/03 (CNN), models/04 (ViT) and models/07
(CNN trained on the geometrically augmented pool). Every model is run over BOTH
held-out splits, because the ensembling analysis builds and selects on `ensemble`
and reports on `test`; the existing caches in analyses/10 and analyses/11 hold test
only. Softmax rather than argmax, so probability-averaging rules can be compared
with voting rules.

Run once (~1 h, dominated by the ViTs):
    conda run -n agroinfiltration_scoring python build_cache.py
"""

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "pred_cache" / "probs.npz"

GRIDS = {
    "cnn": ROOT / "models/03_Medoid_CNN_RandomSearch/01_Categorical_Loss/results",
    "vit": ROOT / "models/04_Medoid_ViT_RandomSearch/01_Categorical_Loss/results",
    "geo": ROOT / "models/07_Medoid_CNN_GeomAug/01_MatchLargestClass/results",
    "cgan": ROOT / "models/08_Medoid_CNN_CGANAug/01_MatchLargestClass/results",
}


def main():
    sys.path.append(str(ROOT / "src"))
    import tensorflow as tf

    import base_dataset

    # Importing model_builders runs the @register_keras_serializable decorators on the
    # ViT's custom layers (ClassToken, AddPositionEmbedding, ExtractClassToken). Without
    # it, load_model cannot deserialise any model from models/04.
    import model_builders  # noqa: F401

    ds = base_dataset.load_dataset(str(ROOT / "data/datasets/medoid_64/medoid_64.npy"))
    Xe = np.asarray(ds["ensemble_images"], dtype="float32")
    Xt = np.asarray(ds["test_images"], dtype="float32")
    out = {
        "ye": np.asarray(ds["ensemble_labels"]).astype(int),
        "yt": np.asarray(ds["test_labels"]).astype(int),
        "files_e": np.array(
            [Path(str(f)).name for f in np.asarray(ds["ensemble_filepaths"]).tolist()]
        ),
        "files_t": np.array([Path(str(f)).name for f in np.asarray(ds["test_filepaths"]).tolist()]),
    }
    print(f"ensemble {Xe.shape}  test {Xt.shape}", flush=True)

    for tag, results in GRIDS.items():
        dirs = sorted(
            (p for p in results.iterdir() if (p / "model.keras").exists()), key=lambda p: p.name
        )
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
            f"duplicate model_id under {results}: a stale sync can leave two folders for "
            f"one config (different avg_vaf in the name, identical weights), which would "
            f"double-weight that member in every committee below."
        )
        out[f"{tag}_e"] = Pe
        out[f"{tag}_t"] = Pt
        out[f"{tag}_ids"] = np.array(ids)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **out)
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
