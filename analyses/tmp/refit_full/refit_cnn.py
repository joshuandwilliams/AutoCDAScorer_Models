"""Refit the top medoid CNN on train+val and score it on test.

models/03 reports test accuracy for the weights of the BEST CV FOLD, which were trained on
4/5 of the pooled train+val set (about 1,831 crops). The feature classifiers in
analyses/13 are refitted on all 2,289 before being tested, so the comparison gives them
more training data. This measures what the CNN does under the feature models' procedure.

No validation set is held out here, so early stopping cannot run. Two epoch counts are
tried instead: the best_epoch models/03 recorded for this config (20), and the full 50 the
grid specified. Five seeds each, because a single run says nothing against a test-set
standard error of about 0.027 on 308 crops.
"""
import sys, glob, time
from pathlib import Path
import numpy as np, pandas as pd, tensorflow as tf

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
import base_dataset                                    # noqa: E402
from model_builders import CNNModelBuilder             # noqa: E402
from metrics import near_miss_accuracy, quadratic_weighted_kappa  # noqa: E402

res = pd.read_csv(glob.glob(str(REPO / "models/03_Medoid_CNN_RandomSearch/*/random_search_results.csv"))[0])
b = res.sort_values("avg_vaf", ascending=False).iloc[0]
params = {"num_filters": int(b.num_filters), "filter_size": int(b.filter_size),
          "learning_rate": float(b.learning_rate), "epochs": int(b.epochs),
          "num_layers": int(b.num_layers), "pooling_size": int(b.pooling_size),
          "activation_function": b.activation_function, "batch_size": int(b.batch_size),
          "reg": None if pd.isna(b.reg) else b.reg, "reg_strength": float(b.reg_strength),
          "opt": b.opt, "dropout": float(b.dropout)}
print(f"config {int(b.model_id)}: {params}")
print(f"models/03 reports avg_vaf {b.avg_vaf:.4f}, test_acc {b.test_acc:.4f}, "
      f"best_epoch {int(b.best_epoch)}")

ds = base_dataset.load_dataset(REPO / "data/datasets/medoid_64/medoid_64.npy")
comb = base_dataset.combine_train_val(ds)
X = np.asarray(comb["images"], dtype="float32")
y = np.asarray(comb["labels"]).astype(int)
Xte = np.asarray(ds["test_images"], dtype="float32")
yte = np.asarray(ds["test_labels"]).astype(int)
print(f"train+val {len(y)} crops, test {len(yte)}")
Y = tf.keras.utils.to_categorical(y, 7)

rows = []
for epochs in (int(b.best_epoch), int(b.epochs)):
    for seed in range(5):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        t0 = time.perf_counter()
        m = CNNModelBuilder().build({**params, "epochs": epochs}, X.shape[1:], 7)
        m.fit(X, Y, epochs=epochs, batch_size=params["batch_size"], verbose=0)
        p = np.argmax(m.predict(Xte, verbose=0), axis=1)
        rows.append({"epochs": epochs, "seed": seed,
                     "test_acc": float(np.mean(p == yte)),
                     "test_nearmiss": near_miss_accuracy(yte, p),
                     "test_qwk": quadratic_weighted_kappa(yte, p, labels=list(range(7))),
                     "train_acc": float(np.mean(np.argmax(m.predict(X, verbose=0), 1) == y)),
                     "seconds": round(time.perf_counter() - t0, 1)})
        print(f"  epochs={epochs} seed={seed}: test {rows[-1]['test_acc']:.4f} "
              f"({rows[-1]['seconds']}s)", flush=True)
        del m

R = pd.DataFrame(rows)
R.to_csv(Path(__file__).parent / "refit_cnn_results.csv", index=False)
print("\n=== summary ===")
print(R.groupby("epochs")[["test_acc", "test_nearmiss", "test_qwk", "train_acc"]]
      .agg(["mean", "std", "min", "max"]).round(4).to_string())
print(f"\nmodels/03 best-fold test_acc (trained on ~1,831 crops): {b.test_acc:.4f}")
