"""Train one fixed model and record how long it took at the allocated CPU count.

Launched by run_benchmark.sh at several --cpus-per-task values; compare the
resulting bench_result_*.csv files to see whether more cores actually help.
"""

import os
import sys
import time
from pathlib import Path

n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
os.environ["OMP_NUM_THREADS"] = str(n_cpus)

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT / "src"))

import tensorflow as tf  # noqa: E402

tf.config.threading.set_intra_op_parallelism_threads(n_cpus)
tf.config.threading.set_inter_op_parallelism_threads(n_cpus)

import base_dataset  # noqa: E402
from model_builders import CNNModelBuilder  # noqa: E402

DATASET = REPO_ROOT / "data" / "datasets" / "base_64" / "base_64.npy"
combined = base_dataset.combine_train_val(base_dataset.load_dataset(str(DATASET)))
images = combined["images"]
labels = combined["labels"].astype(int)
num_classes = len(set(labels.tolist()))
y = tf.keras.utils.to_categorical(labels, num_classes)

# A representative mid-size config, fixed epochs so the timing is comparable.
params = {
    "num_filters": 32,
    "filter_size": 3,
    "num_layers": 3,
    "pooling_size": 2,
    "activation_function": "relu",
    "dropout": 0.0,
    "reg": None,
    "reg_strength": 0.001,
    "opt": "Adam",
    "learning_rate": 0.001,
}
model = CNNModelBuilder().build(params, images.shape[1:], num_classes)

start = time.perf_counter()
model.fit(images, y, epochs=20, batch_size=64, verbose=0)
elapsed = time.perf_counter() - start

with open(f"bench_result_{n_cpus}.csv", "w") as fh:
    fh.write("cpus,seconds\n")
    fh.write(f"{n_cpus},{elapsed:.1f}\n")
print(f"cpus={n_cpus}  seconds={elapsed:.1f}")
