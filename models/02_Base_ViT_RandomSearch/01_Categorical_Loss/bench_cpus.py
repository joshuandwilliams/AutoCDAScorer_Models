"""Train one fixed ViT (20 epochs) and record how long the fit took.

Launched by run_benchmark.sh as a single GPU job (jic-gpu, --nv) to time one
representative config, so the whole-grid wall clock can be estimated before launching
all 1,944 configs. The intra/inter-op thread settings below are harmless on GPU (the
heavy matmuls still run on the device); they only matter if you run this on CPU. The
ViT counterpart of the CNN sweep's bench_cpus.py.
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
from model_builders import ViTModelBuilder  # noqa: E402

DATASET = REPO_ROOT / "data" / "datasets" / "base_64" / "base_64.npy"
combined = base_dataset.combine_train_val(base_dataset.load_dataset(str(DATASET)))
images = combined["images"]
labels = combined["labels"].astype(int)
num_classes = len(set(labels.tolist()))
y = tf.keras.utils.to_categorical(labels, num_classes)

# A representative mid-size ViT config, fixed epochs so the timing is comparable.
params = {
    "patch_size": 8,
    "projection_dim": 64,
    "num_heads": 8,
    "num_transformer_layers": 4,
    "mlp_ratio": 2,
    "activation_function": "gelu",
    "dropout": 0.1,
    "reg": None,
    "reg_strength": 0.0001,
    "opt": "AdamW",
    "learning_rate": 0.001,
}
model = ViTModelBuilder().build(params, images.shape[1:], num_classes)

start = time.perf_counter()
model.fit(images, y, epochs=20, batch_size=64, verbose=0)
elapsed = time.perf_counter() - start

with open(f"bench_result_{n_cpus}.csv", "w") as fh:
    fh.write("cpus,seconds\n")
    fh.write(f"{n_cpus},{elapsed:.1f}\n")
print(f"cpus={n_cpus}  seconds={elapsed:.1f}")
