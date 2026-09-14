"""Train ONE pinned ViT config through the real search harness, and time it.

Why this exists
---------------
The full grid (1,944 configs) has never run: an earlier attempt exhausted GPU memory
on the largest configs. Three mitigations were made (per-fold graph teardown in
``src/random_search.py``, batch 64 -> 32, and more --mem / fewer concurrent tasks), but
none has been exercised on a GPU. This script runs a single config end-to-end through
the SAME code path the grid uses -- ``RandomSearch.run_grid`` with ``ViTModelBuilder``,
5-fold CV, the real early-stopping rule, artefact saving and CSV write -- so that a
success here means the grid path works, not merely that a model can be built.

It also records wall clock and peak GPU memory, which is the input to sizing the full
grid (see ``--summarise``).

Config choice
-------------
``p4`` / ``p8`` / ``p16`` hold the architecture fixed (projection 64, 8 heads, depth 4)
and vary only ``patch_size``. Patch size sets the token count (256 / 64 / 16 on 64x64
crops) and attention is quadratic in tokens, so these three isolate the dominant cost
term. Exactly 648 of the 1,944 grid configs use each patch size, which makes the grid
estimate a straight weighted sum.

``worst`` is the config that actually ran out of memory: patch 4 (256 tokens),
projection 128, depth 6. It is the OOM check and the upper bound on per-config cost.

Usage
-----
    python3.12 run_single_test.py --config p8
    python3.12 run_single_test.py --summarise      # after the four runs
"""

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT / "src"))

import tensorflow as tf  # noqa: E402

# Same memory-growth setting as run_search.py, for the same reason: pre-allocating the
# whole device fragments VRAM across the many models one task builds. Must precede any
# GPU op.
for _gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError:
        pass

import base_dataset  # noqa: E402
from model_builders import ViTModelBuilder  # noqa: E402
from random_search import RandomSearch  # noqa: E402

DEFAULT_DATASET = REPO_ROOT / "data" / "datasets" / "base_64" / "base_64.npy"
TESTRUN_DIR = Path(__file__).resolve().parent / "testrun"

# Held fixed across every test config, so only the architecture varies. These values are
# all drawn from the real PARAM_SPACE in run_search.py.
_FIXED = {
    "learning_rate": [0.001],
    "epochs": [50],
    "mlp_ratio": [2],
    "activation_function": ["gelu"],
    "dropout": [0.1],
    "reg": [None],
    "reg_strength": [0.0001],
    "opt": ["AdamW"],
    "batch_size": [32],
}

# patch_size -> tokens on a 64x64 crop is (64/p)^2, so 4 -> 256, 8 -> 64, 16 -> 16.
CONFIGS = {
    "p4": {"patch_size": [4], "projection_dim": [64], "num_heads": [8],
           "num_transformer_layers": [4]},
    "p8": {"patch_size": [8], "projection_dim": [64], "num_heads": [8],
           "num_transformer_layers": [4]},
    "p16": {"patch_size": [16], "projection_dim": [64], "num_heads": [8],
            "num_transformer_layers": [4]},
    # The config that OOM'd: most tokens, widest projection, deepest stack.
    "worst": {"patch_size": [4], "projection_dim": [128], "num_heads": [8],
              "num_transformer_layers": [6]},
}

# Each patch_size accounts for exactly a third of the grid: the other seven varying
# hyperparameters are independent of it (3 proj x 2 heads x 3 depth x 3 lr x 3 dropout
# x 2 reg x 2 opt = 648), and every combination is valid on 64x64 crops.
CONFIGS_PER_PATCH_SIZE = 648
GRID_TOTAL = 1944


def _require_gpu() -> None:
    """Fail loudly if the GPU is not visible.

    Without ``--nv`` Singularity does not bind the host driver and TensorFlow silently
    falls back to CPU. That is exactly what happened to the previous generation of ViT
    runs in ``from_HPC/models_new`` -- their logs show 'libcuda ... NOT_FOUND' while the
    job held a GPU allocation. A silent CPU fallback would make this timing meaningless,
    so treat it as a hard failure. Mirrors containers/test_tensorflow.slurm.sh.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise SystemExit(
            "FATAL: no GPU visible to TensorFlow.\n"
            "  The job requested one but the container cannot see it. Check that the\n"
            "  singularity command includes --nv. Refusing to run: a CPU fallback would\n"
            "  produce timings that do not reflect the real grid."
        )
    print(f"GPU visible: {[g.name for g in gpus]}")


def _peak_gpu_mb() -> float | None:
    """Peak GPU bytes allocated by this process, in MB, or None if unavailable."""
    try:
        return tf.config.experimental.get_memory_info("GPU:0")["peak"] / 1e6
    except Exception:
        return None


def run_one(name: str, folds: int, dataset_path: str) -> dict:
    space = {**CONFIGS[name], **_FIXED}
    outdir = TESTRUN_DIR / name
    outdir.mkdir(parents=True, exist_ok=True)

    dataset = base_dataset.load_dataset(dataset_path)
    combined = base_dataset.combine_train_val(dataset)
    print(f"Combined train+val size: {len(combined['labels'])}")
    print(f"Config '{name}': {  {k: v[0] for k, v in space.items()} }")

    search = RandomSearch(
        ViTModelBuilder(), space, k=folds, seed=1, cv_seed=42, run_id=name
    )
    start = time.perf_counter()
    rows = search.run_grid(
        train_images=combined["images"],
        train_labels=combined["labels"],
        test_images=dataset["test_images"],
        test_labels=dataset["test_labels"],
        global_dir=str(outdir),
        # top_n=1 so the artefacts (model.keras + plots) are still exercised. Writing
        # them is part of the grid's code path, so a test that skipped it would not
        # prove the path works.
        top_n=1,
    )
    elapsed = time.perf_counter() - start

    if rows.empty:
        # _train_one swallows per-config exceptions so one failure cannot abort a sweep,
        # so an OOM shows up here as zero rows. Surface it as a non-zero exit.
        raise SystemExit(
            f"FAILED: config '{name}' produced no result row. The harness caught an\n"
            f"  exception during build/train -- read the traceback printed above (an OOM\n"
            f"  appears as a ResourceExhaustedError). Peak GPU: {_peak_gpu_mb()} MB"
        )

    row = rows.iloc[0]
    result = {
        "config": name,
        "patch_size": int(row["patch_size"]),
        "projection_dim": int(row["projection_dim"]),
        "num_transformer_layers": int(row["num_transformer_layers"]),
        "tokens": (64 // int(row["patch_size"])) ** 2,
        "folds": folds,
        "wall_seconds": round(elapsed, 1),
        "train_seconds": float(row["train_seconds"]),
        "best_epoch": int(row["best_epoch"]),
        "avg_vaf": float(row["avg_vaf"]),
        "test_acc": float(row["test_acc"]),
        "peak_gpu_mb": _peak_gpu_mb(),
    }
    (outdir / "bench.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


def summarise() -> None:
    """Combine the per-config benchmarks into a full-grid estimate."""
    results = {}
    for name in CONFIGS:
        path = TESTRUN_DIR / name / "bench.json"
        if path.exists():
            results[name] = json.loads(path.read_text())

    if not results:
        raise SystemExit("No bench.json files found under testrun/. Run the configs first.")

    print("\n=== Per-config measurements ===")
    print(f"{'config':>7} {'tokens':>7} {'depth':>6} {'proj':>5} "
          f"{'train_s':>9} {'best_ep':>8} {'peak_MB':>9} {'avg_vaf':>8}")
    for name, r in results.items():
        peak = f"{r['peak_gpu_mb']:.0f}" if r["peak_gpu_mb"] else "n/a"
        print(f"{name:>7} {r['tokens']:>7} {r['num_transformer_layers']:>6} "
              f"{r['projection_dim']:>5} {r['train_seconds']:>9.1f} "
              f"{r['best_epoch']:>8} {peak:>9} {r['avg_vaf']:>8.4f}")

    patch_runs = [results[n] for n in ("p4", "p8", "p16") if n in results]
    if len(patch_runs) < 3:
        print("\nNeed p4, p8 and p16 to estimate the grid. Missing some; stopping here.")
        return

    # Each patch size covers exactly 648 configs, and these three runs share one
    # architecture, so summing them and scaling by 648 estimates the whole grid.
    per_patch_total = sum(r["train_seconds"] for r in patch_runs)
    grid_seconds = CONFIGS_PER_PATCH_SIZE * per_patch_total
    grid_hours = grid_seconds / 3600

    print(f"\n=== Full-grid estimate ({GRID_TOTAL} configs, {patch_runs[0]['folds']}-fold CV) ===")
    print(f"Sum of one config per patch size : {per_patch_total:.0f} s")
    print(f"x {CONFIGS_PER_PATCH_SIZE} configs per patch size : "
          f"{grid_seconds:.0f} s = {grid_hours:.0f} GPU-hours")
    print("\nWall clock by concurrent GPU count:")
    for n_gpu in (2, 4, 8, 16):
        print(f"  {n_gpu:>2} GPUs : {grid_hours / n_gpu:6.1f} h "
              f"= {grid_hours / n_gpu / 24:5.2f} days")

    print("\nCaveat: these three fix projection 64 / 8 heads / depth 4, so they estimate\n"
          "a mid-cost config in each patch bucket. Configs with depth 6 and projection\n"
          "128 cost more, those with depth 2 and projection 32 cost less.")
    if "worst" in results:
        w, mid = results["worst"], results["p4"]
        ratio = w["train_seconds"] / mid["train_seconds"] if mid["train_seconds"] else 0
        print(f"The 'worst' config cost {ratio:.2f}x the p4 mid config, so the true grid\n"
              f"total sits above the figure above. Scale by roughly that factor for a\n"
              f"conservative upper bound: {grid_hours * ratio:.0f} GPU-hours.")
        peak = w["peak_gpu_mb"]
        if peak:
            print(f"\nPeak GPU memory on the worst config: {peak:.0f} MB. This is the\n"
                  f"configuration that previously ran out of memory.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-config ViT test run and timing probe")
    parser.add_argument("-c", "--config", choices=sorted(CONFIGS), help="which config to run")
    parser.add_argument("-k", "--folds", type=int, default=5, help="CV folds (grid uses 5)")
    parser.add_argument("-d", "--dataset", type=str, default=str(DEFAULT_DATASET))
    parser.add_argument("--summarise", action="store_true", help="combine results into an estimate")
    parser.add_argument("--allow-cpu", action="store_true",
                        help="skip the GPU check (local smoke testing only)")
    args = parser.parse_args()

    if args.summarise:
        summarise()
        return
    if not args.config:
        parser.error("--config is required unless --summarise is given")
    if not args.allow_cpu:
        _require_gpu()
    run_one(args.config, args.folds, args.dataset)


if __name__ == "__main__":
    main()
