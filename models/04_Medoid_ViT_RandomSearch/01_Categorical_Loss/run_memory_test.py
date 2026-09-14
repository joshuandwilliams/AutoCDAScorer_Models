"""Does memory accumulate across models trained back-to-back in ONE process?

Why this is a separate test
--------------------------
The earlier single-config probe (models/02_.../run_single_test.py) ran each config in
its own ``singularity exec``. Process teardown frees everything, so it proved the grid
code path works but said nothing about accumulation. The real search does the opposite:
one array task runs ``RandomSearch.run_grid``'s while-loop and trains hundreds of configs
**inside a single process**. That is where a leak shows up.

This is not hypothetical for this repo. Commit 13fa770 records the CNN sweep dying
exactly this way -- "RSS climbed until the 8 GB task was OOM-killed (62/150 in the
overnight grid run)" -- which was host RAM, on the sibling search, before the per-fold
teardown existed.

Design
------
PHASE A (leak detection). Nine configs that share ONE architecture and differ only in
``learning_rate`` and ``dropout``. Neither changes tensor shapes, so every config's
memory footprint is identical by construction. Memory is therefore expected to be flat
across all nine, and any monotonic climb is a leak rather than a difference in workload.
Measuring drift needs a constant workload, which is the whole reason the varied axes are
non-architectural.

PHASE B (headroom). The largest config in the new grid -- patch 4 (256 tokens),
projection 256, depth 8, mlp_ratio 4, at batch 64 -- run once for peak memory and wall
clock. This is the config that sizes ``--mem`` and the time limit for the full run.

Metrics, per config
-------------------
``gpu_peak_mb``    peak device bytes for THIS config. ``reset_memory_stats`` is called
                   before each one, because TensorFlow's peak counter is monotonic and
                   would otherwise only ever report the high-water mark of the run.
``gpu_current_mb`` device bytes still held AFTER the config finished and the harness has
                   run its teardown. This is the leak indicator: it should return to the
                   same value every time.
``rss_mb``         host resident set size, read from /proc/self/status. The CNN sweep was
                   killed on host RAM, not VRAM, so this is tracked separately.

Usage
-----
    python3.12 run_memory_test.py --phase a
    python3.12 run_memory_test.py --phase b
    python3.12 run_memory_test.py --report
"""

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT / "src"))

import tensorflow as tf  # noqa: E402

for _gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError:
        pass

import base_dataset  # noqa: E402
from model_builders import ViTModelBuilder  # noqa: E402
from random_search import RandomSearch  # noqa: E402

DEFAULT_DATASET = REPO_ROOT / "data" / "datasets" / "medoid_64" / "medoid_64.npy"
OUTDIR = Path(__file__).resolve().parent / "memtest"

# Architecture held constant; only learning_rate and dropout vary, and neither changes
# any tensor shape. 3 x 3 = 9 configs of identical memory footprint, trained in series.
PHASE_A = {
    "patch_size": [4],
    "projection_dim": [128],
    "num_heads": [8],
    "num_transformer_layers": [6],
    "mlp_ratio": [2],
    "activation_function": ["gelu"],
    "epochs": [50],
    "reg": [None],
    "reg_strength": [0.0001],
    "opt": ["AdamW"],
    "batch_size": [64],
    "learning_rate": [0.01, 0.001, 0.0001],
    "dropout": [0.0, 0.1, 0.3],
}

# The most expensive config the new grid can produce.
PHASE_B = {
    "patch_size": [4],
    "projection_dim": [256],
    "num_heads": [8],
    "num_transformer_layers": [8],
    "mlp_ratio": [4],
    "activation_function": ["gelu"],
    "epochs": [50],
    "reg": [None],
    "reg_strength": [0.0001],
    "opt": ["AdamW"],
    "batch_size": [64],
    "learning_rate": [0.001],
    "dropout": [0.1],
}


def _rss_mb() -> float | None:
    """Current host RSS in MB, from /proc/self/status (Linux only)."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return None


def _gpu_mb() -> tuple[float | None, float | None]:
    """(current, peak) device MB for GPU:0, or (None, None) if unavailable."""
    try:
        info = tf.config.experimental.get_memory_info("GPU:0")
        return info["current"] / 1e6, info["peak"] / 1e6
    except Exception:
        return None, None


def _reset_gpu_peak() -> None:
    """Reset the monotonic peak counter so the next reading is per-config."""
    try:
        tf.config.experimental.reset_memory_stats("GPU:0")
    except Exception:
        pass


class InstrumentedSearch(RandomSearch):
    """RandomSearch that records memory around each config.

    Overriding ``_train_one`` keeps the measurement on the real code path: the
    fold loop, teardown, artefact save and CSV write all still run exactly as they do
    in the grid.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.samples: list[dict] = []

    def _train_one(self, params, model_id, *args, **kwargs):
        _reset_gpu_peak()
        before_cur, _ = _gpu_mb()
        start = time.perf_counter()

        row = super()._train_one(params, model_id, *args, **kwargs)

        after_cur, after_peak = _gpu_mb()
        rss = _rss_mb()
        sample = {
            "index": len(self.samples),
            "model_id": model_id,
            "learning_rate": params.get("learning_rate"),
            "dropout": params.get("dropout"),
            "seconds": round(time.perf_counter() - start, 1),
            "gpu_before_mb": round(before_cur, 1) if before_cur is not None else None,
            "gpu_current_mb": round(after_cur, 1) if after_cur is not None else None,
            "gpu_peak_mb": round(after_peak, 1) if after_peak is not None else None,
            "rss_mb": round(rss, 1) if rss is not None else None,
            "avg_vaf": float(row["avg_vaf"]) if row is not None else None,
        }
        self.samples.append(sample)
        print(
            f"  [mem] config {sample['index']:>2}  "
            f"gpu_cur={sample['gpu_current_mb']}MB  gpu_peak={sample['gpu_peak_mb']}MB  "
            f"rss={sample['rss_mb']}MB  {sample['seconds']}s",
            flush=True,
        )
        return row


def _require_gpu() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise SystemExit(
            "FATAL: no GPU visible to TensorFlow. Check that singularity was given --nv.\n"
            "  A CPU fallback would make these memory numbers meaningless."
        )
    print(f"GPU visible: {[g.name for g in gpus]}")


def run_phase(phase: str, folds: int, dataset_path: str) -> None:
    space = PHASE_A if phase == "a" else PHASE_B
    outdir = OUTDIR / phase
    outdir.mkdir(parents=True, exist_ok=True)

    dataset = base_dataset.load_dataset(dataset_path)
    combined = base_dataset.combine_train_val(dataset)
    print(f"Dataset: {dataset_path}")
    print(f"Combined train+val size: {len(combined['labels'])}")

    search = InstrumentedSearch(
        ViTModelBuilder(), space, k=folds, seed=1, cv_seed=42, run_id=f"mem{phase}"
    )
    rows = search.run_grid(
        train_images=combined["images"],
        train_labels=combined["labels"],
        test_images=dataset["test_images"],
        test_labels=dataset["test_labels"],
        global_dir=str(outdir),
        top_n=1,
    )
    (outdir / "memory_samples.json").write_text(json.dumps(search.samples, indent=2))

    if rows.empty:
        raise SystemExit(
            f"FAILED: phase {phase} produced no result rows. The harness caught an\n"
            f"  exception during build/train (an OOM appears as ResourceExhaustedError).\n"
            f"  Read the traceback above."
        )
    print(f"\nPhase {phase}: {len(search.samples)} configs trained in ONE process.")


def report() -> None:
    """Judge whether memory drifted across the in-process series."""
    path = OUTDIR / "a" / "memory_samples.json"
    if not path.exists():
        raise SystemExit(f"No samples at {path}. Run --phase a first.")
    s = json.loads(path.read_text())

    print(f"\n=== Phase A: identical architecture, {len(s)} configs in one process ===")
    print(f"{'#':>3} {'gpu_cur_MB':>11} {'gpu_peak_MB':>12} {'rss_MB':>9} {'sec':>7} {'avg_vaf':>8}")
    for r in s:
        vaf = f"{r['avg_vaf']:.4f}" if r["avg_vaf"] is not None else "n/a"
        print(f"{r['index']:>3} {str(r['gpu_current_mb']):>11} {str(r['gpu_peak_mb']):>12} "
              f"{str(r['rss_mb']):>9} {r['seconds']:>7} {vaf:>8}")

    def drift(key):
        vals = [r[key] for r in s if r[key] is not None]
        return (vals[0], vals[-1], vals[-1] - vals[0], max(vals)) if vals else None

    print("\n=== Drift across the series (first -> last) ===")
    # Track measurability separately from the pass/fail result. A metric that could not
    # be read is NOT evidence of no leak, and must never be reported as one.
    measured, drifted = [], []
    for key, label, tol_mb in (
        ("gpu_current_mb", "GPU held after teardown", 200.0),
        ("rss_mb", "Host RSS", 1500.0),
    ):
        d = drift(key)
        if d is None:
            print(f"{label:>26}: NOT MEASURED")
            continue
        first, last, delta, peak = d
        measured.append(label)
        flag = "OK" if abs(delta) <= tol_mb else "DRIFT"
        if flag == "DRIFT":
            drifted.append(label)
        print(f"{label:>26}: {first:.1f} -> {last:.1f} MB "
              f"(delta {delta:+.1f}, max {peak:.1f})  [{flag}, tol +/-{tol_mb:.0f}]")

    print(f"\nThe {len(s)} configs share one architecture and differ only in learning rate and\n"
          "dropout, neither of which changes a tensor shape. A rising trend is therefore a\n"
          "leak, not a heavier workload.")

    if not measured:
        # Both probes returned nothing: no GPU present, and/or /proc/self/status absent
        # (it is Linux-only, so this happens on macOS). Reporting a pass here would be a
        # test that succeeds without measuring anything.
        print("\nVERDICT: INCONCLUSIVE -- neither GPU nor host RSS could be read, so this\n"
              "run provides NO evidence either way. On the HPC both are available; if you\n"
              "see this there, the probes themselves are broken and must be fixed before\n"
              "the result means anything.")
        raise SystemExit(2)
    if drifted:
        print(f"\nVERDICT: MEMORY IS ACCUMULATING in {', '.join(drifted)}. "
              f"Do not launch the full grid.")
        raise SystemExit(1)
    print(f"\nVERDICT: no accumulation detected in {', '.join(measured)}.")

    bpath = OUTDIR / "b" / "memory_samples.json"
    if bpath.exists():
        b = json.loads(bpath.read_text())[0]
        print(f"\n=== Phase B: largest config in the new grid ===")
        print(f"patch 4 / proj 256 / depth 8 / mlp_ratio 4 / batch 64")
        print(f"  peak GPU : {b['gpu_peak_mb']} MB")
        print(f"  host RSS : {b['rss_mb']} MB")
        print(f"  seconds  : {b['seconds']} (5-fold CV)")
        print(f"  avg_vaf  : {b['avg_vaf']}")


def main() -> None:
    p = argparse.ArgumentParser(description="In-process memory accumulation test")
    p.add_argument("--phase", choices=["a", "b"], help="a = leak detection, b = largest config")
    p.add_argument("--report", action="store_true", help="summarise measured samples")
    p.add_argument("-k", "--folds", type=int, default=5)
    p.add_argument("-d", "--dataset", type=str, default=str(DEFAULT_DATASET))
    p.add_argument("--allow-cpu", action="store_true", help="skip the GPU check (local only)")
    args = p.parse_args()

    if args.report:
        report()
        return
    if not args.phase:
        p.error("--phase is required unless --report is given")
    if not args.allow_cpu:
        _require_gpu()
    run_phase(args.phase, args.folds, args.dataset)


if __name__ == "__main__":
    main()
