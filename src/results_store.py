"""Shared, concurrency-safe global top-N store for a distributed random search.

Many SLURM array tasks run the same search in parallel and all contribute to a
single global top-N of models kept on disk, plus one combined results CSV. The
expensive artefacts (plots + Keras model) are only ever written for models that
beat a shared, monotonically-rising threshold, so the vast majority of models
cost nothing on disk.

Coordination primitives (all on the shared filesystem):
  - ``threshold.txt``  : the current cutoff avg_vaf (min score among the kept
                         models once ``top_n`` are kept; -inf before then). Read
                         lock-free for a cheap fast-reject.
  - ``results/``       : up to ``top_n`` folders named ``{avg_vaf}_{model_id}``,
                         so the folder names encode the scores.
  - mkdir-based locks  : ``.topn_lock`` / ``.csv_lock`` -- ``os.mkdir`` is atomic
                         even on NFS/GPFS, so this is a portable cross-node lock.
"""

import os
import shutil
import time

import pandas as pd

RESULTS_CSV = "random_search_results.csv"


def _acquire_lock(lock_dir: str, timeout: float = 1800.0, poll: float = 0.5) -> None:
    """Block until ``lock_dir`` can be created (atomic cross-node mutex)."""
    start = time.perf_counter()
    while True:
        try:
            os.mkdir(lock_dir)
            return
        except FileExistsError:
            if time.perf_counter() - start > timeout:
                raise TimeoutError(f"Timed out waiting for lock {lock_dir}") from None
            time.sleep(poll)


def _release_lock(lock_dir: str) -> None:
    try:
        os.rmdir(lock_dir)
    except OSError:
        pass


class TopNStore:
    """Keeps only the globally best ``top_n`` models' artefacts across all tasks."""

    def __init__(self, root: str, top_n: int):
        self.root = root
        self.top_n = top_n
        self.results_dir = os.path.join(root, "results")
        self.threshold_file = os.path.join(root, "threshold.txt")
        self.lock_dir = os.path.join(root, ".topn_lock")
        os.makedirs(self.results_dir, exist_ok=True)

    def _read_threshold(self) -> float:
        try:
            with open(self.threshold_file) as fh:
                return float(fh.read().strip())
        except (OSError, ValueError):
            return float("-inf")

    def _write_threshold(self, value: float) -> None:
        tmp = f"{self.threshold_file}.{os.getpid()}.tmp"
        with open(tmp, "w") as fh:
            fh.write(repr(value))
        os.replace(tmp, self.threshold_file)  # atomic; readers see old or new, never partial

    def _kept(self) -> list:
        """Return [(avg_vaf, folder_name), ...] for the models currently kept."""
        kept = []
        for name in os.listdir(self.results_dir):
            if name.startswith(".") or not os.path.isdir(os.path.join(self.results_dir, name)):
                continue
            try:
                kept.append((float(name.split("_", 1)[0]), name))
            except ValueError:
                continue
        return kept

    def offer(self, model_id: str, avg_vaf: float, save_fn) -> bool:
        """Offer a model to the global top-N; write its artefacts iff it qualifies.

        ``save_fn(folder)`` renders the artefacts into ``folder``. It is called only
        for models that beat the current threshold, and outside the lock. Returns
        True if the model was kept.
        """
        # Fast path (no lock): below the cutoff -> never touch the disk.
        if avg_vaf <= self._read_threshold():
            return False

        # Build artefacts into a staging folder outside the lock (the slow part).
        staging = os.path.join(self.results_dir, f".staging_{model_id}")
        shutil.rmtree(staging, ignore_errors=True)
        os.makedirs(staging, exist_ok=True)
        try:
            save_fn(staging)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

        _acquire_lock(self.lock_dir)
        try:
            kept = self._kept()
            if len(kept) >= self.top_n:
                worst_score, worst_name = min(kept)
                if avg_vaf <= worst_score:  # the cutoff rose past us while staging
                    shutil.rmtree(staging, ignore_errors=True)
                    return False
                shutil.rmtree(os.path.join(self.results_dir, worst_name), ignore_errors=True)
                kept.remove((worst_score, worst_name))
            final = os.path.join(self.results_dir, f"{avg_vaf:.6f}_{model_id}")
            os.replace(staging, final)
            kept.append((avg_vaf, final))
            threshold = min(s for s, _ in kept) if len(kept) >= self.top_n else float("-inf")
            self._write_threshold(threshold)
            return True
        finally:
            _release_lock(self.lock_dir)


class JobCounter:
    """Shared 'next unclaimed job index' for self-scheduling array tasks.

    Each task repeatedly calls :meth:`claim` to atomically take the next 0-based index
    until the counter reaches ``total``. This gives dynamic load balancing -- fast
    tasks claim more, slow ones fewer -- with no fixed slices to get unlucky with. The
    counter file persists, so a re-submitted job simply resumes where it left off.
    """

    def __init__(self, root: str, total: int):
        self.total = total
        self.counter_file = os.path.join(root, "next.txt")
        self.lock_dir = os.path.join(root, ".counter_lock")

    def claim(self):
        """Atomically return the next index, or None once the grid is exhausted."""
        _acquire_lock(self.lock_dir)
        try:
            try:
                with open(self.counter_file) as fh:
                    i = int(fh.read().strip())
            except (OSError, ValueError):
                i = 0
            if i >= self.total:
                return None
            tmp = f"{self.counter_file}.{os.getpid()}.tmp"
            with open(tmp, "w") as fh:
                fh.write(str(i + 1))
            os.replace(tmp, self.counter_file)
            return i
        finally:
            _release_lock(self.lock_dir)


def append_rows_to_csv(root: str, rows_df: pd.DataFrame, filename: str = RESULTS_CSV) -> None:
    """Merge a task's rows into the single shared, avg_vaf-sorted results CSV (locked)."""
    lock_dir = os.path.join(root, ".csv_lock")
    csv_path = os.path.join(root, filename)
    _acquire_lock(lock_dir)
    try:
        if os.path.exists(csv_path):
            rows_df = pd.concat([pd.read_csv(csv_path), rows_df], ignore_index=True)
        rows_df.sort_values("avg_vaf", ascending=False).reset_index(drop=True).to_csv(
            csv_path, index=False
        )
    finally:
        _release_lock(lock_dir)
