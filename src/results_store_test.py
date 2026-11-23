import os

import pandas as pd
import pytest

from results_store import (
    JobCounter,
    TopNStore,
    _acquire_lock,
    _release_lock,
    append_rows_to_csv,
)


def _saver(name):
    def save(folder):
        with open(os.path.join(folder, "marker.txt"), "w") as fh:
            fh.write(name)

    return save


def _kept_ids(store):
    return {
        name.split("_", 1)[1] for name in os.listdir(store.results_dir) if not name.startswith(".")
    }


class TestTopNStore:
    def test_keeps_top_n_and_evicts_worst(self, tmp_path):
        store = TopNStore(str(tmp_path), top_n=2)
        assert store.offer("a", 0.5, _saver("a")) is True
        assert store.offer("b", 0.7, _saver("b")) is True  # store now full, threshold 0.5
        assert store.offer("c", 0.4, _saver("c")) is False  # below cutoff -> rejected
        assert store.offer("d", 0.9, _saver("d")) is True  # beats 0.5 -> evicts "a"
        assert _kept_ids(store) == {"b", "d"}
        assert store.offer("e", 0.6, _saver("e")) is False  # cutoff now 0.7

    def test_fast_reject_never_calls_save_fn(self, tmp_path):
        store = TopNStore(str(tmp_path), top_n=1)
        store.offer("a", 0.8, _saver("a"))  # fills, threshold 0.8
        called = []
        store.offer("b", 0.1, lambda folder: called.append(True))
        assert called == []

    def test_threshold_file_tracks_cutoff(self, tmp_path):
        store = TopNStore(str(tmp_path), top_n=2)
        store.offer("a", 0.5, _saver("a"))
        assert store._read_threshold() == float("-inf")  # not full yet
        store.offer("b", 0.7, _saver("b"))
        assert store._read_threshold() == 0.5  # full -> min kept score


class TestJobCounter:
    def test_claims_increment_then_exhaust(self, tmp_path):
        c = JobCounter(str(tmp_path), total=3)
        assert [c.claim() for _ in range(5)] == [0, 1, 2, None, None]

    def test_resumes_from_persisted_counter(self, tmp_path):
        JobCounter(str(tmp_path), 5).claim()  # -> 0, file now holds 1
        assert JobCounter(str(tmp_path), 5).claim() == 1  # a fresh counter resumes

    def test_two_workers_get_disjoint_indices(self, tmp_path):
        a, b = JobCounter(str(tmp_path), 4), JobCounter(str(tmp_path), 4)
        assert [a.claim(), b.claim(), a.claim(), b.claim(), a.claim()] == [0, 1, 2, 3, None]


class TestAppendRowsToCsv:
    def test_merges_and_sorts(self, tmp_path):
        append_rows_to_csv(str(tmp_path), pd.DataFrame([{"model_id": "1-0", "avg_vaf": 0.3}]))
        append_rows_to_csv(str(tmp_path), pd.DataFrame([{"model_id": "2-0", "avg_vaf": 0.9}]))
        df = pd.read_csv(tmp_path / "random_search_results.csv")
        assert len(df) == 2
        assert list(df["avg_vaf"]) == [0.9, 0.3]


class TestLocks:
    def test_acquire_times_out_while_held(self, tmp_path):
        lock = tmp_path / ".lock"
        lock.mkdir()
        with pytest.raises(TimeoutError, match="Timed out waiting for lock"):
            _acquire_lock(str(lock), timeout=0.05, poll=0.01)

    def test_release_of_missing_lock_is_ignored(self, tmp_path):
        _release_lock(str(tmp_path / "never_created"))


class TestTopNStoreEdges:
    def test_folders_without_a_score_are_ignored(self, tmp_path):
        store = TopNStore(str(tmp_path), top_n=2)
        (tmp_path / "results" / "notes").mkdir()
        assert store._kept() == []

    def test_failed_save_removes_staging_and_raises(self, tmp_path):
        store = TopNStore(str(tmp_path), top_n=2)

        def broken(folder):
            raise RuntimeError("plot failed")

        with pytest.raises(RuntimeError, match="plot failed"):
            store.offer("m1", 0.5, broken)
        assert list((tmp_path / "results").iterdir()) == []

    def test_rejects_model_if_cutoff_rose_while_staging(self, tmp_path):
        store = TopNStore(str(tmp_path), top_n=1)

        def save_while_another_task_wins(folder):
            # Another task keeps a better model between the fast check and the lock
            (tmp_path / "results" / "0.900000_other").mkdir()

        assert store.offer("m1", 0.5, save_while_another_task_wins) is False
        assert _kept_ids(store) == {"other"}
