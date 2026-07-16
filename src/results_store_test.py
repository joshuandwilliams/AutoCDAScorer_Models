import os

import pandas as pd

from results_store import JobCounter, TopNStore, append_rows_to_csv


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
