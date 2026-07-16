import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from model_builders import CNNModelBuilder
from random_search import RandomSearch


@pytest.fixture
def tiny_data():
    """32 tiny RGB images with all 7 classes present (>= 2 per class for k=2 CV)."""
    rng = np.random.default_rng(0)
    images = rng.random((32, 64, 64, 3)).astype("float32")
    labels = np.arange(32) % 7
    return images, labels


@pytest.fixture
def tiny_space():
    return {
        "num_filters": [4, 8],  # 2 valid configs
        "filter_size": [3],
        "learning_rate": [0.001],
        "epochs": [1],
        "num_layers": [1],
        "pooling_size": [2],
        "activation_function": ["relu"],
        "batch_size": [8],
        "reg": [None],
        "reg_strength": [0.001],
        "opt": ["Adam"],
        "dropout": [0.0],
    }


def _result_folders(root):
    results = root / "results"
    return [d for d in results.iterdir() if d.is_dir()] if results.is_dir() else []


def test_run_returns_ranked_rows_and_shared_outputs(tiny_data, tiny_space, tmp_path, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    images, labels = tiny_data

    search = RandomSearch(CNNModelBuilder(), tiny_space, k=2, seed=1, run_id="1")
    df = search.run(images, labels, n_models=2, global_dir=str(tmp_path))

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df["avg_vaf"]) == sorted(df["avg_vaf"], reverse=True)
    for col in ("avg_val_qwk", "avg_val_nearmiss", "train_seconds"):
        assert col in df.columns
    assert (df["train_seconds"] >= 0).all()

    # A single shared, sorted results CSV with run-id-prefixed model ids.
    combined = pd.read_csv(tmp_path / "random_search_results.csv")
    assert len(combined) == 2
    assert all(str(mid).startswith("1-") for mid in combined["model_id"])

    # results/ holds the kept models (each with a Keras model); no per-model CSVs.
    folders = _result_folders(tmp_path)
    assert 1 <= len(folders) <= 2
    assert all((d / "model.keras").exists() for d in folders)
    assert not list(tmp_path.glob("**/results_*.csv"))


def test_global_top_n_keeps_only_the_best(tiny_data, tmp_path, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    images, labels = tiny_data
    search = RandomSearch(CNNModelBuilder(), _three_config_space(), k=2, seed=1, run_id="1")
    df = search.run(images, labels, n_models=3, global_dir=str(tmp_path), top_n=1)

    assert len(df) == 3  # every model is in the results table...
    folders = _result_folders(tmp_path)
    assert len(folders) == 1  # ...but only the single global best keeps artefacts
    kept_score = float(folders[0].name.split("_", 1)[0])
    assert abs(kept_score - df.iloc[0]["avg_vaf"]) < 1e-5
    assert (folders[0] / "model.keras").exists()


def test_two_runs_share_one_global_store(tiny_data, tiny_space, tmp_path, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    images, labels = tiny_data
    for rid in ("1", "2"):
        RandomSearch(CNNModelBuilder(), tiny_space, k=2, seed=int(rid), run_id=rid).run(
            images, labels, n_models=2, global_dir=str(tmp_path), top_n=2
        )

    combined = pd.read_csv(tmp_path / "random_search_results.csv")
    assert len(combined) == 4  # 2 runs x 2 models, all in one CSV
    ids = {str(i) for i in combined["model_id"]}
    assert any(i.startswith("1-") for i in ids) and any(i.startswith("2-") for i in ids)

    # Exactly the global top-2 keep artefacts.
    folders = _result_folders(tmp_path)
    assert len(folders) == 2
    kept = sorted(float(d.name.split("_", 1)[0]) for d in folders)
    top2 = sorted(combined["avg_vaf"].nlargest(2))
    assert all(abs(a - b) < 1e-5 for a, b in zip(kept, top2, strict=True))


def test_run_reports_test_metrics_when_test_set_given(tiny_data, tiny_space, tmp_path, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    images, labels = tiny_data
    search = RandomSearch(CNNModelBuilder(), tiny_space, k=2, seed=1, run_id="1")
    df = search.run(
        images, labels, test_images=images, test_labels=labels, n_models=1, global_dir=str(tmp_path)
    )
    for col in ("test_acc", "test_nearmiss", "test_qwk"):
        assert col in df.columns


def test_cv_folds_depend_on_cv_seed_not_sampling_seed(tiny_data):
    images, labels = tiny_data
    a = RandomSearch(CNNModelBuilder(), {}, k=2, seed=1, cv_seed=42)
    b = RandomSearch(
        CNNModelBuilder(), {}, k=2, seed=999, cv_seed=42
    )  # diff sampling, same cv_seed
    c = RandomSearch(CNNModelBuilder(), {}, k=2, seed=1, cv_seed=7)  # same sampling, diff cv_seed

    fa, fb, fc = a._folds(images, labels), b._folds(images, labels), c._folds(images, labels)

    # Same cv_seed -> identical folds regardless of the sampling seed.
    assert all(
        np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1])
        for x, y in zip(fa, fb, strict=True)
    )
    # Different cv_seed -> different folds.
    assert any(not np.array_equal(x[1], y[1]) for x, y in zip(fa, fc, strict=True))


def test_sampler_returns_only_valid_configs():
    space = {
        "num_filters": [8],
        "filter_size": [3, 7],
        "learning_rate": [0.001],
        "epochs": [1],
        "num_layers": [1, 4],
        "pooling_size": [2],
        "activation_function": ["relu"],
        "batch_size": [8],
        "reg": [None],
        "reg_strength": [0.001],
        "opt": ["Adam"],
        "dropout": [0.0],
    }
    builder = CNNModelBuilder()
    search = RandomSearch(builder, space, k=2, seed=1, run_id="1")
    configs = search._sample_valid_configs(n_models=4, input_shape=(64, 64, 3))
    assert all(builder.is_valid(c, (64, 64, 3)) for c in configs)
    # (3,7) x (1,4) has 4 combos; (7,4) is invalid, so 3 remain.
    assert len(configs) == 3


def _three_config_space():
    return {
        "num_filters": [4, 8, 16],  # 3 valid configs
        "filter_size": [3],
        "learning_rate": [0.001],
        "epochs": [1],
        "num_layers": [1],
        "pooling_size": [2],
        "activation_function": ["relu"],
        "batch_size": [8],
        "reg": [None],
        "reg_strength": [0.001],
        "opt": ["Adam"],
        "dropout": [0.0],
    }
