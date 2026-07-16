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
    # >= 2 grid points so ParameterSampler(n_iter=2) can sample without replacement.
    return {
        "num_filters": [4, 8],
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


def test_run_returns_ranked_leaderboard(tiny_data, tiny_space, tmp_path, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    images, labels = tiny_data

    search = RandomSearch(CNNModelBuilder(), tiny_space, k=2, seed=1)
    leaderboard = search.run(images, labels, n_models=2, output_dir=str(tmp_path))

    assert isinstance(leaderboard, pd.DataFrame)
    assert len(leaderboard) == 2
    # Ranked by mean CV accuracy, descending.
    assert list(leaderboard["avg_vaf"]) == sorted(leaderboard["avg_vaf"], reverse=True)
    # The new ordinal metric is reported alongside near-miss.
    assert "avg_val_qwk" in leaderboard.columns
    assert "avg_val_nearmiss" in leaderboard.columns
    # Leaderboard file was written.
    assert (tmp_path / "random_search_results.csv").exists()


def test_run_reports_test_metrics_when_test_set_given(tiny_data, tiny_space, tmp_path, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    images, labels = tiny_data

    search = RandomSearch(CNNModelBuilder(), tiny_space, k=2, seed=1)
    leaderboard = search.run(
        images, labels, test_images=images, test_labels=labels, n_models=1, output_dir=str(tmp_path)
    )
    for col in ("test_acc", "test_nearmiss", "test_qwk"):
        assert col in leaderboard.columns
