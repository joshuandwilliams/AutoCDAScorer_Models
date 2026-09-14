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


def test_all_valid_configs_deterministic_and_valid():
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
    b = CNNModelBuilder()
    a1 = RandomSearch(b, space, k=2, run_id="1").all_valid_configs((64, 64, 3))
    a2 = RandomSearch(b, space, k=2, run_id="2").all_valid_configs((64, 64, 3))
    assert a1 == a2  # same order in every task
    assert len(a1) == 3  # (7,4) filtered out
    assert all(b.is_valid(p, (64, 64, 3)) for p in a1)


def test_run_grid_covers_every_config_once(tiny_data, tmp_path, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    images, labels = tiny_data
    search = RandomSearch(CNNModelBuilder(), _three_config_space(), k=2, run_id="1")
    df = search.run_grid(images, labels, global_dir=str(tmp_path), top_n=2)

    assert len(df) == 3  # this single worker did all 3 configs
    combined = pd.read_csv(tmp_path / "random_search_results.csv", dtype={"model_id": str})
    assert sorted(combined["model_id"]) == ["0", "1", "2"]
    assert int(open(tmp_path / "next.txt").read()) == 3  # counter exhausted
    assert len(_result_folders(tmp_path)) == 2  # global top-2 kept


def test_two_grid_workers_cover_grid_exactly_once(tiny_data, tmp_path, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    images, labels = tiny_data
    for rid in ("1", "2"):
        RandomSearch(CNNModelBuilder(), _three_config_space(), k=2, run_id=rid).run_grid(
            images, labels, global_dir=str(tmp_path), top_n=3
        )
    combined = pd.read_csv(tmp_path / "random_search_results.csv", dtype={"model_id": str})
    assert sorted(combined["model_id"]) == ["0", "1", "2"]  # each config done exactly once


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


def test_soft_targets_are_used_for_training_not_validation(monkeypatch):
    """soft_targets must replace the one-hot TRAINING target and leave validation alone.

    Guards the models/05 soft-label run: if this plumbing broke, that search would
    silently train on one-hot medians and look like a duplicate of models/03.
    """
    import numpy as np
    import tensorflow as tf

    from model_builders import CNNModelBuilder
    from random_search import RandomSearch

    seen = {}

    class Recorder(CNNModelBuilder):
        def build(self, params, input_shape, num_classes):
            model = super().build(params, input_shape, num_classes)
            real_fit = model.fit

            def fit(x, y, **kw):
                seen.setdefault("train_y", []).append(np.asarray(y))
                seen.setdefault("val_y", []).append(np.asarray(kw["validation_data"][1]))
                return real_fit(x, y, **kw)

            model.fit = fit
            return model

    rng = np.random.default_rng(0)
    X = rng.random((70, 16, 16, 3)).astype("float32")
    y = np.tile(np.arange(7), 10)
    # A soft target that is deliberately NOT one-hot and NOT peaked at the hard label.
    soft = np.full((70, 7), 1.0 / 7.0, dtype="float32")

    space = {
        "num_filters": [4], "filter_size": [3], "learning_rate": [0.01], "epochs": [1],
        "num_layers": [1], "pooling_size": [2], "activation_function": ["relu"],
        "batch_size": [16], "reg": [None], "reg_strength": [0.0001], "opt": ["Adam"],
        "dropout": [0.0],
    }
    s = RandomSearch(Recorder(), space, k=2, seed=1, cv_seed=42, run_id="t")
    s._cross_validate(
        list(__import__("sklearn.model_selection", fromlist=["ParameterGrid"])
             .ParameterGrid(space))[0],
        X, y, sorted(set(y.tolist())), soft_targets=soft,
    )

    train_y = np.concatenate(seen["train_y"])
    val_y = np.concatenate(seen["val_y"])
    # Training target is the uniform soft distribution, not one-hot.
    assert np.allclose(train_y, 1.0 / 7.0), "training did not use the soft targets"
    # Validation target is still one-hot on the hard label.
    assert set(np.unique(val_y).tolist()) <= {0.0, 1.0}, "validation target was softened"
    assert np.allclose(val_y.sum(axis=1), 1.0)


def test_grouped_folds_keep_augmented_copies_with_their_source():
    """No group may straddle the train/validation boundary of any fold.

    Guards the models/07 augmented runs. An augmented crop shares its ``source`` index with
    the real crop it was made from; if grouping broke, a validation fold would contain
    augmented copies of images whose originals were trained on, and ``avg_vaf`` would
    measure recall of the training set rather than generalisation -- while still looking
    like a perfectly normal, slightly high number.
    """
    import numpy as np

    from model_builders import CNNModelBuilder
    from random_search import RandomSearch

    rng = np.random.default_rng(0)
    # 40 source crops, each with 3 augmented copies: 160 rows, 4 per group.
    labels = np.repeat(np.arange(4), 10)
    source = np.arange(40)
    y = np.concatenate([labels] * 4)
    groups = np.concatenate([source] * 4)
    images = rng.random((len(y), 8, 8, 3)).astype("float32")

    search = RandomSearch(CNNModelBuilder(), _three_config_space(), k=4, cv_seed=42)
    folds = search._folds(images, y, groups)
    assert len(folds) == 4
    for train_idx, val_idx in folds:
        assert not set(groups[train_idx]) & set(groups[val_idx])
    # Every row must be validated exactly once across the folds.
    seen = np.concatenate([v for _, v in folds])
    assert sorted(seen.tolist()) == list(range(len(y)))

    # Ungrouped folds on the same data DO split groups -- proving the test above is not
    # passing for some unrelated reason.
    plain = search._folds(images, y)
    assert any(set(groups[t]) & set(groups[v]) for t, v in plain)


def test_split_accuracy_reports_real_and_augmented_separately(monkeypatch):
    """avg_vaf_orig / avg_vaf_aug must be computed over the right subsets.

    A model forced to predict a constant class gives an accuracy equal to that class's
    share of the subset, so the expected values are known exactly and the columns can be
    checked rather than merely asserted to exist.
    """
    import numpy as np

    from model_builders import CNNModelBuilder
    from random_search import RandomSearch

    class AlwaysZero(CNNModelBuilder):
        def build(self, params, input_shape, num_classes):
            model = super().build(params, input_shape, num_classes)
            # Only predict is stubbed. fit still runs (one epoch on 60 tiny images), so
            # the History object the harness reads per-epoch accuracy out of is real.
            model.predict = lambda x, **kw: np.tile(
                np.eye(num_classes)[0], (len(x), 1)
            ).astype("float32")
            return model

    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(2), 40)               # 40 zeros, 40 ones
    groups = np.arange(80)
    # Real rows are all class 0; augmented rows are all class 1. A constant-zero predictor
    # therefore scores 1.0 on the real subset and 0.0 on the augmented one, whatever the
    # fold split happens to be.
    is_aug = y == 1
    images = rng.random((80, 8, 8, 3)).astype("float32")

    search = RandomSearch(AlwaysZero(), _three_config_space(), k=4, cv_seed=42)
    cv = search._cross_validate(
        {"num_filters": 4, "filter_size": 3, "learning_rate": 0.01, "epochs": 1,
         "num_layers": 1, "pooling_size": 2, "activation_function": "relu",
         "batch_size": 8, "reg": None, "reg_strength": 0.0001, "opt": "Adam",
         "dropout": 0.0},
        images, y, [0, 1], groups=groups, is_aug=is_aug,
    )
    assert np.allclose(cv["val_acc_orig_fold"], 1.0)
    assert np.allclose(cv["val_acc_aug_fold"], 0.0)
    assert np.allclose(cv["train_acc_orig_fold"], 1.0)
    assert np.allclose(cv["train_acc_aug_fold"], 0.0)
    # The pooled number must sit between the two, not equal either.
    assert all(0.0 < a < 1.0 for a in cv["val_acc_fold"])
