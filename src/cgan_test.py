import numpy as np
import pytest
import tensorflow as tf

from cgan import (
    IMG_SIZE,
    NUM_CLASSES,
    FrechetCDADistance,
    build_discriminator,
    build_generator,
    discriminator_accuracy,
    postprocess_images,
    preprocess_images,
    sample,
    train_cgan,
)

LATENT = 8
GEN_FILTERS = [8, 4, 4, 4]
DISC_FILTERS = [4, 4]


def images(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random((n, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)


def one_hot(labels) -> np.ndarray:
    return tf.keras.utils.to_categorical(labels, NUM_CLASSES)


@pytest.fixture
def feature_model_path(tmp_path):
    """A small saved CNN whose last pooling layer gives the features."""
    tf.keras.utils.set_random_seed(0)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input((IMG_SIZE, IMG_SIZE, 3)),
            tf.keras.layers.Conv2D(4, 3),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(NUM_CLASSES),
        ]
    )
    path = tmp_path / "run_1" / "model.keras"
    path.parent.mkdir()
    model.save(path)
    return path


class TestScaling:
    def test_round_trip(self):
        x = images(2)
        scaled = preprocess_images(x)
        assert scaled.min() >= -1 and scaled.max() < 1
        np.testing.assert_allclose(postprocess_images(scaled), x, atol=1e-6)

    def test_postprocess_clips(self):
        np.testing.assert_array_equal(postprocess_images(np.array([-2.0, 2.0])), [0.0, 1.0])


class TestNetworks:
    def test_generator_output(self):
        gen = build_generator(LATENT, GEN_FILTERS)
        out = gen([tf.zeros((2, LATENT)), tf.constant(one_hot([0, 6]))]).numpy()
        assert out.shape == (2, IMG_SIZE, IMG_SIZE, 3)
        assert out.min() >= -1 and out.max() <= 1

    def test_discriminator_output(self):
        disc = build_discriminator(DISC_FILTERS, dropout=0.1)
        out = disc([preprocess_images(images(3)), one_hot([0, 1, 2])])
        assert out.shape == (3, 1)

    def test_sample_gives_each_class(self):
        gen = build_generator(LATENT, GEN_FILTERS)
        imgs, labs = sample(gen, LATENT, n_per_class=2, seed=1)
        assert imgs.shape == (2 * NUM_CLASSES, IMG_SIZE, IMG_SIZE, 3)
        assert labs.tolist() == [c for c in range(NUM_CLASSES) for _ in range(2)]
        assert imgs.min() >= 0 and imgs.max() <= 1
        again, _ = sample(gen, LATENT, n_per_class=2, seed=1)
        np.testing.assert_array_equal(imgs, again)

    def test_sample_without_seed(self):
        gen = build_generator(LATENT, GEN_FILTERS)
        imgs, _ = sample(gen, LATENT, n_per_class=1)
        assert imgs.shape == (NUM_CLASSES, IMG_SIZE, IMG_SIZE, 3)


class TestDiscriminatorAccuracy:
    class Fixed:
        """Returns a fixed logit for every real image and another for every fake."""

        def __init__(self, real_logit, fake_logit):
            self.real_logit, self.fake_logit = real_logit, fake_logit
            self.calls = 0

        def __call__(self, inputs, training=False):
            self.calls += 1
            logit = self.real_logit if self.calls == 1 else self.fake_logit
            return tf.fill((len(inputs[0]), 1), logit)

    def test_perfect(self):
        acc = discriminator_accuracy(self.Fixed(1.0, -1.0), images(3), None, images(2), None)
        assert acc == 1.0

    def test_always_says_real(self):
        acc = discriminator_accuracy(self.Fixed(1.0, 1.0), images(3), None, images(1), None)
        assert acc == 0.75


class TestFrechetCDADistance:
    def test_identical_sets_score_zero(self, feature_model_path):
        fcd = FrechetCDADistance(feature_model_path)
        assert fcd.name == "run_1"
        assert fcd.layer.startswith("global_average_pooling2d")
        x = images(16)
        assert fcd(x, x) == pytest.approx(0.0, abs=1e-3)

    def test_different_sets_score_higher(self, feature_model_path):
        fcd = FrechetCDADistance(feature_model_path)
        x = images(16)
        assert fcd(x, np.zeros_like(x)) > fcd(x, images(16, seed=1))

    def test_model_without_pooling_raises(self, tmp_path):
        model = tf.keras.Sequential(
            [tf.keras.layers.Input((IMG_SIZE, IMG_SIZE, 3)), tf.keras.layers.Flatten()]
        )
        path = tmp_path / "no_pool.keras"
        model.save(path)
        with pytest.raises(ValueError, match="no pooling layer"):
            FrechetCDADistance(path)


class SequenceFCD:
    """Stands in for FrechetCDADistance, returning the given values in turn."""

    def __init__(self, values):
        self.values = iter(values)

    def __call__(self, real, fake):
        return next(self.values)


PARAMS = {
    "latent_dim": LATENT,
    "generator_filters": GEN_FILTERS,
    "discriminator_filters": DISC_FILTERS,
    "learning_rate": 1e-3,
    "batch_size": 4,
}


class TestTrainCgan:
    def test_early_stop_when_distance_stops_improving(self, tmp_path):
        history, best = train_cgan(
            images(8),
            [0, 1, 2, 3, 4, 5, 6, 0],
            PARAMS,
            epochs=10,
            out_dir=tmp_path,
            val_images=images(8, seed=1),
            fcd=SequenceFCD([3.0, 4.0, 5.0, 6.0]),
            fcd_every=1,
            patience=2,
            disc_acc_window=(0.0, 1.0),
        )
        assert best == {"fcd": 3.0, "epoch": 1, "stopped_at": 3, "epochs_run": 3}
        assert [h["epoch"] for h in history] == [1, 2, 3]
        assert {"gen_loss", "disc_loss", "fcd", "disc_acc"} <= set(history[0])
        for name in ("generator_best", "discriminator_best", "generator_final"):
            assert (tmp_path / "models" / f"{name}.keras").exists()

    def test_no_stop_outside_accuracy_window(self, tmp_path):
        _, best = train_cgan(
            images(8),
            [0] * 8,
            PARAMS,
            epochs=3,
            out_dir=tmp_path,
            fcd=SequenceFCD([3.0, 4.0, 5.0]),
            fcd_every=1,
            patience=1,
            disc_acc_window=(2.0, 3.0),
        )
        assert best["stopped_at"] is None
        assert best["epochs_run"] == 3

    def test_wall_clock_budget(self, tmp_path):
        history, best = train_cgan(
            images(8), [0] * 8, PARAMS, epochs=5, out_dir=tmp_path, sample_every=1, max_seconds=0
        )
        assert best == {"fcd": np.inf, "epoch": -1, "stopped_at": 1, "epochs_run": 1}
        assert len(history) == 1
        saved = np.load(tmp_path / "samples" / "epoch_00001.npz")
        assert saved["images"].shape == (4 * NUM_CLASSES, IMG_SIZE, IMG_SIZE, 3)


def test_fcd_uses_the_real_part_of_a_complex_square_root(feature_model_path, monkeypatch):
    import cgan

    fcd = FrechetCDADistance(feature_model_path)
    x = images(16)
    expected = fcd(x, x)
    real_sqrtm = cgan.linalg.sqrtm
    monkeypatch.setattr(cgan.linalg, "sqrtm", lambda m: real_sqrtm(m) + 0j)
    assert fcd(x, x) == pytest.approx(expected)


def test_training_step_runs_eagerly(tmp_path):
    # Eager execution runs the @tf.function step as plain Python, so its lines are
    # measured. In graph mode coverage cannot see inside the traced function.
    tf.config.run_functions_eagerly(True)
    try:
        history, _ = train_cgan(images(8), [0] * 8, PARAMS, epochs=1, out_dir=tmp_path)
    finally:
        tf.config.run_functions_eagerly(False)
    assert np.isfinite(history[0]["gen_loss"]) and np.isfinite(history[0]["disc_loss"])


def test_progress_is_printed_every_100_epochs(tmp_path, capsys):
    train_cgan(images(4), [0] * 4, PARAMS, epochs=100, out_dir=tmp_path, sample_every=1000)
    assert "epoch 100: gen" in capsys.readouterr().out
