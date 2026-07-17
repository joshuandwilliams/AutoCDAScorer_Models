import pytest
import tensorflow as tf

from model_builders import CNNModelBuilder, ViTModelBuilder, _make_optimizer


def _params(**overrides):
    params = {
        "num_filters": 8,
        "filter_size": 3,
        "learning_rate": 0.001,
        "num_layers": 2,
        "pooling_size": 2,
        "activation_function": "relu",
        "reg": "L2",
        "reg_strength": 0.001,
        "opt": "Adam",
        "dropout": 0.2,
    }
    params.update(overrides)
    return params


def _vit_params(**overrides):
    params = {
        "patch_size": 8,
        "projection_dim": 32,
        "num_heads": 4,
        "num_transformer_layers": 2,
        "mlp_ratio": 2,
        "learning_rate": 0.001,
        "activation_function": "gelu",
        "reg": None,
        "reg_strength": 0.0001,
        "opt": "Adam",
        "dropout": 0.1,
    }
    params.update(overrides)
    return params


class TestCNNModelBuilder:
    def test_builds_compiled_model_with_correct_shapes(self):
        model = CNNModelBuilder().build(_params(), input_shape=(64, 64, 3), num_classes=7)
        assert isinstance(model, tf.keras.Model)
        assert model.input_shape == (None, 64, 64, 3)
        assert model.output_shape == (None, 7)
        # Compiled: optimizer and loss are set.
        assert model.optimizer is not None
        assert model.loss == "categorical_crossentropy"

    def test_regularisation_and_no_reg_paths(self):
        for reg in (None, "L1", "L2"):
            model = CNNModelBuilder().build(_params(reg=reg), (64, 64, 3), 7)
            assert isinstance(model, tf.keras.Model)

    def test_dropout_zero_omits_dropout_layer(self):
        model = CNNModelBuilder().build(_params(dropout=0.0), (64, 64, 3), 7)
        assert not any(isinstance(layer, tf.keras.layers.Dropout) for layer in model.layers)


class TestIsValid:
    def test_shallow_small_kernel_is_valid(self):
        assert CNNModelBuilder().is_valid(_params(filter_size=3, num_layers=2), (64, 64, 3)) is True

    def test_deep_large_kernel_collapses_and_is_invalid(self):
        # filter_size 7 x num_layers 4 shrinks a 64x64 map below the kernel (the case
        # that was skipped at runtime before).
        assert (
            CNNModelBuilder().is_valid(_params(filter_size=7, num_layers=4), (64, 64, 3)) is False
        )

    def test_an_invalid_config_would_raise_if_built(self):
        with pytest.raises(ValueError):
            CNNModelBuilder().build(_params(filter_size=7, num_layers=4), (64, 64, 3), 7)


class TestViTModelBuilder:
    def test_builds_compiled_model_with_correct_shapes(self):
        model = ViTModelBuilder().build(_vit_params(), input_shape=(64, 64, 3), num_classes=7)
        assert isinstance(model, tf.keras.Model)
        assert model.input_shape == (None, 64, 64, 3)
        assert model.output_shape == (None, 7)
        # Compiled softmax + categorical loss: same contract the harness relies on.
        assert model.optimizer is not None
        assert model.loss == "categorical_crossentropy"

    def test_functional_model_round_trips_through_keras_save(self, tmp_path):
        # The best fold is saved as .keras; a functional ViT with the registered
        # custom layers must reload without needing custom_objects passed in.
        model = ViTModelBuilder().build(_vit_params(num_transformer_layers=1), (64, 64, 3), 7)
        path = tmp_path / "model.keras"
        model.save(path)
        reloaded = tf.keras.models.load_model(path)
        assert reloaded.output_shape == (None, 7)

    def test_regularisation_and_no_reg_paths(self):
        for reg in (None, "L1", "L2"):
            model = ViTModelBuilder().build(_vit_params(reg=reg), (64, 64, 3), 7)
            assert isinstance(model, tf.keras.Model)

    def test_dropout_zero_omits_dropout_layer(self):
        model = ViTModelBuilder().build(_vit_params(dropout=0.0), (64, 64, 3), 7)
        assert not any(isinstance(layer, tf.keras.layers.Dropout) for layer in model.layers)


class TestViTIsValid:
    def test_divisible_patch_and_heads_is_valid(self):
        assert (
            ViTModelBuilder().is_valid(_vit_params(patch_size=8, num_heads=4), (64, 64, 3)) is True
        )

    def test_patch_not_dividing_image_is_invalid(self):
        # 64 is not divisible by 5, so the patch grid cannot tile the image.
        assert ViTModelBuilder().is_valid(_vit_params(patch_size=5), (64, 64, 3)) is False

    def test_projection_dim_not_divisible_by_heads_is_invalid(self):
        assert (
            ViTModelBuilder().is_valid(_vit_params(projection_dim=32, num_heads=5), (64, 64, 3))
            is False
        )


class TestMakeOptimizer:
    @pytest.mark.parametrize(
        "opt", ["Adam", "AdamW", "SGD", "Momentum", "RMSProp", "Nadam", "Adamax"]
    )
    def test_known_optimizers(self, opt):
        assert _make_optimizer(opt, 0.001) is not None

    def test_unknown_optimizer_raises(self):
        with pytest.raises(ValueError):
            _make_optimizer("NotAnOptimizer", 0.001)
