import pytest
import tensorflow as tf

from model_builders import CNNModelBuilder, _make_optimizer


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


class TestMakeOptimizer:
    @pytest.mark.parametrize("opt", ["Adam", "SGD", "Momentum", "RMSProp", "Nadam", "Adamax"])
    def test_known_optimizers(self, opt):
        assert _make_optimizer(opt, 0.001) is not None

    def test_unknown_optimizer_raises(self):
        with pytest.raises(ValueError):
            _make_optimizer("NotAnOptimizer", 0.001)
