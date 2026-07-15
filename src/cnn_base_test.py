import os
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import numpy as np
import pytest
import tensorflow as tf

from cnn_base import (
    _CustomEarlyStoppingAndSave,
    _define_model,
    _kfold_validation,
    _plot_confusion_matrix,
    _plot_epoch_train_val_acc,
    _plot_kfold_val_acc,
    _sum_confusion_matrices,
    train_model,
)


@pytest.fixture
def callback_and_mock_model() -> tuple[_CustomEarlyStoppingAndSave, MagicMock]:
    """
    Fixture to provide mock model and fresh callback instance for each test
    """
    mock_model = MagicMock()
    mock_model.stop_training = False
    mock_model.get_weights.return_value = [np.array([1.0])]
    mock_model.set_weights.return_value = None

    callback = _CustomEarlyStoppingAndSave(patience=2, warmup=1, divergence_threshold=0.1)
    callback.set_model(mock_model)
    return callback, mock_model


class TestCustomEarlyStoppingAndSave:

    def test_initialization(self):
        callback = _CustomEarlyStoppingAndSave(patience=5, divergence_threshold=0.2, warmup=10)
        assert callback.patience == 5
        assert callback.divergence_threshold == 0.2
        assert callback.warmup == 10
        assert callback.best_accuracy == 0
        assert callback.wait == 0
        assert callback.best_weights is None

    def test_warmup_period(self, callback_and_mock_model):
        callback, mock_model = callback_and_mock_model
        logs = {"accuracy": 0.9, "val_accuracy": 0.8}

        callback.on_epoch_end(epoch=0, logs=logs)

        assert not mock_model.stop_training
        assert callback.wait == 0
        assert callback.best_accuracy == 0

    def test_divergence_stop(self, callback_and_mock_model):
        callback, mock_model = callback_and_mock_model
        logs = {"accuracy": 0.9, "val_accuracy": 0.75}

        callback.on_epoch_end(epoch=1, logs=logs)  # Epochs are 0 indexed, so epoch 0 is warmup.
        assert mock_model.stop_training

    def test_improvement_resets_wait_and_saves_weights(self, callback_and_mock_model):
        callback, mock_model = callback_and_mock_model
        callback.best_accuracy = 0.8
        callback.wait = 1
        new_weights = [np.array([2.0])]
        mock_model.get_weights.return_value = new_weights
        logs = {"accuracy": 0.86, "val_accuracy": 0.85}  # Larger than 0.8

        callback.on_epoch_end(epoch=1, logs=logs)

        assert callback.best_accuracy == 0.85  # new best accuracy
        assert callback.wait == 0
        assert callback.best_weights == new_weights
        mock_model.get_weights.assert_called_once()
        assert not mock_model.stop_training

    def test_no_improvement_increments_wait(self, callback_and_mock_model):
        callback, mock_model = callback_and_mock_model
        callback.best_accuracy = 0.8
        callback.wait = 0
        logs = {"accuracy": 0.81, "val_accuracy": 0.79}

        callback.on_epoch_end(epoch=1, logs=logs)

        assert callback.wait == 1
        assert callback.best_accuracy == 0.8
        assert not mock_model.stop_training

    def test_patience_stop_and_restores_weights(self, callback_and_mock_model):
        callback, mock_model = callback_and_mock_model
        callback.best_accuracy = 0.8
        callback.best_weights = [np.array([1.0])]
        callback.wait = 1
        logs = {"accuracy": 0.81, "val_accuracy": 0.79}

        callback.on_epoch_end(epoch=1, logs=logs)

        assert callback.wait == 2
        assert mock_model.stop_training
        mock_model.set_weights.assert_called_once_with(callback.best_weights)

    def test_on_train_end_restores_weights(self, callback_and_mock_model):
        callback, mock_model = callback_and_mock_model
        best_weights = [np.array([5.0])]
        callback.best_weights = best_weights

        callback.on_train_end()

        mock_model.set_weights.assert_called_once_with(best_weights)

    def test_on_train_end_no_weights(self, callback_and_mock_model):
        callback, mock_model = callback_and_mock_model
        callback.best_weights = None

        callback.on_train_end()

        mock_model.set_weights.assert_not_called()


@pytest.fixture
def mock_training_data():
    """
    Provides mock images, labels, and a set of randomly selected hyperparameters
    for testing a model training process.
    """
    num_samples = 32
    images = np.random.rand(num_samples, 64, 64, 3)
    labels = np.arange(num_samples) % 7

    selected_params = {
        "num_filters": 8,
        "filter_size": 3,
        "learning_rate": 0.001,
        "epochs": 2,
        "k": 2,
        "num_layers": 3,
        "pooling_size": 2,
        "activation_function": "relu",
        "batch_size": 8,
        "reg": "L2",
        "opt": "Adam",
        "dropout": 0.2,
    }

    return images, labels, selected_params


class TestDefineModel:

    def test_valid_inputs(self, mock_training_data):
        images, labels, selected_params = mock_training_data
        img_size = images.shape[2]
        class_labels = sorted(np.unique(labels))

        model = _define_model(selected_params, img_size, class_labels)

        assert isinstance(model, tf.keras.Model)

        expected_input_shape = (None, img_size, img_size, 3)
        assert model.input_shape == expected_input_shape

        expected_output_shape = (None, len(class_labels))
        assert model.output_shape == expected_output_shape

        selected_params["reg"] = "L1"
        _define_model(selected_params, img_size, class_labels)

        selected_params["reg"] = None
        _define_model(selected_params, img_size, class_labels)


class TestKfoldValidation:

    def test_valid_inputs(self, mock_training_data):
        images, labels, selected_params = mock_training_data

        results = _kfold_validation(images, labels, selected_params)

        assert isinstance(results, dict)

        expected_keys = [
            "validation_accuracies_fold",
            "train_accuracies_fold",
            "validation_accuracies_epoch",
            "train_accuracies_epoch",
            "validation_nearmiss_fold",
            "confusion_matrices",
            "best_model",
            "best_epoch",
        ]
        assert all(key in results for key in expected_keys)

        for key, value in results.items():
            if key == "best_model":
                assert isinstance(value, tf.keras.Model)
            elif key == "best_epoch":
                assert isinstance(value, (int, np.integer))
                assert value >= 0  # Assumes epoch numbers start from 0
            else:
                # This covers all the list-based results
                assert len(value) > 0

        selected_params["opt"] = "SGD"
        _kfold_validation(images, labels, selected_params)

        # Test Momentum optimizer path
        selected_params["opt"] = "Momentum"
        _kfold_validation(images, labels, selected_params)

        # Test RMSProp optimizer path
        selected_params["opt"] = "RMSProp"
        _kfold_validation(images, labels, selected_params)


class TestPlotConfusionMatrix:
    def test_runs_without_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(plt, "show", lambda: None)  # Disable plt.show() for this test.

        mock_matrix = np.array([[10, 2], [3, 12]])
        mock_labels = np.array([0, 1])
        mock_axis_labels = ("True", "Predicted")
        output_path = tmp_path / "test.png"

        _plot_confusion_matrix(mock_matrix, mock_labels, mock_axis_labels, output=str(output_path))

        assert os.path.exists(output_path)


class TestSumConfusionMatrices:
    def test_valid_summation(self):
        mock_labels = [0, 1, 2]
        mock_matrices_data = [
            {"ground_truth_values": [0, 1, 2], "predicted_values": [0, 1, 1]},
            {"ground_truth_values": [0, 1, 2], "predicted_values": [0, 2, 2]},
        ]

        result = _sum_confusion_matrices(mock_labels, mock_matrices_data)

        # Correctly calculated expected result
        cm1 = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 0]])
        cm2 = np.array([[1, 0, 0], [0, 0, 1], [0, 0, 1]])
        expected_sum = cm1 + cm2

        np.testing.assert_array_equal(result, expected_sum)
        assert result.dtype == np.int32


class TestPlotKfoldValAcc:
    def test_runs_without_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(plt, "show", lambda: None)

        # Use the temporary path as a prefix for the model name
        output_prefix = tmp_path / "test_model_kfold"
        expected_output_file = str(output_prefix) + "scatter_plot.png"

        mock_cycles = 5
        mock_accuracies = [0.85, 0.88, 0.86, 0.90, 0.87]

        # Pass the prefix as the model_name
        _plot_kfold_val_acc(str(output_prefix), mock_cycles, mock_accuracies)

        assert os.path.exists(expected_output_file)


class TestPlotEpochTrainValAcc:
    def test_runs_without_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(plt, "show", lambda: None)

        # Use the temporary path as a prefix for the model name
        output_prefix = tmp_path / "test_model_epoch"
        expected_output_file = str(output_prefix) + "accuracies.png"

        mock_train_acc = [[0.8, 0.82, 0.85], [0.81, 0.83]]
        mock_val_acc = [[0.78, 0.80, 0.81], [0.79, 0.81]]
        mock_k = 2

        # Pass the prefix as the model_name
        _plot_epoch_train_val_acc(str(output_prefix), mock_train_acc, mock_val_acc, mock_k)

        assert os.path.exists(expected_output_file)


class TestTrainModel:
    def test_valid_input(self, mock_training_data, monkeypatch, tmp_path):
        monkeypatch.setattr(plt, "show", lambda: None)
        monkeypatch.chdir(tmp_path)

        images, labels, selected_params = mock_training_data

        slurm_array, index = 1, 1

        train_model(slurm_array, index, selected_params, images, labels)
