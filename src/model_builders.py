"""Pluggable model builders for the random-search harness.

A ``ModelBuilder`` maps a hyperparameter dict to a *compiled* Keras model. This
decouples the search harness from the architecture: swap ``CNNModelBuilder`` for,
say, a ``ViTModelBuilder`` and the rest of the pipeline is unchanged.

Expected params keys used here:
    num_filters, filter_size, num_layers, pooling_size, activation_function,
    dropout, reg ("L1"|"L2"|None), reg_strength, opt, learning_rate
(``epochs`` and ``batch_size`` are consumed by the search harness, not the builder.)
"""

from abc import ABC, abstractmethod

import tensorflow as tf
from tensorflow.keras import Input, layers


def _make_optimizer(opt: str, learning_rate: float):
    """Return a fresh Keras optimizer for the given name and learning rate."""
    optimizers = {
        "Adam": lambda: tf.keras.optimizers.Adam(learning_rate=learning_rate),
        "SGD": lambda: tf.keras.optimizers.SGD(learning_rate=learning_rate),
        "Momentum": lambda: tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9),
        "RMSProp": lambda: tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
        "Nadam": lambda: tf.keras.optimizers.Nadam(learning_rate=learning_rate),
        "Adamax": lambda: tf.keras.optimizers.Adamax(learning_rate=learning_rate),
    }
    if opt not in optimizers:
        raise ValueError(f"Unknown optimizer '{opt}'. Options: {sorted(optimizers)}")
    return optimizers[opt]()


class ModelBuilder(ABC):
    """Builds a compiled ``tf.keras.Model`` from a hyperparameter dict.

    Subclass and implement :meth:`build` to plug a new architecture into
    ``RandomSearch`` without changing the search code.
    """

    @abstractmethod
    def build(self, params: dict, input_shape: tuple, num_classes: int) -> tf.keras.Model:
        """Return a compiled model for ``params`` given the input shape and class count."""
        raise NotImplementedError


class CNNModelBuilder(ModelBuilder):
    """The baseline sequential CNN: stacked Conv2D + MaxPool blocks -> Flatten -> softmax.

    Deliberately plain (constant filter count, ``valid`` padding, no BatchNorm) so it
    serves as a clean reference point for the later augmented / transfer / ensemble
    studies. The regularisation *strength* is exposed as a hyperparameter
    (``reg_strength``) rather than hard-coded.
    """

    def build(self, params: dict, input_shape: tuple, num_classes: int) -> tf.keras.Model:
        reg_type = params.get("reg")
        reg_strength = params.get("reg_strength", 0.01)
        if reg_type == "L1":
            regularizer = tf.keras.regularizers.l1(reg_strength)
        elif reg_type == "L2":
            regularizer = tf.keras.regularizers.l2(reg_strength)
        else:
            regularizer = None

        num_filters = params["num_filters"]
        filter_size = params["filter_size"]
        activation = params["activation_function"]
        pooling_size = params["pooling_size"]
        num_layers = params["num_layers"]
        dropout = params.get("dropout", 0.0)

        model = tf.keras.Sequential()
        model.add(Input(shape=input_shape))
        for _ in range(num_layers):
            model.add(
                layers.Conv2D(
                    num_filters,
                    filter_size,
                    activation=activation,
                    kernel_regularizer=regularizer,
                )
            )
            model.add(layers.MaxPooling2D(pool_size=(pooling_size, pooling_size)))
        model.add(layers.Flatten())
        if dropout > 0:
            model.add(layers.Dropout(dropout))
        model.add(layers.Dense(num_classes, activation="softmax", kernel_regularizer=regularizer))

        model.compile(
            optimizer=_make_optimizer(params["opt"], params["learning_rate"]),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model
