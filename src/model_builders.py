"""Pluggable model builders for the random-search harness.

A ``ModelBuilder`` maps a hyperparameter dict to a *compiled* Keras model. This
decouples the search harness from the architecture: swap ``CNNModelBuilder`` for
``ViTModelBuilder`` (both defined below) and the rest of the pipeline is unchanged.

Every builder returns a **softmax** model compiled with ``categorical_crossentropy``,
because :class:`~random_search.RandomSearch` feeds one-hot labels and reads accuracy
from ``model.evaluate`` -- keeping that contract is what lets one harness score any
architecture.

Expected params keys, by builder:
    CNNModelBuilder: num_filters, filter_size, num_layers, pooling_size,
        activation_function, dropout, reg ("L1"|"L2"|None), reg_strength, opt,
        learning_rate
    ViTModelBuilder: patch_size, projection_dim, num_heads, num_transformer_layers,
        mlp_ratio, activation_function, dropout, reg ("L1"|"L2"|None), reg_strength,
        opt, learning_rate
(``epochs`` and ``batch_size`` are consumed by the search harness, not the builder.)
"""

from abc import ABC, abstractmethod

import tensorflow as tf
from tensorflow.keras import Input, layers


def _make_optimizer(opt: str, learning_rate: float):
    """Return a fresh Keras optimizer for the given name and learning rate."""
    optimizers = {
        "Adam": lambda: tf.keras.optimizers.Adam(learning_rate=learning_rate),
        "AdamW": lambda: tf.keras.optimizers.AdamW(learning_rate=learning_rate),
        "SGD": lambda: tf.keras.optimizers.SGD(learning_rate=learning_rate),
        "Momentum": lambda: tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9),
        "RMSProp": lambda: tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
        "Nadam": lambda: tf.keras.optimizers.Nadam(learning_rate=learning_rate),
        "Adamax": lambda: tf.keras.optimizers.Adamax(learning_rate=learning_rate),
    }
    if opt not in optimizers:
        raise ValueError(f"Unknown optimizer '{opt}'. Options: {sorted(optimizers)}")
    return optimizers[opt]()


def _make_regularizer(reg_type, reg_strength: float):
    """Return an L1/L2 kernel regularizer for ``reg_type`` (or None for no reg)."""
    if reg_type == "L1":
        return tf.keras.regularizers.l1(reg_strength)
    if reg_type == "L2":
        return tf.keras.regularizers.l2(reg_strength)
    return None


class ModelBuilder(ABC):
    """Builds a compiled ``tf.keras.Model`` from a hyperparameter dict.

    Subclass and implement :meth:`build` to plug a new architecture into
    ``RandomSearch`` without changing the search code.
    """

    @abstractmethod
    def build(self, params: dict, input_shape: tuple, num_classes: int) -> tf.keras.Model:
        """Return a compiled model for ``params`` given the input shape and class count."""
        raise NotImplementedError

    def is_valid(self, params: dict, input_shape: tuple) -> bool:
        """Whether ``params`` yields a constructible model for ``input_shape``.

        Override when some hyperparameter combinations cannot be built (the default
        assumes every combination is valid). ``RandomSearch`` samples only configs for
        which this returns True, so the requested number of models always run.
        """
        return True


class CNNModelBuilder(ModelBuilder):
    """The baseline sequential CNN: stacked Conv2D + MaxPool blocks -> Flatten -> softmax.

    Deliberately plain (constant filter count, ``valid`` padding, no BatchNorm) so it
    serves as a clean reference point for the later augmented / transfer / ensemble
    studies. The regularisation *strength* is exposed as a hyperparameter
    (``reg_strength``) rather than hard-coded.
    """

    def is_valid(self, params: dict, input_shape: tuple) -> bool:
        """False if the stacked ``valid``-padding conv/pool blocks would shrink the
        feature map below the kernel size (Keras cannot build such a model).

        Mirrors the layer sequence in :meth:`build`: each block does a valid conv
        (``h -> h - filter_size + 1``) then a max-pool (``h -> h // pooling_size``).
        """
        h, w = input_shape[0], input_shape[1]
        filter_size = params["filter_size"]
        pooling_size = params["pooling_size"]
        for _ in range(params["num_layers"]):
            h, w = h - filter_size + 1, w - filter_size + 1
            if h < 1 or w < 1:
                return False
            h, w = h // pooling_size, w // pooling_size
            if h < 1 or w < 1:
                return False
        return True

    def build(self, params: dict, input_shape: tuple, num_classes: int) -> tf.keras.Model:
        regularizer = _make_regularizer(params.get("reg"), params.get("reg_strength", 0.01))

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


@tf.keras.utils.register_keras_serializable(package="autocdascorer_models")
class ClassToken(layers.Layer):
    """Prepend a single learnable ``[CLS]`` token to a sequence of patch embeddings.

    Turns ``(batch, num_patches, dim)`` into ``(batch, num_patches + 1, dim)``. The
    token's final-layer representation is what :class:`ExtractClassToken` reads for
    classification, exactly as in the standard ViT. The token is a trainable weight,
    so it is created in ``build`` (which knows ``dim``) and needs no config args.
    """

    def build(self, input_shape):
        self.cls = self.add_weight(
            shape=(1, 1, input_shape[-1]), initializer="zeros", trainable=True, name="cls_token"
        )

    def call(self, x):
        batch = tf.shape(x)[0]
        cls = tf.tile(self.cls, [batch, 1, 1])  # broadcast the token across the batch
        return tf.concat([cls, x], axis=1)

    def compute_output_shape(self, input_shape):
        length = None if input_shape[1] is None else input_shape[1] + 1
        return (input_shape[0], length, input_shape[2])


@tf.keras.utils.register_keras_serializable(package="autocdascorer_models")
class AddPositionEmbedding(layers.Layer):
    """Add a learnable position embedding to every token (CLS token included).

    The sequence length is fixed once patch size and image size are chosen, so the
    embedding table is a single trainable weight of shape ``(1, seq_len, dim)`` built
    lazily from the input shape.
    """

    def build(self, input_shape):
        self.pos = self.add_weight(
            shape=(1, input_shape[1], input_shape[-1]),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
            name="pos_embedding",
        )

    def call(self, x):
        return x + self.pos


@tf.keras.utils.register_keras_serializable(package="autocdascorer_models")
class ExtractClassToken(layers.Layer):
    """Slice out the ``[CLS]`` token (position 0) to feed the classification head."""

    def call(self, x):
        return x[:, 0]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])


class ViTModelBuilder(ModelBuilder):
    """A compact Vision Transformer for the small 64x64 CDA crops.

    Built with the functional API (not a subclassed ``tf.keras.Model``) so the harness
    can ``model.save('model.keras')`` the best fold cleanly. The pipeline is the
    textbook ViT: a strided-conv patch embedding (a linear projection of each
    non-overlapping patch), a prepended learnable ``[CLS]`` token, learnable position
    embeddings, a stack of pre-norm transformer encoder blocks, then an MLP head on the
    ``[CLS]`` token. Output is softmax + ``categorical_crossentropy`` so it drops into
    the same :class:`~random_search.RandomSearch` harness as :class:`CNNModelBuilder`.
    """

    def is_valid(self, params: dict, input_shape: tuple) -> bool:
        """False for configs that cannot form a valid patch grid or attention split.

        - ``patch_size`` must tile the image exactly (no dropped border pixels), which
          also guarantees at least one patch.
        - ``projection_dim`` must be divisible by ``num_heads`` so each head gets an
          integer ``key_dim`` of at least 1.
        """
        h, w = input_shape[0], input_shape[1]
        patch_size = params["patch_size"]
        if patch_size < 1 or h % patch_size != 0 or w % patch_size != 0:
            return False
        if (h // patch_size) * (w // patch_size) < 1:
            return False
        if params["projection_dim"] % params["num_heads"] != 0:
            return False
        return True

    def build(self, params: dict, input_shape: tuple, num_classes: int) -> tf.keras.Model:
        patch_size = params["patch_size"]
        projection_dim = params["projection_dim"]
        num_heads = params["num_heads"]
        num_layers = params["num_transformer_layers"]
        mlp_ratio = params.get("mlp_ratio", 2)
        activation = params.get("activation_function", "gelu")
        dropout = params.get("dropout", 0.0)
        regularizer = _make_regularizer(params.get("reg"), params.get("reg_strength", 0.01))

        h, w = input_shape[0], input_shape[1]
        num_patches = (h // patch_size) * (w // patch_size)
        key_dim = projection_dim // num_heads

        inputs = Input(shape=input_shape)
        # Patch embedding: a stride-``patch_size`` conv is a linear projection of each
        # non-overlapping patch, giving one ``projection_dim`` token per patch.
        x = layers.Conv2D(
            projection_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
            kernel_regularizer=regularizer,
        )(inputs)
        x = layers.Reshape((num_patches, projection_dim))(x)
        x = ClassToken()(x)
        x = AddPositionEmbedding()(x)
        if dropout > 0:
            x = layers.Dropout(dropout)(x)

        for _ in range(num_layers):
            # Pre-norm transformer encoder block with residual connections.
            y = layers.LayerNormalization(epsilon=1e-6)(x)
            attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout)(
                y, y
            )
            x = layers.Add()([attn, x])
            y = layers.LayerNormalization(epsilon=1e-6)(x)
            y = layers.Dense(
                projection_dim * mlp_ratio, activation=activation, kernel_regularizer=regularizer
            )(y)
            if dropout > 0:
                y = layers.Dropout(dropout)(y)
            y = layers.Dense(projection_dim, kernel_regularizer=regularizer)(y)
            x = layers.Add()([y, x])

        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = ExtractClassToken()(x)
        if dropout > 0:
            x = layers.Dropout(dropout)(x)
        outputs = layers.Dense(num_classes, activation="softmax", kernel_regularizer=regularizer)(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=_make_optimizer(params["opt"], params["learning_rate"]),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model
