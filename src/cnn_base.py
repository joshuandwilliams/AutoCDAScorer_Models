import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib.patches import Rectangle
from sklearn.metrics import confusion_matrix
from tensorflow.keras import Input, layers
from tensorflow.keras.callbacks import Callback


class _CustomEarlyStoppingAndSave(Callback):
    def __init__(self, patience=4, divergence_threshold=0.15, warmup=15):
        super().__init__()
        self.patience = patience
        self.divergence_threshold = divergence_threshold
        self.warmup = warmup
        self.best_weights = None
        self.best_accuracy = 0
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        # Patience warm-up period
        if epoch < self.warmup:
            return

        train_accuracy = logs.get("accuracy")
        val_accuracy = logs.get("val_accuracy")

        # Check divergence
        if abs(train_accuracy - val_accuracy) > self.divergence_threshold:
            print(f"Stopping training due to divergence > {self.divergence_threshold}")
            self.model.stop_training = True
            return

        # Check val_acc improvement
        if val_accuracy > self.best_accuracy:
            self.best_accuracy = val_accuracy
            self.wait = 0
            self.best_weights = self.model.get_weights()  # Save best weights
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(
                    f"Stopping training due to no improvement in validation accuracy for {self.patience} epochs"
                )
                self.model.stop_training = True
                # Restore the best weights
                if self.best_weights is not None:
                    self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        # If training stops for any reason, restore the best weights
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)


def _define_model(
    selected_params: dict[str, Any], img_size: int, class_labels: list[str]
) -> tf.keras.Model:
    """
    Builds and returns a sequential CNN model based on specified hyperparameters.

    Parameters:
    - selected_params: A dictionary containing all the hyperparameters for the model.
    - img_size: The height and width of the input images (e.g., 64 for a 64x64 image).
    - class_labels: A list of the string names for all classes, used to define the output layer size.

    Returns:
    - A TensorFlow Keras Sequential model, ready for compilation.
    """
    reg = selected_params["reg"]
    num_filters = selected_params["num_filters"]
    filter_size = selected_params["filter_size"]
    activation_function = selected_params["activation_function"]
    pooling_size = selected_params["pooling_size"]
    num_layers = selected_params["num_layers"]
    dropout = selected_params["dropout"]

    if reg == "L1":
        regularization = tf.keras.regularizers.l1(0.01)
    elif reg == "L2":
        regularization = tf.keras.regularizers.l2(0.01)
    else:
        regularization = None

    model = tf.keras.Sequential()

    model.add(Input(shape=(img_size, img_size, 3)))

    model.add(
        layers.Conv2D(
            num_filters,
            filter_size,
            activation=activation_function,
            kernel_regularizer=regularization,
        )
    )
    model.add(layers.MaxPooling2D(pool_size=(pooling_size, pooling_size)))

    for _ in range(num_layers - 1):
        model.add(
            layers.Conv2D(
                num_filters,
                filter_size,
                activation=activation_function,
                kernel_regularizer=regularization,
            )
        )
        model.add(layers.MaxPooling2D(pool_size=(pooling_size, pooling_size)))
    model.add(layers.Flatten())

    # Optional dropout
    if dropout > 0:
        model.add(layers.Dropout(dropout))

    model.add(
        layers.Dense(len(class_labels), activation="softmax", kernel_regularizer=regularization)
    )

    return model


def define_model2(
    selected_params: dict[str, Any], img_size: int, class_labels: list[str]
) -> tf.keras.Model:
    """
    Builds a CNN optimized for simple feature detection (e.g., spots/blobs).

    Key Architectural Decisions:
    1. He Initialization: For proper scaling of ReLU activations (He et al., 2015).
    2. Batch Normalization: To stabilize training on high-contrast data (Ioffe & Szegedy, 2015).
    3. Global Average Pooling: To enforce translation invariance and reduce parameters (Lin et al., 2013).
    """
    # 1. Extract Hyperparameters
    num_filters = selected_params["num_filters"]
    filter_size = selected_params["filter_size"]
    dropout = selected_params["dropout"]
    num_layers = selected_params["num_layers"]
    reg_type = selected_params["reg"]
    reg_strength = selected_params.get("reg_strength", 0.001)  # Lower default for simple data

    # 2. Configure Regularizer
    if reg_type == "L1":
        regularizer = tf.keras.regularizers.l1(reg_strength)
    elif reg_type == "L2":
        regularizer = tf.keras.regularizers.l2(reg_strength)
    else:
        regularizer = None

    # 3. Build Model
    model = tf.keras.Sequential()
    model.add(Input(shape=(img_size, img_size, 3)))

    for i in range(num_layers):
        # Scale filters: Start small, increase depth.
        # For simple data, we might not need to double every layer, but it is standard.
        current_filters = num_filters * (2**i)

        model.add(
            layers.Conv2D(
                current_filters,
                filter_size,
                padding="same",  # Keep spatial resolution
                kernel_initializer="he_normal",  # Best for ReLU
                kernel_regularizer=regularizer,
                use_bias=False,  # Bias redundant with BN
            )
        )

        model.add(layers.BatchNormalization())
        model.add(layers.Activation("relu"))

        # Max Pooling reduces the image size.
        # For very small inputs, ensure this doesn't reduce dim to 0.
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # 4. Output Block
    # Replace Flatten -> Dense with GAP
    model.add(layers.GlobalAveragePooling2D())

    if dropout > 0:
        model.add(layers.Dropout(dropout))

    model.add(layers.Dense(len(class_labels), activation="softmax"))

    return model


def _kfold_validation(
    images: np.ndarray, labels: np.ndarray, selected_params: dict[str, Any]
) -> dict[str, Any]:
    """
    Manual k-fold cross validation for a deep learning model.

    Parameters:
    - images: The images for training and validation
    - labels: The labels for training and validation
    - selected_params: A dictionary containing all the hyperparameters for the model run

    Returns:
    - A dictionary containing the results of the cross-validation, including accuracies, the best model, and confusion matrices.
    """
    # Unpack hyperparameters
    k = selected_params["k"]
    learning_rate = selected_params["learning_rate"]
    epochs = selected_params["epochs"]
    batch_size = selected_params["batch_size"]
    opt = selected_params["opt"]

    # Initialise lists to store results
    val_accuracies_fold = []
    train_accuracies_fold = []
    val_accuracies_epoch = []
    train_accuracies_epoch = []
    val_nearmisses_fold = []
    val_nearmisses_epoch = []
    confusion_matrices = []

    max_val_acc = 0
    best_model = None
    best_epoch = None

    fold_size = len(images) // k
    class_labels = sorted(np.unique(labels))
    img_size = images.shape[2]

    for fold in range(k):
        print(f"Fold: {fold + 1} of {k}")

        model = _define_model(selected_params, img_size, class_labels)

        # Define current fold
        start_index = fold * fold_size
        end_index = start_index + fold_size

        val_images = images[start_index:end_index]
        val_labels = labels[start_index:end_index]
        train_images = np.concatenate((images[:start_index], images[end_index:]), axis=0)
        train_labels = np.concatenate((labels[:start_index], labels[end_index:]), axis=0)

        train_labels_encoded = tf.keras.utils.to_categorical(train_labels, len(class_labels))
        val_labels_encoded = tf.keras.utils.to_categorical(val_labels, len(class_labels))

        if opt == "Adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif opt == "SGD":
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        elif opt == "Momentum":
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif opt == "RMSProp":
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

        class NearMissCallback(Callback):
            def __init__(self, val_images, val_labels):
                super().__init__()
                self.val_images = val_images
                self.val_labels = val_labels
                self.nearmiss_history = []

            def on_epoch_end(self, epoch, logs=None):
                val_preds = np.argmax(self.model.predict(self.val_images, verbose=0), axis=1)
                near_miss_correct = [
                    pred >= max(0, true - 1) and pred <= min(6, true + 1)
                    for pred, true in zip(val_preds, self.val_labels, strict=False)
                ]
                self.nearmiss_history.append(sum(near_miss_correct) / len(self.val_labels))

        custom_early_stopping_save = _CustomEarlyStoppingAndSave(
            patience=4, divergence_threshold=0.15, warmup=15
        )
        nearmiss_callback = NearMissCallback(val_images, val_labels)
        history = model.fit(
            train_images,
            train_labels_encoded,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(val_images, val_labels_encoded),
            verbose=0,
            callbacks=[custom_early_stopping_save, nearmiss_callback],
        )

        # Store train/val accuracies per fold
        _, val_accuracy_fold = model.evaluate(val_images, val_labels_encoded, verbose=0)
        _, train_accuracy_fold = model.evaluate(train_images, train_labels_encoded, verbose=0)
        val_accuracies_fold.append(val_accuracy_fold)
        train_accuracies_fold.append(train_accuracy_fold)

        # Validation near-miss per fold
        val_preds = np.argmax(model.predict(val_images), axis=1)
        val_true = np.argmax(val_labels_encoded, axis=1)

        # Check near-miss correctness (±1)
        near_miss_correct = []
        for pred, true in zip(val_preds, val_true, strict=False):
            if pred >= max(0, true - 1) and pred <= min(6, true + 1):
                near_miss_correct.append(True)
            else:
                near_miss_correct.append(False)

        val_nearmiss_fold = sum(near_miss_correct) / len(val_true)
        val_nearmisses_fold.append(val_nearmiss_fold)

        val_nearmisses_epoch.append(nearmiss_callback.nearmiss_history)

        # Tracking best model across all folds.
        if val_accuracy_fold > max_val_acc:
            print(f"New Best Validation Accuracy: {val_accuracy_fold}")
            max_val_acc = val_accuracy_fold
            best_model = model
            best_epoch = np.argmax(history.history["val_accuracy"]) + 1  # 0 indexed so +1.

        # Store train/val accuracies for each epoch of current fold.
        train_accuracies_epoch.append(history.history["accuracy"])
        val_accuracies_epoch.append(history.history["val_accuracy"])

        # Calculate predictions on the validation set
        val_pred = model.predict(val_images, verbose=0)
        val_pred_labels = np.argmax(val_pred, axis=1)

        # Generate the confusion matrix for the current fold
        confusion_matrices.append(
            {"predicted_values": val_pred_labels, "ground_truth_values": val_labels}
        )

    results = {
        "validation_accuracies_fold": val_accuracies_fold,
        "train_accuracies_fold": train_accuracies_fold,
        "validation_accuracies_epoch": val_accuracies_epoch,
        "train_accuracies_epoch": train_accuracies_epoch,
        "validation_nearmiss_fold": val_nearmisses_fold,
        "validation_nearmiss_epoch": val_nearmisses_epoch,
        "confusion_matrices": confusion_matrices,
        "best_model": best_model,
        "best_epoch": best_epoch,
    }

    return results


def _plot_confusion_matrix(
    conf_matrix: np.ndarray,
    labels: np.ndarray,
    axis_labels: tuple[str, str],
    output: str = "confusion_matrix.png",
) -> None:
    """
    Plots a confusion matrix with labels, and highlights the diagonal cells with black borders.

    Parameters:
    -----------
    conf_matrix : np.ndarray
            The confusion matrix to plot.
    labels : np.ndarray
            Array of labels for the matrix axes.
    axis_labels : tuple[str, str]
            A tuple containing the labels for the X and Y axes, e.g., ('Predicted', 'Actual').

    Returns:
    --------
    None
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation="nearest", cmap=plt.cm.Greens)
    cbar = plt.colorbar()
    cbar.set_label("Frequency")

    labels = labels.astype(int)
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)

    # Adding text annotations and black borders for diagonal cells
    ax = plt.gca()  # Get current axis
    for i in range(len(conf_matrix)):
        for j in range(len(conf_matrix[i])):
            plt.text(
                j,
                i,
                str(conf_matrix[i][j]),
                horizontalalignment="center",
                color="white" if conf_matrix[i, j] > np.max(conf_matrix) / 2 else "black",
            )

            # Add a black border if it's on the diagonal (i == j)
            if i == j:
                ax.add_patch(
                    Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=2)
                )

    plt.ylabel(axis_labels[0])
    plt.xlabel(axis_labels[1])
    plt.tight_layout()

    plt.savefig(output)
    plt.show()


def _sum_confusion_matrices(
    class_labels: list, confusion_matrices: list[dict[str, Any]]
) -> np.ndarray:
    """
    Calculates the element-wise sum of multiple confusion matrices.

    Parameters:
    -----------
    class_labels : list
        A list of all possible class labels. The length determines the matrix size.
    confusion_matrices : list
        A list of dictionary-like objects, where each object contains
        'predicted_values' and 'ground_truth_values'.

    Returns:
    --------
    np.ndarray
        A single NumPy array representing the summed confusion matrix.
    """
    num_labels = len(class_labels)
    confusion_matrix_sum = np.zeros((num_labels, num_labels), dtype=np.int32)

    for matrix_data in confusion_matrices:
        # Generate a confusion matrix for the current fold/run
        cm = confusion_matrix(
            y_true=matrix_data["ground_truth_values"],
            y_pred=matrix_data["predicted_values"],
            labels=np.arange(num_labels),  # Ensures consistent matrix size
        )
        # Add it to the total sum
        confusion_matrix_sum += cm

    return confusion_matrix_sum


def _plot_kfold_val_acc(model_name, training_cycles, val_final_acc):
    plt.figure(figsize=(10, 6))
    plt.scatter(range(1, training_cycles + 1), val_final_acc)
    plt.xlabel("Fold")
    plt.ylabel("Validation Accuracy")
    plt.ylim(0, 1)
    plt.xticks(range(1, training_cycles + 1))
    avg_val_acc = np.mean(val_final_acc)
    plt.axhline(avg_val_acc, color="r", linestyle="--", label=f"Mean Accuracy: {avg_val_acc:.2f}")
    plt.legend()
    filename = model_name + "scatter_plot.png"
    plt.savefig(filename)


def _plot_epoch_train_val_acc(
    model_name: str, train_accuracies: list, val_accuracies: list, k: int
) -> None:
    filename = model_name + "accuracies.png"

    plt.figure(figsize=(10, 6))

    max_epoch = max(
        max(len(train_acc) for train_acc in train_accuracies),
        max(len(val_acc) for val_acc in val_accuracies),
    )

    # Adjust plotting for variable length epochs
    for i in range(k):
        epochs_train = len(train_accuracies[i])
        epochs_val = len(val_accuracies[i])
        plt.plot(range(1, epochs_train + 1), train_accuracies[i], color="red", alpha=0.2)
        plt.plot(range(1, epochs_val + 1), val_accuracies[i], color="green", alpha=0.2)

    # Calculate mean accuracies per epoch, considering different lengths
    mean_train_acc = [
        np.mean([train_accuracies[j][i] for j in range(k) if i < len(train_accuracies[j])])
        for i in range(max_epoch)
    ]
    mean_val_acc = [
        np.mean([val_accuracies[j][i] for j in range(k) if i < len(val_accuracies[j])])
        for i in range(max_epoch)
    ]

    plt.plot(
        range(1, max_epoch + 1), mean_train_acc, label="Training Accuracy", color="red", linewidth=2
    )
    plt.plot(
        range(1, max_epoch + 1),
        mean_val_acc,
        label="Validation Accuracy",
        color="green",
        linewidth=2,
    )

    # Labels and title
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(filename)
    plt.show()


def train_model(
    slurm_array: int,
    index: int,
    selected_params: dict[str, Any],
    train_images: np.ndarray,
    train_labels: np.ndarray,
    test_images: np.ndarray = None,
    test_labels: np.ndarray = None,
) -> None:
    """
    Main wrapper function to run k-fold validation for a given set of
    hyperparameters, save the results, and plot the outputs.
    """
    print(f"Training model {index} with param set: {selected_params}")

    # 1. Call the refactored k-fold validation function
    # It now takes the params dictionary directly and returns a results dictionary
    kfold_results = _kfold_validation(
        images=train_images, labels=train_labels, selected_params=selected_params
    )

    # 2. Unpack results from the returned dictionary
    vaf = kfold_results["validation_accuracies_fold"]
    taf = kfold_results["train_accuracies_fold"]
    vae = kfold_results["validation_accuracies_epoch"]
    tae = kfold_results["train_accuracies_epoch"]
    vnf = kfold_results["validation_nearmiss_fold"]
    best_model = kfold_results["best_model"]
    best_epoch = kfold_results["best_epoch"]
    confusion_matrices_list = kfold_results["confusion_matrices"]

    # Calculate summary statistics.
    avg_vaf = np.mean(vaf)
    best_vaf = np.max(vaf)
    avg_vnf = np.mean(vnf)
    best_vnf = vnf[np.argmax(vaf)]
    best_taf = taf[np.argmax(vaf)]
    best_divergence = best_taf - best_vaf
    print(f"Validation Accuracy: {best_vaf:.4f}")
    print(f"Validation Near-Miss Accuracy: {best_vnf:.4f}")

    # Calculate Test Accuracies
    test_accuracy = None
    test_nearmiss_accuracy = None
    if test_images is not None and test_labels is not None and best_model is not None:
        class_labels = sorted(np.unique(train_labels))
        test_labels_encoded = tf.keras.utils.to_categorical(test_labels, len(class_labels))

        _, test_accuracy = best_model.evaluate(test_images, test_labels_encoded, verbose=0)

        test_preds = np.argmax(best_model.predict(test_images, verbose=0), axis=1)
        near_miss_correct = [
            pred >= max(0, true - 1) and pred <= min(6, true + 1)
            for pred, true in zip(test_preds, test_labels, strict=False)
        ]
        test_nearmiss_accuracy = np.mean(near_miss_correct)

        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Near-Miss Accuracy: {test_nearmiss_accuracy:.4f}")

    # Create output folder
    acc_formatted = f"{(avg_vaf * 100):.2f}"
    folder = f"./array_task{slurm_array}/model{index}_{acc_formatted}/"
    os.makedirs(folder, exist_ok=True)

    # Generate and save plots
    _plot_epoch_train_val_acc(folder, tae, vae, selected_params["k"])
    _plot_kfold_val_acc(folder, selected_params["k"], vaf)

    class_labels = sorted(np.unique(train_labels))
    # 3. Handle the confusion matrix: sum first, then plot
    summed_cm = _sum_confusion_matrices(class_labels, confusion_matrices_list)
    _plot_confusion_matrix(
        conf_matrix=summed_cm,
        labels=np.array(class_labels),
        axis_labels=("Predicted Label", "Ground Truth Label"),
        output=os.path.join(folder, "total_confusion_matrix.png"),
    )

    # Save the best model from the k-fold validation
    if best_model:
        model_path = os.path.join(folder, f"model_{slurm_array}_{index}.keras")
        best_model.save(model_path)

    # 4. Save results to a CSV file
    results_data = selected_params.copy()
    results_data.update(
        {
            "avg_vaf": avg_vaf,
            "best_vaf": best_vaf,
            "avg_vnf": avg_vnf,
            "best_vnf": best_vnf,
            "best_divergence": best_divergence,
            "best_taf": test_accuracy,
            "best_tnf": test_nearmiss_accuracy,
            "best_epoch_trained": best_epoch,
            "model_id": f"{slurm_array}_{index}",
        }
    )

    df = pd.DataFrame([results_data])
    data_path = os.path.join(folder, f"results_{slurm_array}_{index}.csv")
    df.to_csv(data_path, index=False)

    return
