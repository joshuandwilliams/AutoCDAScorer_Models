from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from sklearn.metrics import confusion_matrix
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
                    Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="black",
                        linewidth=plt.rcParams["axes.linewidth"],
                    )
                )

    plt.ylabel(axis_labels[0])
    plt.xlabel(axis_labels[1])
    plt.tight_layout()

    plt.savefig(output)
    plt.show()
    plt.close()


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
    plt.close()


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
    plt.close()
