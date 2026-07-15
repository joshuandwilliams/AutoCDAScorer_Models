import logging
import os
from collections import Counter
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

logger = logging.getLogger(__name__)


def _check_valid_dataset(dataset: dict[str, np.ndarray | tf.Tensor]) -> None:
    """
    Checks:
     - Dataset contains valid keys
     - The values are not None
     - The values are np.ndarray or tf.Tensor
     - Warn if a value has length 0
     - Each split has consistent length

    Parameters:
    - dataset : Dict[str, Union[np.ndarray, tf.Tensor]]
            Dataset to be checked

    Returns:
    - None
    """

    # Check if expected keys exist
    expected_keys = {
        "train_images",
        "train_labels",
        "train_filepaths",
        "val_images",
        "val_labels",
        "val_filepaths",
        "ensemble_images",
        "ensemble_labels",
        "ensemble_filepaths",
        "test_images",
        "test_labels",
        "test_filepaths",
    }
    if not expected_keys.issubset(dataset.keys()):
        missing_keys = expected_keys - dataset.keys()
        raise ValueError(f"Missing keys in function input: {missing_keys}")

    for key in expected_keys:
        value = dataset[key]

        # Check if value is None
        if value is None:
            raise ValueError(f"Value for key '{key}' is None.")

        # Check if value is np.ndarray or tf.Tensor
        if not isinstance(value, (np.ndarray, tf.Tensor)):
            raise TypeError(
                f"Value for key '{key}' must be a np.ndarray or tf.Tensor, but got {type(value)}."
            )

    # Check consistent lengths for each split
    splits = ["train", "val", "ensemble", "test"]
    for split in splits:
        images_key = f"{split}_images"
        labels_key = f"{split}_labels"
        filepaths_key = f"{split}_filepaths"

        images_val = dataset[images_key]
        labels_val = dataset[labels_key]
        filepaths_val = dataset[filepaths_key]

        def _get_len(value):
            if isinstance(value, np.ndarray):
                return value.size if value.ndim == 0 else value.shape[0]
            elif isinstance(value, tf.Tensor):
                return (
                    tf.size(value).numpy() if value.shape.rank == 0 else tf.shape(value)[0].numpy()
                )

        images_len = _get_len(images_val)
        labels_len = _get_len(labels_val)
        filepaths_len = _get_len(filepaths_val)

        # Allow all to be zero, otherwise they must match
        if not (images_len == labels_len == filepaths_len):
            raise ValueError(
                f"Inconsistent lengths for '{split}' split: "
                f"images_len={images_len}, labels_len={labels_len}, filepaths_len={filepaths_len}"
            )

        for length_name, length_value in zip(
            ["images_len", "labels_len", "filepaths_len"],
            [images_len, labels_len, filepaths_len],
            strict=False,
        ):
            if length_value == 0:
                logger.warning(f"'{length_name}' for '{split}' split has length zero.")


def load_images_and_labels(
    data_dir: str | Path, image_size: int = 64
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads square tif images, labels, and filenames from a directory.
    Each class of images is assumed to be stored in a separate subdirectory.

    Parameters:
    - data_dir : Union[str, Path]
            Directory containing subdirectories of images. The name of each subdirectory is considered to be a separate class.
    - image_size : int
            The size to which each image should be resized (width, height). Default is 64.

    Returns:
    - Tuple[np.ndarray, np.ndarray, np.ndarray]
            A tuple containing arrays of images, labels, and filepaths.
    """
    data_path = Path(data_dir)
    images, labels, filenames = [], [], []

    for class_dir in [d for d in data_path.iterdir() if d.is_dir()]:
        class_label = class_dir.name
        images_found_in_dir = 0

        for image_path in class_dir.glob("*"):
            if image_path.suffix.lower() in [".tif", ".tiff"]:  # Only tif images
                image = cv.imread(str(image_path))
                if image is not None:
                    resized_image = cv.resize(image, (image_size, image_size))
                    images.append(resized_image)
                    labels.append(class_label)
                    filenames.append(image_path.name)
                    images_found_in_dir += 1
                else:
                    logger.warning(f"Failed to load image {image_path}")

        if images_found_in_dir == 0:
            logger.warning(f"There are no tif images in {class_dir}")

    return np.array(images), np.array(labels).astype(int), np.array(filenames)


def split_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    filepaths: np.ndarray,
    train_ratio: float = 0.6,
    val_ratio: float = 0.15,
    ensemble_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """
    Splits the input images, labels, and filepaths into training, validation, ensemble, and test sets, with equal proportions of each class.

    Parameters:
    - images : np.ndarray
            An array of images to be split
    - labels : np.ndarray
            An array of corresponding labels for the images
    - filepaths : np.ndarray
            An array of filepaths corresponding to the images
    - train_ratio : float, optional
            The proportion of the dataset to include in the training set. Default is 0.6.
    - val_ratio : float, optional
            The proportion of the dataset to include in the validation set. Default is 0.15
    - ensemble_ratio : float_optional
            The proportion of the dataset to include in the ensemble set. Default is 0.15
    - seed : int, optional
            The seed for sampling (reproducibility)

    Returns:
    - Dict[str, np.ndarray]
            A dictionary containing the images, labels, and filepaths for the training, validation, ensemble, and test sets.
    """
    test_ratio = 1 - train_ratio - val_ratio - ensemble_ratio

    if test_ratio <= 0:
        raise ValueError(
            "Invalid ratios. The sum of train_ratio, val_ratio, and ensemble_ratio must be less than 1"
        )

    train_images, images_rem, train_labels, labels_rem, train_filepaths, filepaths_rem = (
        train_test_split(
            images,
            labels,
            filepaths,
            train_size=train_ratio,
            random_state=seed,
            shuffle=True,
            stratify=labels,
        )
    )

    val_vs_rest_ratio = val_ratio / (val_ratio + ensemble_ratio + test_ratio)
    val_images, images_rem2, val_labels, labels_rem2, val_filepaths, filepaths_rem2 = (
        train_test_split(
            images_rem,
            labels_rem,
            filepaths_rem,
            train_size=val_vs_rest_ratio,
            random_state=seed,
            shuffle=True,
            stratify=labels_rem,
        )
    )

    ensemble_vs_test_ratio = ensemble_ratio / (ensemble_ratio + test_ratio)
    (
        ensemble_images,
        test_images,
        ensemble_labels,
        test_labels,
        ensemble_filepaths,
        test_filepaths,
    ) = train_test_split(
        images_rem2,
        labels_rem2,
        filepaths_rem2,
        train_size=ensemble_vs_test_ratio,
        random_state=seed,
        shuffle=True,
        stratify=labels_rem2,
    )

    dataset = {
        "train_images": np.array(train_images),
        "train_labels": np.array(train_labels),
        "train_filepaths": np.array(train_filepaths),
        "val_images": np.array(val_images),
        "val_labels": np.array(val_labels),
        "val_filepaths": np.array(val_filepaths),
        "ensemble_images": np.array(ensemble_images),
        "ensemble_labels": np.array(ensemble_labels),
        "ensemble_filepaths": np.array(ensemble_filepaths),
        "test_images": np.array(test_images),
        "test_labels": np.array(test_labels),
        "test_filepaths": np.array(test_filepaths),
    }

    _check_valid_dataset(dataset)

    return dataset


def _downsample_oversized_classes(
    images: np.ndarray,
    labels: np.ndarray,
    filepaths: np.ndarray,
    per_class_train_size: int | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Downsamples classes in the dataset that exceed the specified number of samples per class.

    Parameters:
    - images : np.ndarray
            Array of training images.
    - labels : np.ndarray
            Array of training labels.
    - filepaths : np.ndarray
            Array of training filepaths.
    - per_class_train_size : Optional[int]
            The maximum number of samples per class. If None, the function downsamples to the size of the smallest class. Default is None.
    - seed : int, optional
            The seed for sampling (reproducibility)

    Returns:
    - undersampled_images : np.ndarray
            Array of undersampled images.
    - undersampled_labels : np.ndarray
            Array of undersampled labels.
    - undersampled_filepaths : np.ndarray
            Array of undersampled filepaths.
    """
    smallest_class_size = min(Counter(labels).values())

    if per_class_train_size is None:
        per_class_train_size = smallest_class_size
    else:
        if not isinstance(per_class_train_size, int):
            raise TypeError("per_class_train_size must be an integer or None")
        if per_class_train_size <= 0 or per_class_train_size > smallest_class_size:
            raise ValueError(
                f"per_class_train_size must be greater than zero and less than or equal to the smallest class size ({smallest_class_size})"
            )

    undersampled_images = []
    undersampled_labels = []
    undersampled_filepaths = []

    for label in np.unique(labels):
        samples_indices = np.where(labels == label)[0]

        if len(samples_indices) > per_class_train_size:
            samples_indices = resample(
                samples_indices, replace=False, n_samples=per_class_train_size, random_state=seed
            )

        undersampled_images.extend(
            images[samples_indices].tolist()
        )  # Need to be lists for extend()
        undersampled_labels.extend(labels[samples_indices].tolist())
        undersampled_filepaths.extend(np.array(filepaths)[samples_indices].tolist())

    undersampled_images = np.array(undersampled_images)
    undersampled_labels = np.array(undersampled_labels)
    undersampled_filepaths = np.array(undersampled_filepaths)

    np.random.seed(seed)
    shuffle_indices = np.random.permutation(len(undersampled_labels))
    undersampled_images = undersampled_images[shuffle_indices]
    undersampled_labels = undersampled_labels[shuffle_indices]
    undersampled_filepaths = undersampled_filepaths[shuffle_indices]

    return undersampled_images, undersampled_labels, undersampled_filepaths


def _preprocess_images(images: np.ndarray) -> tf.Tensor:
    """
    Normalizes image pixel values to the range [0, 1] and converts to tf.Tensor.

    Parameters:
    - images : np.ndarray
            A numpy array of images in RGB format.

    Returns:
    - tf_images : tf.Tensor
            A tf.Tensor of normalized images
    """
    if len(images) == 0:
        logger.warning("'images' is empty")
        return images

    images = images / 256.0
    tf_images = tf.convert_to_tensor(images)

    return tf_images


def _preprocess_labels(labels: np.ndarray) -> tf.Tensor:
    """
    One-hot encodes labels for classification tasks.

    Parameters:
    - labels : np.ndarray
            A numpy array of labels (encoded as integers)

    Returns:
    - tf_labels : tf.Tensor
            A tf.Tensor of one-hot encoded labels
    """
    if len(labels) == 0:
        logger.warning("'labels' is empty")
        return labels

    num_classes = len(np.unique(labels))
    tf_labels = tf.keras.utils.to_categorical(labels, num_classes)

    return tf.convert_to_tensor(tf_labels, dtype=tf.float32)


def preprocess_split_dataset(
    dataset: dict[str, np.ndarray], seed: int, onehot: bool = True
) -> dict[str, np.ndarray | tf.Tensor]:
    """
    Runs preprocessing of images and labels on the split dataset.

    Parameters:
    - dataset : Dict[str, np.ndarray]
            Dataset split into training, validation, ensemble, and test portions.
    - seed : int
            Seed for random sampling.
    - onehot : bool
            Determines if labels are one-hot encoded or left as integers.

    Returns:
    - Dict[str, Union[np.ndarray, tf.Tensor]]
            Dataset processed with preprocess_images() and preprocess_labels().
    """

    _check_valid_dataset(dataset)

    if not isinstance(onehot, bool):
        raise TypeError("onehot parameter must be True or False (bool)")

    # Train
    train_images, train_labels, train_filepaths = _downsample_oversized_classes(
        dataset["train_images"], dataset["train_labels"], dataset["train_filepaths"], seed=seed
    )
    processed_train_images = _preprocess_images(train_images)

    # Val
    val_images, val_labels, val_filepaths = _downsample_oversized_classes(
        dataset["val_images"], dataset["val_labels"], dataset["val_filepaths"], seed=seed
    )
    processed_val_images = _preprocess_images(val_images)

    # Ensemble
    ensemble_images, ensemble_labels, ensemble_filepaths = _downsample_oversized_classes(
        dataset["ensemble_images"],
        dataset["ensemble_labels"],
        dataset["ensemble_filepaths"],
        seed=seed,
    )
    processed_ensemble_images = _preprocess_images(ensemble_images)

    # Test
    test_images, test_labels, test_filepaths = _downsample_oversized_classes(
        dataset["test_images"], dataset["test_labels"], dataset["test_filepaths"], seed=seed
    )
    processed_test_images = _preprocess_images(test_images)

    if onehot:
        train_labels = _preprocess_labels(train_labels)
        val_labels = _preprocess_labels(val_labels)
        ensemble_labels = _preprocess_labels(ensemble_labels)
        test_labels = _preprocess_labels(test_labels)

    return {
        "train_images": processed_train_images,
        "train_labels": train_labels,
        "train_filepaths": train_filepaths,
        "val_images": processed_val_images,
        "val_labels": val_labels,
        "val_filepaths": val_filepaths,
        "ensemble_images": processed_ensemble_images,
        "ensemble_labels": ensemble_labels,
        "ensemble_filepaths": ensemble_filepaths,
        "test_images": processed_test_images,
        "test_labels": test_labels,
        "test_filepaths": test_filepaths,
    }


def save_dataset(dataset: dict[str, np.ndarray | tf.Tensor], base_out_path: str) -> None:
    """
    Saves dataset to .npy and creates a metadata .csv file.
    The base_out_path should not include a file extension.
    """
    _check_valid_dataset(dataset)

    # Save the binary data
    npy_path = f"{base_out_path}.npy"
    np.save(npy_path, dataset, allow_pickle=True)

    # Generate and save metadata CSV
    splits = ["train", "val", "ensemble", "test"]
    rows = []

    for split in splits:
        # Accessing the data from the dataset dictionary [cite: 12]
        filepaths = dataset[f"{split}_filepaths"]
        labels = dataset[f"{split}_labels"]

        # Ensure we are handling numpy/list for iteration
        if isinstance(filepaths, tf.Tensor):
            filepaths = filepaths.numpy()
        if isinstance(labels, tf.Tensor):
            labels = labels.numpy()

        for path, label in zip(filepaths, labels, strict=False):
            rows.append(
                {
                    "split": split,
                    "filename": Path(path).name,
                    "full_path": str(path),
                    "label": label,
                }
            )

    df = pd.DataFrame(rows)
    csv_path = f"{base_out_path}.csv"
    df.to_csv(csv_path, index=False)


def load_dataset(path: str) -> dict[str, np.ndarray | tf.Tensor]:
    """
    Loads processed dataset from a numpy file.

    Parameters:
    - path : str
            The filepath of the npy dataset.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset file not found at: {path}.")

    data = np.load(path, allow_pickle=True)
    dataset = data.item()
    _check_valid_dataset(dataset)

    # Convert specific keys back to Tensors
    for split in ["train", "val", "ensemble", "test"]:
        img_key = f"{split}_images"
        lbl_key = f"{split}_labels"

        dataset[img_key] = tf.convert_to_tensor(dataset[img_key])
        dataset[lbl_key] = tf.convert_to_tensor(dataset[lbl_key])

    return dataset


def combine_train_val(
    dataset: dict[str, np.ndarray | tf.Tensor],
) -> dict[str, np.ndarray | tf.Tensor]:
    """
    Combines training and validation parts of dataset (to be used with KFold Cross-Validation)

    Parameters:
    - dataset : Dictionary object containing the images, labels, and filepaths of each data part.

    Returns:
    - Dictionary containing combined images, labels, and filepaths.
    """
    _check_valid_dataset(dataset)

    images = np.concatenate((dataset["train_images"], dataset["val_images"]), axis=0)
    labels = np.concatenate((dataset["train_labels"], dataset["val_labels"]), axis=0)

    filepaths = np.concatenate((dataset["train_filepaths"], dataset["val_filepaths"]), axis=0)

    return {"images": images, "labels": labels, "filepaths": filepaths}
