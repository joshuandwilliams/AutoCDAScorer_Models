import os
from pathlib import Path

import logging
import numpy as np
import pytest
import re
import tensorflow as tf
from collections import Counter

from base_dataset import load_images_and_labels, _check_valid_dataset, split_dataset, _downsample_oversized_classes, _preprocess_images, _preprocess_labels, preprocess_split_dataset, save_dataset, load_dataset

@pytest.fixture
def mock_dataset():
	return {
		'train_images': np.random.rand(100, 64, 64, 3),
		'train_labels': np.random.randint(0, 5, 100),
		'train_filepaths': np.array([f"image_train_{i}.jpg" for i in range(100)]),

		'val_images': np.random.rand(100, 64, 64, 3),
		'val_labels': np.random.randint(0, 5, 100),
		'val_filepaths': np.array([f"image_val_{i}.jpg" for i in range(100)]),

		'ensemble_images': np.random.rand(100, 64, 64, 3),
		'ensemble_labels': np.random.randint(0, 5, 100),
		'ensemble_filepaths': np.array([f"image_ensemble_{i}.jpg" for i in range(100)]),

		'test_images': np.random.rand(100, 64, 64, 3),
		'test_labels': np.random.randint(0, 5, 100),
		'test_filepaths': np.array([f"image_test_{i}.jpg" for i in range(100)])
	}

@pytest.fixture
def mock_images_labels_filepaths():
	num_samples = 100
	num_classes = 5

	images = np.random.rand(num_samples, 64, 64, 3)
	labels = np.array([i % num_classes for i in range(num_samples)])
	filepaths = np.array([f"image_{i}.jpg" for i in range(num_samples)])

	return images, labels, filepaths

class TestCheckValidDataset:

	def test_valid_dataset(self, mock_dataset):
		_check_valid_dataset(mock_dataset)

	def test_missing_key(self, mock_dataset):
		del mock_dataset['train_labels']

		expected_error = re.escape("Missing keys in function input: {'train_labels'}")
		with pytest.raises(ValueError, match=expected_error):
			_check_valid_dataset(mock_dataset)

	def test_key_empty(self, mock_dataset):
		mock_dataset['train_labels'] = None

		expected_error = re.escape("Value for key 'train_labels' is None.")
		with pytest.raises(ValueError, match=expected_error):
			_check_valid_dataset(mock_dataset)

	def test_key_wrong_type(self, mock_dataset):
		mock_dataset['train_labels'] = "Hello"

		expected_error = re.escape("Value for key 'train_labels' must be a np.ndarray or tf.Tensor, but got <class 'str'>.")
		with pytest.raises(TypeError, match=expected_error):
			_check_valid_dataset(mock_dataset)

	def test_length_mismatch(self, mock_dataset):
		mock_dataset['train_labels'] = mock_dataset['train_labels'][:-1]

		expected_error = re.escape(
			"Inconsistent lengths for 'train' split: "
			"images_len=100, labels_len=99, filepaths_len=100"
		)
		with pytest.raises(ValueError, match=expected_error):
			_check_valid_dataset(mock_dataset)

	def test_split_empty(self, mock_dataset, caplog):
		mock_dataset['train_images'] = np.array([])
		mock_dataset['train_labels'] = np.array([])
		mock_dataset['train_filepaths'] = np.array([])

		caplog.set_level(logging.WARNING)
		_check_valid_dataset(mock_dataset)

		assert len(caplog.records) == 3

		expected_messages = [
			"'images_len' for 'train' split has length zero.",
			"'labels_len' for 'train' split has length zero.",
			"'filepaths_len' for 'train' split has length zero."
		]

		actual_messages = [record.message for record in caplog.records]

		for msg in expected_messages:
			assert msg in actual_messages
		
		for record in caplog.records:
			assert record.levelname == "WARNING"


class TestLoadImagesAndLabels:

	@pytest.fixture
	def setup_fake_fs(self, fs):
		fs.create_file("/data/0/image0_1.tif")
		fs.create_file("/data/1/image1_1.tif")
		fs.create_file("/data/1/image1_2.tiff")
		fs.create_dir("/data/empty_dir")
		fs.create_file("/data/0/some_file.txt")
		return Path("/data")
	
	def test_valid_image_loading(self, setup_fake_fs, mocker):
		data_path = setup_fake_fs

		mock_imread = mocker.patch('cv2.imread')
		mock_resize = mocker.patch('cv2.resize')
		mock_image_array = np.zeros((64, 64, 3), dtype=np.uint8)
		mock_imread.return_value = mock_image_array
		mock_resize.return_value = mock_image_array

		images, labels, filepaths = load_images_and_labels(data_path, image_size=64)

		assert len(images == 3)
		sorted_results = sorted(zip(labels, filepaths))
		sorted_labels = [label for label, path in sorted_results]
		assert sorted_labels == [0, 1, 1]
		sorted_filepaths = [path for label, path in sorted_results]
		assert sorted_filepaths == ['image0_1.tif', 'image1_1.tif', 'image1_2.tiff']

		assert mock_imread.call_count == 3
		assert mock_resize.call_count == 3

	def test_empty_directory(self, setup_fake_fs):
		empty_dir_path = setup_fake_fs / "empty_dir"

		images, labels, filepaths = load_images_and_labels(empty_dir_path)

		assert len(images) == 0
		assert len(labels) == 0
		assert len(filepaths) == 0

	def test_unreadable_image(self, setup_fake_fs, mocker):
		data_path = setup_fake_fs
		
		mocker.patch('cv2.imread', return_value=None)

		mock_log_warning = mocker.patch('base_dataset.logger.warning')

		images, labels, filepaths = load_images_and_labels(data_path)

		assert len(images) == 0
		assert len(labels) == 0
		assert len(filepaths) == 0

		assert mock_log_warning.call_count == 6 # 3 "failed to load", 3 "no tif images"
		mock_log_warning.assert_any_call("Failed to load image /data/0/image0_1.tif")

class TestSplitDataset:

	def test_valid_data_splitting(self, mock_images_labels_filepaths):

		images, labels, filepaths = mock_images_labels_filepaths
		original_count = len(images)

		result = split_dataset(images, labels, filepaths)

		assert len(result) == 12

		train_count = len(result['train_images'])
		val_count = len(result['val_images'])
		ensemble_count = len(result['ensemble_images'])
		test_count = len(result['test_images'])

		assert (train_count + val_count + ensemble_count + test_count) == original_count

		assert len(result['train_labels']) == train_count
		assert len(result['val_filepaths']) == val_count

	def test_invalid_ratios(self, mock_images_labels_filepaths):
		images, labels, filepaths = mock_images_labels_filepaths
		with pytest.raises(ValueError, match="Invalid ratios"):
			split_dataset(
				images,
				labels,
				filepaths,
				train_ratio=0.7,
				val_ratio=0.2,
				ensemble_ratio=0.15
				)

class TestDownsampleOversizedClasses:
	
	def test_valid_input(self, mock_images_labels_filepaths):

		images, labels, filepaths = mock_images_labels_filepaths
		smallest_size = min(Counter(labels).values())

		# per_class_train_size = None
		expected_total_size = smallest_size * len(np.unique(labels))
		u_images, u_labels, u_filepaths = _downsample_oversized_classes(images, labels, filepaths)

		assert len(u_images) == expected_total_size
		assert len(u_labels) == expected_total_size
		assert len(u_filepaths) == expected_total_size

		# per_class_train_size = integer
		integer_size = 2
		expected_total_size = integer_size * len(np.unique(labels))
		u_images, u_labels, u_filepaths = _downsample_oversized_classes(images, labels, filepaths, integer_size)

		assert len(u_images) == expected_total_size
		assert len(u_labels) == expected_total_size
		assert len(u_filepaths) == expected_total_size

	def test_output_shuffled_order(self, mock_images_labels_filepaths):
		images, labels, filepaths = mock_images_labels_filepaths
		u_images, u_labels, u_filepaths = _downsample_oversized_classes(images, labels, filepaths)

		assert not np.array_equal(u_labels, np.sort(u_labels))

	@pytest.mark.parametrize(
		"invalid_size, expected_exception, error_match",
		[
			("test", TypeError, "per_class_train_size must be an integer or None"),
			(1000, ValueError, r"per_class_train_size must be greater than zero and less than or equal to the smallest class size \(20\)"),
			(-1, ValueError, r"per_class_train_size must be greater than zero and less than or equal to the smallest class size \(20\)"),
			(0, ValueError, r"per_class_train_size must be greater than zero and less than or equal to the smallest class size \(20\)"),
			(3.5, TypeError, "per_class_train_size must be an integer or None")
		]
	)
	def test_invalid_class_size(self, mock_images_labels_filepaths, invalid_size, expected_exception, error_match):
		images, labels, filepaths = mock_images_labels_filepaths

		with pytest.raises(expected_exception, match=error_match):
			_downsample_oversized_classes(images, labels, filepaths, per_class_train_size=invalid_size)

class TestPreprocessImages:

	def test_valid_input(self):
		images = np.random.rand(5, 64, 64, 3)
		processed_images = _preprocess_images(images)

		assert isinstance(processed_images, tf.Tensor)
		assert processed_images.shape[0] == 5

	def test_empty_input(self, caplog):
		images = np.array([])
		
		caplog.set_level(logging.WARNING)
		
		_preprocess_images(images)

		assert len(caplog.records) == 1
		record = caplog.records[0]
		assert record.levelname == "WARNING"
		assert record.message == "'images' is empty"

class TestPreprocessLabels:

	def test_valid_input(self):
		labels = np.array([0, 0, 0, 1, 1, 1, 1, 1])
		processed_labels = _preprocess_labels(labels)

		assert isinstance(processed_labels, tf.Tensor)
		assert processed_labels.shape[0] == 8

	def test_empty_input(self, caplog):
		labels = np.array([])

		caplog.set_level(logging.WARNING)
		
		_preprocess_labels(labels)

		assert len(caplog.records) == 1
		record = caplog.records[0]
		assert record.levelname == "WARNING"
		assert record.message == "'labels' is empty"

class TestPreprocessSplitDataset:
	
	def test_valid_input(self, mock_dataset):
		data = mock_dataset

		num_classes = np.max(mock_dataset['val_labels']) + 1

		train_labels = mock_dataset['train_labels']
		final_train_size = num_classes * min(Counter(train_labels).values())

		val_labels = mock_dataset['val_labels']
		final_val_size = num_classes * min(Counter(val_labels).values())

		processed_data = preprocess_split_dataset(dataset=data, seed=42)

		_check_valid_dataset(processed_data)

		assert isinstance(processed_data['train_images'], tf.Tensor)
		assert processed_data['train_images'].shape == (final_train_size, 64, 64, 3)

		assert isinstance(processed_data['val_images'], tf.Tensor)
		assert processed_data['val_labels'].shape == (final_val_size, num_classes)
		assert processed_data['val_labels'].dtype == tf.float32

class TestSaveDataset:
	
	def test_valid_input(self, mock_dataset, fs):
		out_path = "/test_dataset.npy"
		save_dataset(mock_dataset, out_path) # fs should automatically catch the saved file
		assert os.path.exists(out_path)

class TestLoadDataset:

	def test_valid_input(self, mock_dataset, fs):
		test_path = "/valid_dataset.npy"
		save_dataset(mock_dataset, test_path)
		
		loaded_dataset = load_dataset(test_path)

		_check_valid_dataset(loaded_dataset)

	def test_invalid_path(self):
		path = "/non_existent.npy"

		expected_error = re.escape(f"Dataset file not found at: {path}.")
		with pytest.raises(FileNotFoundError, match=expected_error):
			load_dataset(path)
