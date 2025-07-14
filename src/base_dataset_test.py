from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from collections import Counter

from base_dataset import load_images_and_labels, split_dataset, downsample_oversized_classes, preprocess_images, preprocess_labels, preprocess_split_dataset, save_dataset, load_dataset

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

	@pytest.fixture
	def example_data(self):
		num_samples = 100
		num_classes = 5

		images = np.random.rand(num_samples, 64, 64, 3)
		labels = np.array([i % num_classes for i in range(num_samples)])
		filepaths = np.array([f"image_{i}.jpg" for i in range(num_samples)])

		return images, labels, filepaths

	def test_valid_data_splitting(self, example_data):

		images, labels, filepaths = example_data
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
		
	def test_empty_images(self):
		empty_images = np.array([])
		empty_labels = np.array([])
		empty_filepaths = np.array([])

		with pytest.raises(ValueError, match="Input arrays must not be empty"):
			split_dataset(empty_images, empty_labels, empty_filepaths)

	def test_unequal_input_lengths(self, example_data):
		images, labels, filepaths = example_data
		labels_with_unequal_length = labels[:-1]

		with pytest.raises(ValueError, match="Input arrays must have the same length"):
			split_dataset(images, labels_with_unequal_length, filepaths)		

	def test_invalid_ratios(self, example_data):
		images, labels, filepaths = example_data
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

	@pytest.fixture
	def example_data(self):
		images = np.random.rand(8, 64, 64, 3)
		labels = np.array([0, 0, 0, 1, 1, 1, 1, 1])
		filepaths = np.array([f"image_{i}.jpg" for i in range(8)])

		return images, labels, filepaths, 3
	
	def test_valid_input(self, example_data):

		images, labels, filepaths, smallest_size = example_data

		# per_class_train_size = None
		expected_total_size = smallest_size * len(np.unique(labels))
		u_images, u_labels, u_filepaths = downsample_oversized_classes(images, labels, filepaths)

		assert len(u_images) == expected_total_size
		assert len(u_labels) == expected_total_size
		assert len(u_filepaths) == expected_total_size

		# per_class_train_size = integer
		integer_size = 2
		expected_total_size = integer_size * len(np.unique(labels))
		u_images, u_labels, u_filepaths = downsample_oversized_classes(images, labels, filepaths, integer_size)

		assert len(u_images) == expected_total_size
		assert len(u_labels) == expected_total_size
		assert len(u_filepaths) == expected_total_size

	def test_output_shuffled_order(self, example_data):
		images, labels, filepaths, _ = example_data

		u_images, u_labels, u_filepaths = downsample_oversized_classes(images, labels, filepaths)

		assert not np.array_equal(u_labels, np.sort(u_labels))

	@pytest.mark.parametrize(
		"invalid_size, expected_exception, error_match",
		[
			("test", TypeError, "per_class_train_size must be an integer or None"),
			(1000, ValueError, r"per_class_train_size must be greater than zero and less than or equal to the smallest class size \(3\)"),
			(-1, ValueError, r"per_class_train_size must be greater than zero and less than or equal to the smallest class size \(3\)"),
			(0, ValueError, r"per_class_train_size must be greater than zero and less than or equal to the smallest class size \(3\)"),
			(3.5, TypeError, "per_class_train_size must be an integer or None")
		]
	)
	def test_invalid_class_size(self, example_data, invalid_size, expected_exception, error_match):
		images, labels, filepaths, _ = example_data

		with pytest.raises(expected_exception, match=error_match):
			downsample_oversized_classes(images, labels, filepaths, per_class_train_size=invalid_size)


	def test_data_empty(self):
		empty_arr = np.array([])
		with pytest.raises(ValueError, match="Input arrays must not be empty"):
			downsample_oversized_classes(empty_arr, empty_arr, empty_arr)

	def test_uneven_lengths(self, example_data):
		images, labels, filepaths, _ = example_data
		mismatched_labels = labels[:-1] 

		with pytest.raises(ValueError, match="Input arrays must have the same length"):
			downsample_oversized_classes(images, mismatched_labels, filepaths)

class TestPreprocessImages:

	def test_valid_input(self):
		images = np.random.rand(5, 64, 64, 3)
		processed_images = preprocess_images(images)

		assert isinstance(processed_images, tf.Tensor)
		assert processed_images.shape[0] == 5

	def test_empty_input(self):
		images = np.array([])
		
		with pytest.raises(ValueError, match="Image array must not be empty"):
			preprocess_images(images)

class TestPreprocessLabels:

	def test_valid_input(self):
		labels = np.array([0, 0, 0, 1, 1, 1, 1, 1])
		processed_labels = preprocess_labels(labels)

		assert isinstance(processed_labels, tf.Tensor)
		assert processed_labels.shape[0] == 8

	def test_empty_input(self):
		labels = np.array([])

		with pytest.raises(ValueError, match="Labels array must not be empty"):
			preprocess_labels(labels)

class TestPreprocessSplitDataset:

	@pytest.fixture
	def mock_dataset(self):
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
	
	def test_valid_input(self, mock_dataset):
		data = mock_dataset

		num_classes = np.max(mock_dataset['val_labels']) + 1

		train_labels = mock_dataset['train_labels']
		final_train_size = num_classes * min(Counter(train_labels).values())

		val_labels = mock_dataset['val_labels']
		final_val_size = num_classes * min(Counter(val_labels).values())

		processed_data = preprocess_split_dataset(dataset=data, seed=42)

		expected_keys = {
            'train_images', 'train_labels', 'train_filepaths',
            'val_images', 'val_labels', 'val_filepaths',
            'ensemble_images', 'ensemble_labels', 'ensemble_filepaths',
            'test_images', 'test_labels', 'test_filepaths'
        }

		assert expected_keys.issubset(processed_data.keys()), "Missing keys in processed_data output"

		assert isinstance(processed_data['train_images'], tf.Tensor)
		assert processed_data['train_images'].shape == (final_train_size, 64, 64, 3)

		assert isinstance(processed_data['val_images'], tf.Tensor)
		assert processed_data['val_labels'].shape == (final_val_size, num_classes)
		assert processed_data['val_labels'].dtype == tf.float32

	def test_missing_items(self, mock_dataset):
		data = mock_dataset
		del data['train_images']

		with pytest.raises(ValueError, match="Missing keys in function input: {'train_images'}"):
			preprocess_split_dataset(data, seed=42)