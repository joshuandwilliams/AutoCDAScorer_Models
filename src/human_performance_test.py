import pytest
import numpy as np
import pandas as pd
from typing import Dict
from sklearn.metrics import confusion_matrix
from unittest import mock
from collections import Counter

from human_performance import generate_scorer_key, anonymise_scorers, downsample_score_classes_csv, generate_scorer_cms, cm_accuracies, plot_confusion_matrix, find_pairwise_agreement, extract_score_pairs

class TestGenerateScorerKey:
	
	@pytest.fixture
	def sample_data(self) -> pd.DataFrame:
		"""
		Fixture providing sample data with the 'Scorer1' column.
		
		Returns:
		--------
		pd.DataFrame
			A DataFrame containing a column 'Scorer1' with sample scorer names.
		"""
		data = {
			'Scorer1': ['Alice', 'Bob', 'Charlie', 'Alice', 'Bob', 'David']
		}
		return pd.DataFrame(data)

	def test_generate_scorer_key_valid_data(self, sample_data: pd.DataFrame) -> None:
		"""
		Test the function with valid data and check if it returns the expected keys and scorers.
		
		Parameters:
		-----------
		sample_data : pd.DataFrame
			A DataFrame containing sample data with 'Scorer1' column.

		Returns:
		--------
		None
		"""
		key = generate_scorer_key(sample_data, random_state=42)
		
		assert isinstance(key, dict)
		assert len(key) == 4
		assert 1 in key
		assert 2 in key

	def test_generate_scorer_key_random_state(self, sample_data: pd.DataFrame) -> None:
		"""
		Test if the random_state produces consistent anonymization keys.
		
		Parameters:
		-----------
		sample_data : pd.DataFrame
			A DataFrame containing sample data with 'Scorer1' column.

		Returns:
		--------
		None
		"""
		key_1 = generate_scorer_key(sample_data, random_state=42)
		key_2 = generate_scorer_key(sample_data, random_state=42)
		assert key_1 == key_2

		key_3 = generate_scorer_key(sample_data, random_state=99)
		assert key_1 != key_3

	def test_generate_scorer_key_missing_column(self) -> None:
		"""
		Test if the function raises a ValueError when 'Scorer1' column is missing.
		
		Parameters:
		-----------
		None

		Returns:
		--------
		None
		"""
		data = pd.DataFrame({'OtherColumn': [1, 2, 3]})
		
		with pytest.raises(ValueError, match="The 'Scorer1' column is missing or empty"):
			generate_scorer_key(data)

	def test_generate_scorer_key_empty_column(self) -> None:
		"""
		Test if the function raises a ValueError when 'Scorer1' column is empty or has only NaNs.
		
		Parameters:
		-----------
		None

		Returns:
		--------
		None
		"""
		data = pd.DataFrame({'Scorer1': [None, None, None]})
		
		with pytest.raises(ValueError, match="The 'Scorer1' column is missing or empty"):
			generate_scorer_key(data)

class TestAnonymiseScorers:
	@pytest.fixture
	def setup_data(self) -> pd.DataFrame:
		"""
		Fixture to set up sample data for testing the anonymise_scorers function.

		Returns:
		--------
		pd.DataFrame
			A DataFrame containing sample scorer names for testing.
		"""
		return pd.DataFrame({
			'Scorer1': ['Alice', 'Bob', 'Charlie', 'Alice'],
			'Scorer2': ['Bob', 'Charlie', 'Alice', 'Alice'],
			'Scorer3': ['Charlie', 'Alice', 'Bob', 'Bob'],
		})

	@pytest.fixture
	def setup_key(self) -> Dict[int, str]:
		"""
		Fixture to set up the key for anonymization.

		Returns:
		--------
		Dict[int, str]
			A dictionary mapping scorer names to integer IDs.
		"""
		return {
			1: 'Alice',
			2: 'Bob',
			3: 'Charlie'
		}

	def test_anonymise_scorers(self, setup_data: pd.DataFrame, setup_key: Dict[int, str]) -> None:
		"""
		Test anonymise_scorers function with correct data to ensure names are anonymized correctly.

		Parameters:
		-----------
		setup_data : pd.DataFrame
			A DataFrame containing sample scorer names for testing.

		setup_key : Dict[int, str]
			A dictionary mapping integer IDs to scorer names.

		Returns:
		--------
		None
		"""
		expected_output = pd.DataFrame({
			'Scorer1': pd.Series([1, 2, 3, 1], dtype='Int64'),
			'Scorer2': pd.Series([2, 3, 1, 1], dtype='Int64'),
			'Scorer3': pd.Series([3, 1, 2, 2], dtype='Int64'),
		})

		result = anonymise_scorers(setup_data.copy(), setup_key)
		
		pd.testing.assert_frame_equal(result, expected_output)

	def test_anonymise_scorers_missing_key(self, setup_data: pd.DataFrame) -> None:
		"""
		Test anonymise_scorers function when some names in the DataFrame are missing in the key.

		Parameters:
		-----------
		setup_data : pd.DataFrame
			A DataFrame containing sample scorer names for testing.

		Returns:
		--------
		None
		"""
		key = {
			1: 'Alice',
			2: 'Bob'
			# 'Charlie' is missing from the key
		}
		
		expected_output = pd.DataFrame({
			'Scorer1': pd.Series([1, 2, pd.NA, 1], dtype='Int64'),
			'Scorer2': pd.Series([2, pd.NA, 1, 1], dtype='Int64'),
			'Scorer3': pd.Series([pd.NA, 1, 2, 2], dtype='Int64'),
		})

		result = anonymise_scorers(setup_data.copy(), key)
		
		pd.testing.assert_frame_equal(result, expected_output)

	def test_anonymise_scorers_empty_df(self) -> None:
		"""
		Test anonymise_scorers function with an empty DataFrame.

		Returns:
		--------
		None
		"""
		empty_df = pd.DataFrame(columns=['Scorer1', 'Scorer2', 'Scorer3'])
		key = {
			1: 'Alice',
			2: 'Bob',
			3: 'Charlie'
		}
		
		expected_output = pd.DataFrame(columns=['Scorer1', 'Scorer2', 'Scorer3'], dtype='Int64')

		result = anonymise_scorers(empty_df.copy(), key)
		
		pd.testing.assert_frame_equal(result, expected_output)

	def test_anonymise_scorers_empty_key(self, setup_data: pd.DataFrame) -> None:
		"""
		Test anonymise_scorers function with an empty key.

		Parameters:
		-----------
		setup_data : pd.DataFrame
			A DataFrame containing sample scorer names for testing.

		Returns:
		--------
		None
		"""
		key = {}
		
		expected_output = pd.DataFrame({
			'Scorer1': pd.Series([pd.NA] * len(setup_data), dtype='Int64'),
			'Scorer2': pd.Series([pd.NA] * len(setup_data), dtype='Int64'),
			'Scorer3': pd.Series([pd.NA] * len(setup_data), dtype='Int64'),
		})

		result = anonymise_scorers(setup_data.copy(), key)
		
		pd.testing.assert_frame_equal(result, expected_output)

class TestDownsampleScoreClassesCSV:
	
	@pytest.fixture
	def setup_data(self) -> pd.DataFrame:
		"""
		Fixture to set up sample data for testing the downsample_score_classes_csv function.
		
		Returns:
		--------
		pd.DataFrame
			A DataFrame containing sample Score, Scorer, and Median_Score values for testing.
		"""
		return pd.DataFrame({
			'Score': [1, 2, 2, 3, 3, 3, 1, 2, 3],
			'Scorer': ['A', 'B', 'C', 'A', 'B', 'C', 'D', 'E', 'F'],
			'Median_Score': [3, 1, 2, 3, 1, 2, 3, 2, 3]
		})
	
	def test_downsample_score_classes_csv(self, setup_data: pd.DataFrame) -> None:
		"""
		Test that the standardize_class_sizes function correctly standardizes the class sizes.
		
		Parameters:
		-----------
		setup_data : pd.DataFrame
			A DataFrame containing sample rows for testing.

		Returns:
		--------
		None
		"""
		result = downsample_score_classes_csv(setup_data.copy(), seed=42, print_results=True)
		
		class_sizes = Counter(result['Median_Score'])
		
		min_class_size = setup_data['Median_Score'].value_counts().min()
		
		assert all(size == min_class_size for size in class_sizes.values())

class TestGenerateScorerCMS:
	@pytest.fixture
	def setup_data(self) -> pd.DataFrame:
		"""
		Fixture to set up sample data for testing the generate_scorer_cms function.

		Returns:
		--------
		pd.DataFrame
			A DataFrame containing sample Median_Score, Scorer, and Score values for testing.
		"""
		return pd.DataFrame({
			'Median_Score': [3, 2, 3, 1, 2, 3, 1, 2, 1],
			'Scorer': ['A', 'A', 'B', 'B', 'B', 'C', 'C', 'C', 'C'],
			'Score': [3, 2, 3, 1, 1, 3, 1, 2, pd.NA]
		}).astype({'Score': 'Int64', 'Median_Score': 'Int64'})

	def test_generate_scorer_cms_basic(self, setup_data: pd.DataFrame) -> None:
		"""
		Test that confusion matrices are correctly generated for each scorer.

		Parameters:
		-----------
		setup_data : pd.DataFrame
			A sample DataFrame containing scores for testing.

		Returns:
		--------
		None
		"""
		expected_output = {
			'A': confusion_matrix([3, 2], [3, 2]),
			'B': confusion_matrix([3, 1, 2], [3, 1, 1]),
			'C': confusion_matrix([3, 1, 2], [3, 1, 2])
		}

		result = generate_scorer_cms(setup_data)

		assert result.keys() == expected_output.keys()
		for scorer in expected_output:
			np.testing.assert_array_equal(result[scorer], expected_output[scorer])

	def test_generate_scorer_cms_empty_dataframe(self) -> None:
		"""
		Test that an empty DataFrame returns an empty dictionary of confusion matrices.

		Returns:
		--------
		None
		"""
		empty_df = pd.DataFrame(columns=['Median_Score', 'Scorer', 'Score'])

		result = generate_scorer_cms(empty_df)

		assert result == {}

class TestCmAccuracies:
	@pytest.fixture
	def setup_confusion_matrix(self) -> np.ndarray:
		"""
		Fixture to set up a sample confusion matrix for testing the cm_accuracies function.

		Returns:
		--------
		np.ndarray
			A sample confusion matrix for testing.
		"""
		return np.array([
			[5, 2, 0],
			[1, 6, 1],
			[0, 2, 4]
		])

	@pytest.fixture
	def setup_class_labels(self) -> list:
		"""
		Fixture to set up sample class labels for testing the cm_accuracies function.

		Returns:
		--------
		list
			A list of class labels corresponding to the confusion matrix.
		"""
		return ['Class 0', 'Class 1', 'Class 2']

	def test_exact_accuracy(self, setup_confusion_matrix: np.ndarray, setup_class_labels: list) -> None:
		"""
		Test the exact accuracy calculation from the confusion matrix.

		Parameters:
		-----------
		setup_confusion_matrix : np.ndarray
			A sample confusion matrix.
		setup_class_labels : list
			The class labels corresponding to the confusion matrix.

		Returns:
		--------
		None
		"""
		result = cm_accuracies(setup_confusion_matrix, setup_class_labels, print_results=True)
		expected_accuracy = (5 + 6 + 4) / np.sum(setup_confusion_matrix)
		assert result['exact_accuracy'] == pytest.approx(expected_accuracy, rel=1e-3)

	def test_within_one_accuracy(self, setup_confusion_matrix: np.ndarray, setup_class_labels: list) -> None:
		"""
		Test the within-one accuracy calculation from the confusion matrix.

		Parameters:
		-----------
		setup_confusion_matrix : np.ndarray
			A sample confusion matrix.
		setup_class_labels : list
			The class labels corresponding to the confusion matrix.

		Returns:
		--------
		None
		"""
		result = cm_accuracies(setup_confusion_matrix, setup_class_labels)
		correct_within_one = (5 + 6 + 4) + (2 + 1 + 2 + 1)
		expected_within_one_accuracy = correct_within_one / np.sum(setup_confusion_matrix)
		assert result['within_one_accuracy'] == pytest.approx(expected_within_one_accuracy, rel=1e-3)

	def test_per_class_accuracies(self, setup_confusion_matrix: np.ndarray, setup_class_labels: list) -> None:
		"""
		Test the per-class accuracies calculation from the confusion matrix.

		Parameters:
		-----------
		setup_confusion_matrix : np.ndarray
			A sample confusion matrix.
		setup_class_labels : list
			The class labels corresponding to the confusion matrix.

		Returns:
		--------
		None
		"""
		result = cm_accuracies(setup_confusion_matrix, setup_class_labels)
		expected_class_accuracies = {
			'Class 0': 5 / 7,
			'Class 1': 6 / 8,
			'Class 2': 4 / 6
		}
		assert result['class_accuracies'] == pytest.approx(expected_class_accuracies, rel=1e-3)

	def test_empty_confusion_matrix(self) -> None:
		"""
		Test the cm_accuracies function with an empty confusion matrix.

		Returns:
		--------
		None
		"""
		empty_cm = np.array([[]])
		class_labels = []
		result = cm_accuracies(empty_cm, class_labels)

		assert result['exact_accuracy'] == 0
		assert result['within_one_accuracy'] == 0
		assert result['class_accuracies'] == {}

	def test_single_class_confusion_matrix(self) -> None:
		"""
		Test the cm_accuracies function with a confusion matrix containing only one class.

		Returns:
		--------
		None
		"""
		single_class_cm = np.array([[10]])
		class_labels = ['Class 0']
		result = cm_accuracies(single_class_cm, class_labels)

		# Single class exact accuracy should be 1.0, and per-class accuracy should also be 1.0
		assert result['exact_accuracy'] == 1.0
		assert result['within_one_accuracy'] == 1.0
		assert result['class_accuracies']['Class 0'] == 1.0

class TestPlotConfusionMatrix:
	
	@pytest.fixture
	def setup_conf_matrix(self):
		"""
		Fixture to set up a sample confusion matrix for testing.
		
		Returns:
		--------
		Tuple[np.ndarray, np.ndarray, List[str]]
			A tuple containing a confusion matrix, labels, and axis labels.
		"""
		conf_matrix = np.array([[5, 2], [1, 7]])
		labels = np.array([0, 1])
		axis_labels = ["True Label", "Predicted Label"]
		return conf_matrix, labels, axis_labels

	@mock.patch("matplotlib.pyplot.savefig")
	@mock.patch("matplotlib.pyplot.show")
	def test_plot_confusion_matrix(self, mock_show, mock_savefig, setup_conf_matrix):
		"""
		Test that plot_confusion_matrix runs without errors and calls plt.show().
		
		Parameters:
		-----------
		mock_show : mock.MagicMock
			Mock for plt.show() to avoid actually showing the plot during testing.
		
		Returns:
		--------
		None
		"""
		conf_matrix, labels, axis_labels = setup_conf_matrix
		plot_confusion_matrix(conf_matrix, labels, axis_labels)
		
		mock_show.assert_called_once()
		mock_savefig.assert_called_once()

	@mock.patch("matplotlib.pyplot.show")
	@mock.patch("matplotlib.pyplot.savefig")
	@mock.patch("matplotlib.pyplot.text")
	def test_text_calls(self, mock_text, mock_savefig, mock_show, setup_conf_matrix):
		"""
		Test that plt.text() is called the correct number of times for the confusion matrix.
		
		Parameters:
		-----------
		mock_text : mock.MagicMock
			Mock for plt.text() to check that it was called correctly.
		
		Returns:
		--------
		None
		"""
		conf_matrix, labels, axis_labels = setup_conf_matrix
		plot_confusion_matrix(conf_matrix, labels, axis_labels)
		
		assert mock_text.call_count == conf_matrix.size

		mock_show.assert_called_once()
		mock_savefig.assert_called_once()

class TestFindPairwiseAgreement:
	
	@pytest.fixture
	def setup_data(self) -> pd.DataFrame:
		"""
		Fixture to set up sample data for testing the find_pairwise_agreement function.
		"""
		return pd.DataFrame({
			'Score1': [3, 3, 3, 3, 3, 3, pd.NA],
			'Score2': [3, 2, 1, 3, 1, pd.NA, pd.NA],
			'Score3': [3, 3, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA]
		})

	def test_exact_agreement_all_agree(self, setup_data: pd.DataFrame) -> None:
		"""
		Test exact agreement where all scores agree.
		"""
		row = setup_data.iloc[0]
		result = find_pairwise_agreement(row, precision=0)
		assert result == 1.0

	def test_exact_agreement_some_agree(self, setup_data: pd.DataFrame) -> None:
		"""
		Test exact agreement where some scores agree.
		"""
		row = setup_data.iloc[1]
		result = find_pairwise_agreement(row, precision=0)
		assert result == 0.3333333333333333

	def test_exact_agreement_none_agree(self, setup_data: pd.DataFrame) -> None:
		"""
		Test exact agreement where no scores agree.
		"""
		row = setup_data.iloc[2]
		result = find_pairwise_agreement(row, precision=0)
		assert result == 0.0

	def test_near_miss_agreement_all_agree(self, setup_data: pd.DataFrame) -> None:
		"""
		Test near-miss agreement with precision 1 where all scores agree within the tolerance.
		"""
		row = setup_data.iloc[0]
		result = find_pairwise_agreement(row, precision=1)
		assert result == 1.0

	def test_near_miss_agreement_some_agree(self, setup_data: pd.DataFrame) -> None:
		"""
		Test near-miss agreement with precision 1 where some scores agree within the tolerance.
		"""
		row = setup_data.iloc[1]
		result = find_pairwise_agreement(row, precision=1)
		assert result == 1.0

	def test_near_miss_agreement_none_agree(self, setup_data: pd.DataFrame) -> None:
		"""
		Test near-miss agreement with precision 1 where no scores agree within the tolerance.
		"""
		row = setup_data.iloc[2]
		result = find_pairwise_agreement(row, precision=1)
		assert result == 0.0

	def test_all_missing_scores(self, setup_data: pd.DataFrame) -> None:
		"""
		Test with all scores missing.
		"""
		row = setup_data.iloc[6]
		result = find_pairwise_agreement(row, precision=0)
		assert pd.isna(result)

class TestExtractScorePairs:

	def test_no_missing_values(self):
		data = {'Score1': [10, 40], 'Score2': [20, 50], 'Score3': [30, 60]}
		input_df = pd.DataFrame(data)

		# FIX: Use integers instead of floats to match the expected output dtype
		expected_data = [
			(10, 20), (10, 30), (20, 30),
			(40, 50), (40, 60), (50, 60)
		]
		expected_df = pd.DataFrame(expected_data, columns=["ScoreA", "ScoreB"])

		result_df = extract_score_pairs(input_df)
		pd.testing.assert_frame_equal(result_df, expected_df)

	def test_with_missing_values(self):
		data = {
			'Score1': [10, 40, 70],
			'Score2': [20, np.nan, np.nan],
			'Score3': [30, 60, 90]
		}
		input_df = pd.DataFrame(data)

		expected_data = [
			(10.0, 20.0), (10.0, 30.0), (20.0, 30.0),
			(40.0, 60.0),
			(70.0, 90.0)
		]
		expected_df = pd.DataFrame(expected_data, columns=["ScoreA", "ScoreB"])
		
		result_df = extract_score_pairs(input_df)
		pd.testing.assert_frame_equal(result_df, expected_df)

	def test_empty_dataframe(self):
		input_df = pd.DataFrame(columns=['Score1', 'Score2', 'Score3'])
		expected_df = pd.DataFrame(columns=["ScoreA", "ScoreB"])
		
		result_df = extract_score_pairs(input_df)
		pd.testing.assert_frame_equal(result_df, expected_df, check_dtype=False)

	def test_no_valid_pairs(self):
		data = {
			'Score1': [10, np.nan, 100],
			'Score2': [np.nan, 50, np.nan],
			'Score3': [np.nan, np.nan, np.nan]
		}
		input_df = pd.DataFrame(data)
		expected_df = pd.DataFrame([], columns=["ScoreA", "ScoreB"])

		result_df = extract_score_pairs(input_df)
		pd.testing.assert_frame_equal(result_df, expected_df, check_dtype=False)