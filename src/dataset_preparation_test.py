import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
from unittest.mock import patch, MagicMock

from dataset_preparation import read_CDAScorer_data, _fill_scores, combine_CDA_dataframes, clean_coordinates, clean_scores, add_median_column, _overlap, _find_overlapping_coordinates, _find_closest_coordinates, get_central_scorer, save_cropped_images

class TestReadCDAScorerData:
	
	@pytest.fixture
	def tmp_path(self, tmp_path_factory: pytest.TempPathFactory) -> Path:
		"""
		Fixture to provide a temporaty directory for testing

		Parameters:
		-----------
		tmp_path_factory : pytest.TempPathFactory
			The pytest fixture factory for creating temporary paths
		
		Returns:
		--------
		Path
			A temporary directory path
		"""
		return tmp_path_factory.mktemp("data")
	
	def test_valid_folder_with_csv_files(self, tmp_path: Path) -> None:
		"""
		Test to read valid CSV files from a folder

		Parameters:
		-----------
		tmp_path : Path
			Path to the temporary directory containing mock CSV files
		
		Returns:
		--------
		None
		"""
		csv_content = "img,maxrow,maxcol,row,col,pos,score,x1,x2,y1,y2\nimage1,5,5,1,1,1,10,10,20,20,30"
		
		csv_file1 = tmp_path / "scorer1.csv"
		csv_file2 = tmp_path / "scorer2.csv"
		
		csv_file1.write_text(csv_content)
		csv_file2.write_text(csv_content)
		
		result = read_CDAScorer_data(tmp_path)
		
		assert isinstance(result, dict)
		assert "scorer1" in result and "scorer2" in result
		assert isinstance(result["scorer1"], pd.DataFrame)
		assert result["scorer1"].shape == (1, 11)

	def test_folder_not_existing(self) -> None:
		"""
		Test handling of a non-existent folder

		Returns:
		--------
		None
		"""
		with pytest.raises(FileNotFoundError):
			read_CDAScorer_data("non_existent_folder")

	def test_no_csv_files(self, tmp_path: Path) -> None:
		"""
		Test handling of a folder with no CSV files

		Parameters:
		-----------
		tmp_path : Path
			Path to the temporary directory which will be empty

		Returns:
		--------
		None        
		"""
		with pytest.raises(ValueError, match="No CSV files were found"):
			read_CDAScorer_data(tmp_path)

	def test_csv_with_incorrect_columns(self, tmp_path: Path) -> None:
		"""
		Test handling of a CSV file with incorrect columns

		Parameters:
		-----------
		tmp_path : Path
			Path to the temporary directory containing a CSV with incorrect columns
		
		Returns:
		--------
		None
		"""
		csv_content = "wrong_column\nvalue"
		
		csv_file = tmp_path / "scorer1.csv"
		csv_file.write_text(csv_content)
		
		with pytest.raises(ValueError, match="incorrect columns"):
			read_CDAScorer_data(tmp_path)

	def test_mixed_file_types(self, tmp_path: Path) -> None:
		"""
		Test handling of a folder with mixed file types (CSV and non-CSV files)

		Parameters:
		-----------
		tmp_path : Path
			Path to the temporary directory containing both valid CSV and invalid non-CSV files
		
		Returns:
		--------
		None
		"""
		csv_content = "img,maxrow,maxcol,row,col,pos,score,x1,x2,y1,y2\nimage1,5,5,1,1,1,10,10,20,20,30"
		csv_file = tmp_path / "scorer1.csv"
		non_csv_file = tmp_path / "not_a_csv.txt"
		
		csv_file.write_text(csv_content)
		non_csv_file.write_text("This is not a CSV")
		
		result = read_CDAScorer_data(tmp_path)
		
		assert isinstance(result, dict)
		assert "scorer1" in result
		assert "not_a_csv" not in result

class TestFillScores:

	@pytest.fixture
	def setup_data(self) -> Tuple[Dict[str, pd.DataFrame], pd.Series]:
		"""
		Fixture to set up mock CDA dataframes and a sample row for testing

		Returns:
		--------
		Tuple[Dict[str, pd.DataFrame], pd.Series]
			A tuple where the first element is a dictionary of CDA dataframes and the second
			element is a pandas Series representing a row
		"""
		scorer1_data = pd.DataFrame({
			'img': ['image1', 'image2'],
			'row': [1, 2],
			'col': [1, 1],
			'pos': [1, 1],
			'score': [2, 3],
			'x1': [10, 15],
			'x2': [20, 25],
			'y1': [30, 35],
			'y2': [40, 45]
		})

		scorer2_data = pd.DataFrame({
			'img': ['image3'],
			'row': [1],
			'col': [2],
			'pos': [1],
			'score': [4],
			'x1': [20],
			'x2': [30],
			'y1': [40],
			'y2': [50]
		})

		cda_dataframes = {
			'scorer1': scorer1_data,
			'scorer2': scorer2_data
		}

		row = pd.Series({
			'img': 'image1',
			'row': 1,
			'col': 1,
			'pos': 1,
			'scorer1': 'scorer1',
			'scorer2': 'scorer2'
		})

		return cda_dataframes, row

	def test_fill_scores_valid_data(self, setup_data: Tuple[Dict[str, pd.DataFrame], pd.Series]) -> None:
		"""
		Test _fill_scores function with valid data for scorer number 1

		Parameters:
		-----------
		setup_data: Tuple[Dict[str, pd.DataFrame], pd.Series]
			A tuple containing CDA dataframes and a sample row
		
		Returns:
		--------
		None
		"""
		cda_dataframes, row = setup_data
		result = _fill_scores(cda_dataframes, row, scorer_number=1)
		expected = [2, 10, 20, 30, 40]
		assert result == expected

	def test_fill_scores_valid_data_scorer2(self, setup_data: Tuple[Dict[str, pd.DataFrame], pd.Series]) -> None:
		"""
		Test _fill_scores function with valid data for scorer number 2 (different columns to scorer number 1)

		Parameters:
		-----------
		setup_data: Tuple[Dict[str, pd.DataFrame], pd.Series]
			A tuple containing CDA dataframes and a sample row
		
		Returns:
		--------
		None
		"""
		cda_dataframes, row = setup_data
		row['img'] = 'image3'
		row['row'] = 1
		row['col'] = 2
		row['pos'] = 1
		result = _fill_scores(cda_dataframes, row, scorer_number=2)
		expected = [4, 20, 30, 40, 50]
		assert result == expected

	def test_fill_scores_no_match(self, setup_data: Tuple[Dict[str, pd.DataFrame], pd.Series]) -> None:
		"""
		Test _fill_scores function where there is no match in the data for scorer number 1

		Parameters:
		-----------
		setup_data: Tuple[Dict[str, pd.DataFrame], pd.Series]
			A tuple containing CDA dataframes and a sample row
		
		Returns:
		--------
		None
		"""
		cda_dataframes, row = setup_data
		row['img'] = 'nonexistent_image'
		result = _fill_scores(cda_dataframes, row, scorer_number=1)
		expected = [None, None, None, None, None]
		assert result == expected

	def test_fill_scores_invalid_scorer_number(self, setup_data: Tuple[Dict[str, pd.DataFrame], pd.Series]) -> None:
		"""
		Test _fill_scores function with an invalid scorer number

		Parameters:
		-----------
		setup_data: Tuple[Dict[str, pd.DataFrame], pd.Series]
			A tuple containing CDA dataframes and a sample row
		
		Returns:
		--------
		None
		"""
		cda_dataframes, row = setup_data
		
		# Set an invalid scorer number
		result = _fill_scores(cda_dataframes, row, scorer_number=999)
		expected = [None, None, None, None, None]
		assert result == expected

class TestCombineCDADataframes:
	
	@pytest.fixture
	def setup_data(self, tmp_path: Path) -> tuple[Dict[str, pd.DataFrame], Path]:
		"""
		Fixture to set up mock CDA dataframes and a scorer_info CSV file.

		Parameters:
		-----------
		setup_data : tuple[Dict[str, pd.Dataframe], Path]
			A tuple containing CDA dataframes and the path to the scorer_info CSV file.

		Returns:
		--------
		None
		"""
		scorer1_data = pd.DataFrame({
			'img': ['image1', 'image2'],
			'row': [1, 2],
			'col': [1, 1],
			'pos': [1, 1],
			'score': [10, 20],
			'x1': [10, 15],
			'x2': [20, 25],
			'y1': [30, 35],
			'y2': [40, 45],
			'maxrow': [5, 5],
			'maxcol': [5, 5]
		})

		scorer2_data = pd.DataFrame({
			'img': ['image1', 'image3'],
			'row': [1, 1],
			'col': [1, 2],
			'pos': [1, 1],
			'score': [30, 40],
			'x1': [20, 25],
			'x2': [30, 35],
			'y1': [40, 45],
			'y2': [50, 55],
			'maxrow': [5, 5],
			'maxcol': [5, 5]
		})

		cda_dataframes = {
			'scorer1': scorer1_data,
			'scorer2': scorer2_data
		}

		scorer_info = pd.DataFrame({
			'basename': ['image1', 'image2', 'image3'],
			'scorer1': ['scorer1', 'scorer2', 'scorer3'],
			'scorer2': ['scorer2', 'scorer3', 'scorer1'],
			'scorer3': ['scorer3', 'scorer1', 'scorer2']
		})
		scorer_info_path = tmp_path / "scorer_info.csv"
		scorer_info.to_csv(scorer_info_path, index=False)

		return cda_dataframes, scorer_info_path

	def test_combine_cda_dataframes(self, setup_data: tuple[Dict[str, pd.DataFrame], Path]) -> None:
		"""
		Test combine_CDA_dataframes function with mock CDA dataframes and scorer_info.

		Parameters:
		-----------
		setup_data : tuple[Dict[str, pd.Dataframe], Path]
			A tuple containing CDA dataframes and the path to the scorer_info CSV file.

		Returns:
		--------
		None
		"""
		cda_dataframes, scorer_info_path = setup_data
		result = combine_CDA_dataframes(cda_dataframes, scorer_info_path)
		result.to_csv('result_output.csv', index=False)

		expected_data = {
			'Basename': ['image1', 'image2', 'image3'],
			'MaxRow': [5, 5, 5],
			'MaxCol': [5, 5, 5],
			'Row': [1, 2, 1],
			'Col': [1, 1, 2],
			'Pos': [1, 1, 1],
			'Scorer1': ['scorer1', 'scorer2', 'scorer3'],
			'Score1': [10, pd.NA, pd.NA],
			'X1_1': [10, pd.NA, pd.NA],
			'X2_1': [20, pd.NA, pd.NA],
			'Y1_1': [30, pd.NA, pd.NA],
			'Y2_1': [40, pd.NA, pd.NA],
			'Scorer2': ['scorer2', 'scorer3', 'scorer1'],
			'Score2': [30, pd.NA, pd.NA],
			'X1_2': [20, pd.NA, pd.NA],
			'X2_2': [30, pd.NA, pd.NA],
			'Y1_2': [40, pd.NA, pd.NA],
			'Y2_2': [50, pd.NA, pd.NA],
			'Scorer3': ['scorer3', 'scorer1', 'scorer2'],
			'Score3': [pd.NA, 20, 40],
			'X1_3': [pd.NA, 15, 25],
			'X2_3': [pd.NA, 25, 35],
			'Y1_3': [pd.NA, 35, 45],
			'Y2_3': [pd.NA, 45, 55]
		}
		expected_df = pd.DataFrame(expected_data)

		int_columns = ['Score1', 'Score2', 'Score3', 'X1_1', 'X2_1', 'Y1_1', 'Y2_1', 'X1_2', 'X2_2', 'Y1_2', 'Y2_2', 'X1_3', 'X2_3', 'Y1_3', 'Y2_3']
		expected_df[int_columns] = expected_df[int_columns].astype('Int64')

		pd.testing.assert_frame_equal(result, expected_df)

	def test_invalid_scorer_info_path(self, setup_data: tuple[Dict[str, pd.DataFrame], Path]) -> None:
		"""
		Test combine_CDA_dataframes function with non-existent filepath

		Parameters:
		-----------
		setup_data : tuple[Dict[str, pd.Dataframe], Path]
			A tuple containing CDA dataframes and the path to the scorer_info CSV file.

		Returns:
		--------
		None
		"""
		cda_dataframes, _ = setup_data
		invalid_scorer_info_file = Path("non_existent_scorer_info.csv")

		with pytest.raises(FileNotFoundError):
			combine_CDA_dataframes(cda_dataframes, invalid_scorer_info_file)

class TestCleanCoordinates:

	@pytest.fixture
	def setup_data(self) -> pd.DataFrame:
		"""
		Fixture to set up sample data for testing the clean_coordinates function.

		Returns:
		--------
		pd.DataFrame
			A DataFrame containing rows with various coordinate cases for testing.
		"""
		df = pd.DataFrame({
			'Score1': [5, 7, 8, 6],
			'X1_1': [0, 10, 5, 5],
			'X2_1': [10, 20, 15, 15],
			'Y1_1': [0, 10, 5, 5],
			'Y2_1': [10, 20, 15, 15],
			'Score2': [8, 6, 9, 7],
			'X1_2': [20, 15, 10, 5],
			'X2_2': [30, 25, 20, 15],
			'Y1_2': [20, 15, 10, 5],
			'Y2_2': [30, 25, 20, 15],

			'Score3': [6, 9, 7, 8],
			'X1_3': [5, 5, 5, 5],
			'X2_3': [15, 15, 15, 15],
			'Y1_3': [5, 5, 5, 5],
			'Y2_3': [15, 15, 15, 15],
		})
		df = df.astype('Int64')

		return df

	def test_clean_coordinates_no_change(self, setup_data: pd.DataFrame) -> None:
		"""
		Test clean_coordinates function where no cleaning is required.

		Parameters:
		-----------
		setup_data : pd.DataFrame
			A DataFrame containing sample rows for testing.

		Returns:
		--------
		None
		"""
		data = setup_data.copy()
		result = data.apply(clean_coordinates, axis=1)
		pd.testing.assert_frame_equal(result, data)


	def test_clean_coordinates_with_nan_coordinate(self, setup_data: pd.DataFrame) -> None:
		"""
		Test clean_coordinates function with valid data.

		Parameters:
		-----------
		setup_data : pd.DataFrame
			A DataFrame containing sample rows for testing.

		Returns:
		--------
		None
		"""
		row = setup_data.iloc[0].copy()
		row['X2_1'] = pd.NA
		cleaned_row = clean_coordinates(row)

		assert all(pd.isna(cleaned_row[col]) for col in ['X1_1', 'X2_1', 'Y1_1', 'Y2_1', 'Score1'])


	def test_clean_coordinates_fix_coordinates(self, setup_data: pd.DataFrame) -> None:
		"""
		Test clean_coordinates function to ensure coordinates are adjusted correctly
		so that x2 > x1 and y2 > y1.

		Parameters:
		-----------
		setup_data : pd.DataFrame
			A DataFrame containing sample rows for testing.

		Returns:
		--------
		None
		"""
		data = setup_data.copy()
		data.at[0, 'X1_1'] = 10
		data.at[0, 'X2_1'] = 5
		data.at[0, 'Y1_1'] = 10
		data.at[0, 'Y2_1'] = 5

		result = data.apply(clean_coordinates, axis=1)
		
		assert result.at[0, 'X1_1'] < result.at[0, 'X2_1']
		assert result.at[0, 'Y1_1'] < result.at[0, 'Y2_1']

	def test_clean_coordinates_error(self, setup_data: pd.DataFrame) -> None:
		"""
		Test clean_coordinates function to ensure ValueError is raised for non-square coordinates.

		Parameters:
		-----------
		setup_data : pd.DataFrame
			A DataFrame containing sample rows for testing.

		Returns:
		--------
		None
		"""
		data = setup_data.copy()
		data.at[2, 'X2_1'] = 20
		with pytest.raises(ValueError, match="Coord set is not square"):
			data.apply(clean_coordinates, axis=1)

class TestCleanScores:

	@pytest.fixture
	def setup_data(self) -> pd.DataFrame:
		"""
		Fixture to set up sample data for testing the clean_scores function.

		Returns:
		--------
		pd.DataFrame
			A DataFrame containing rows with various score cases for testing.
		"""
		df = pd.DataFrame({
			'Score1': [5, 7, 8, 0, 6, 1],
			'Score2': [2, 7, 4, 6, 3, 9],
			'Score3': [8, 8, 7, 5, 8, 9],
		})
		df = df.astype('Int64')
		return df

	def test_clean_scores_no_rows_removed(self, setup_data: pd.DataFrame) -> None:
		"""
		Test clean_scores function to ensure no rows are removed if they meet criteria for keeping.

		Parameters:
		-----------
		setup_data : pd.DataFrame
			A DataFrame containing sample rows for testing.

		Returns:
		--------
		None
		"""
		data = setup_data.copy()
		result = clean_scores(data)
		pd.testing.assert_frame_equal(result, data)

	def test_clean_scores_remove_few_annotations(self, setup_data: pd.DataFrame) -> None:
		"""
		Test clean_scores function to remove rows with fewer than two non-null annotations.

		Parameters:
		-----------
		setup_data : pd.DataFrame
			A DataFrame containing sample rows for testing.

		Returns:
		--------
		None
		"""
		data = setup_data.copy()
		data.iloc[0, :] = pd.NA  # Modify the first row to have all NA values
		data.iloc[1, 1:] = pd.NA  # Modify the second row to have two NA values
		result = clean_scores(data)
		expected_result = data.drop([0, 1]).reset_index(drop=True)
		pd.testing.assert_frame_equal(result, expected_result)

	def test_clean_scores_remove_disagreeing_scores(self, setup_data: pd.DataFrame) -> None:
		"""
		Test clean_scores function to remove rows with exactly two disagreeing scores.

		Parameters:
		-----------
		setup_data : pd.DataFrame
			A DataFrame containing sample rows for testing.

		Returns:
		--------
		None
		"""
		data = setup_data.copy()
		data.iloc[0, 0] = pd.NA
		result = clean_scores(data)
		expected_result = data.drop(0).reset_index(drop=True)
		pd.testing.assert_frame_equal(result, expected_result)

class TestAddMedianColumn:
	@pytest.fixture
	def setup_data(self) -> pd.DataFrame:
		"""
		Fixture to set up sample data for testing the add_median_column function.

		Returns:
		--------
		pd.DataFrame
			A DataFrame containing rows with sample scores for testing.
		"""
		df = pd.DataFrame({
			'Score1': [5, 7, 8, 0, 6, 1],
			'Score2': [2, 7, 4, 6, 3, 9],
			'Score3': [8, 8, 7, 5, 8, 9],
		})
		df = df.astype('Int64')
		return df

	def test_add_median_column_correct_data(self, setup_data: pd.DataFrame) -> None:
		"""
		Test add_median_column function with correct data to ensure the median column is calculated correctly.

		Parameters:
		-----------
		setup_data : pd.DataFrame
			A DataFrame containing sample rows for testing.

		Returns:
		--------
		None
		"""
		data = setup_data.copy()
		result = add_median_column(data)
		
		expected_medians = data[['Score1', 'Score2', 'Score3']].median(axis=1)
		expected_medians.name = 'Median_Score'
		pd.testing.assert_series_equal(result['Median_Score'], expected_medians)

	def test_add_median_column_with_na(self, setup_data: pd.DataFrame) -> None:
		"""
		Test add_median_column function with a missing value to ensure the median is correctly calculated with NA.

		Parameters:
		-----------
		setup_data : pd.DataFrame
			A DataFrame containing sample rows for testing.

		Returns:
		--------
		None
		"""
		data = setup_data.copy()
		data.at[1, 'Score3'] = pd.NA
		result = add_median_column(data)
		
		expected_medians = data[['Score1', 'Score2', 'Score3']].median(axis=1)
		expected_medians.name = 'Median_Score'
		pd.testing.assert_series_equal(result['Median_Score'], expected_medians)

class TestOverlap:
	
	@pytest.fixture
	def sample_rectangles(self) -> dict:
		"""
		Fixture to provide sample rectangles for testing.

		Returns:
		--------
		dict
			A dictionary with sample rectangles.
		"""
		return {
			'rect1': [0, 5, 0, 5],
			'rect2': [3, 8, 3, 8],
			'rect3': [6, 10, 6, 10],
			'rect4': [1, 4, 1, 4]
		}

	def test_overlap_true(self, sample_rectangles: dict) -> None:
		"""
		Test that the function correctly identifies overlapping rectangles.

		Parameters:
		-----------
		sample_rectangles : dict
			A dictionary containing sample rectangles for testing.

		Returns:
		--------
		None
		"""
		rect1 = sample_rectangles['rect1']
		rect2 = sample_rectangles['rect2']
		result = _overlap(rect1, rect2)
		assert result is True

	def test_overlap_false(self, sample_rectangles: dict) -> None:
		"""
		Test that the function correctly identifies non-overlapping rectangles.

		Parameters:
		-----------
		sample_rectangles : dict
			A dictionary containing sample rectangles for testing.

		Returns:
		--------
		None
		"""
		rect1 = sample_rectangles['rect1']
		rect3 = sample_rectangles['rect3']
		result = _overlap(rect1, rect3)
		assert result is False

	def test_overlap_with_inner_rectangle(self, sample_rectangles: dict) -> None:
		"""
		Test that the function correctly identifies an inner rectangle as overlapping.

		Parameters:
		-----------
		sample_rectangles : dict
			A dictionary containing sample rectangles for testing.

		Returns:
		--------
		None
		"""
		rect1 = sample_rectangles['rect1']
		rect4 = sample_rectangles['rect4']
		result = _overlap(rect1, rect4)
		assert result is True

class TestFindOverlappingCoordinates:
	
	@pytest.fixture
	def coord_lists(self) -> List[List[int]]:
		"""
		Fixture to provide a set of coordinates where all rectangles overlap.

		Returns:
		--------
		List[List[int]]
			A list of coordinate sets where all rectangles overlap.
		"""
		return [
			[0, 5, 0, 5],  
			[2, 6, 2, 6],  
			[3, 7, 3, 7],  
		]

	def test_all_rectangles_overlap(self, coord_lists: List[List[int]]) -> None:
		"""
		Test that all rectangles overlap and the function returns all coordinates.

		Parameters:
		-----------
		coord_lists : List[List[int]]
			A fixture providing a set of overlapping rectangle coordinates.

		Returns:
		--------
		None
		"""
		result = _find_overlapping_coordinates(coord_lists)
		assert result is not None
		assert len(result) == 3
		assert coord_lists[0] in result and coord_lists[1] in result and coord_lists[2] in result

	def test_two_rectangles_overlap(self, coord_lists: List[List[int]]) -> None:
		"""
		Test that when one rectangle does not overlap, only the overlapping rectangles are returned.

		Parameters:
		-----------
		coord_lists : List[List[int]]
			A fixture providing a set of rectangle coordinates, two of which overlap.

		Returns:
		--------
		None
		"""
		coord_lists[2] = [10, 15, 10, 15]
		result = _find_overlapping_coordinates(coord_lists)
		assert result is not None
		assert len(result) == 2
		assert coord_lists[0] in result and coord_lists[1] in result
		assert coord_lists[2] not in result

	def test_no_rectangles_overlap(self, coord_lists: List[List[int]]) -> None:
		"""
		Test that when no rectangles overlap, the function returns None.

		Parameters:
		-----------
		coord_lists : List[List[int]]
			A fixture providing a set of rectangle coordinates, none of which overlap.

		Returns:
		--------
		None
		"""
		coord_lists[1] = [10, 15, 10, 15]
		coord_lists[2] = [20, 25, 20, 25]
		result = _find_overlapping_coordinates(coord_lists)
		assert result is None

class TestFindClosestCoordinates:

	@pytest.fixture
	def coord_lists(self) -> List[List[float]]:
		"""
		Fixture to provide a list of coordinate sets for testing the find_closest_coordinates function.

		Returns:
		--------
		List[List[float]]
			A list of coordinate sets representing rectangles.
		"""
		return [
			[0, 5, 0, 5],
			[2, 6, 2, 6],
			[3, 7, 3, 7],
		]

	def test_find_closest_coordinates_all_overlap(self, coord_lists: List[List[float]]) -> None:
		"""
		Test find_closest_coordinates function where all rectangles overlap.

		Parameters:
		-----------
		coord_lists : List[List[float]]
			A list of coordinate sets representing rectangles.

		Returns:
		--------
		None
		"""
		result = _find_closest_coordinates(coord_lists)
		expected = [2, 6, 2, 6]
		assert result == expected

	def test_find_closest_coordinates_some_overlap(self, coord_lists: List[List[float]]) -> None:
		"""
		Test find_closest_coordinates function where only some rectangles overlap.

		Parameters:
		-----------
		coord_lists : List[List[float]]
			A list of coordinate sets representing rectangles.

		Returns:
		--------
		None
		"""
		coords = coord_lists.copy()
		coords[2] = [20, 25, 20, 25]
		result = _find_closest_coordinates(coords)
		expected = [2, 6, 2, 6]
		assert result == expected

	def test_find_closest_coordinates_none_overlap(self, coord_lists: List[List[float]]) -> None:
		"""
		Test find_closest_coordinates function where no rectangles overlap.

		Parameters:
		-----------
		coord_lists : List[List[float]]
			A list of coordinate sets representing rectangles.

		Returns:
		--------
		None
		"""
		coords = coord_lists.copy()
		coords[1] = [12, 20, 12, 20]
		coords[2] = [22, 30, 22, 30]
		result = _find_closest_coordinates(coords)
		assert result is None
	
	def test_find_closest_coordinates_equally_close(self, coord_lists: List[List[float]]) -> None:
		"""
		Test find_closest_coordinates function where two coordinate sets are equally as close to the centre and are different sizes

		Parameters:
		-----------
		coord_lists : List[List[float]]
			A list of coordinate sets representing rectangles.

		Returns:
		--------
		None
		"""
		coords = coord_lists.copy()
		coords[0] = [0, 8, 0, 8]
		coords[1] = [3, 5, 3, 5]
		coords[2] = [20, 25, 20, 25]
		result = _find_closest_coordinates(coords)
		assert result == [0, 8, 0, 8]

class TestGetCentralScorer:
	@pytest.fixture
	def coord_fixture(self) -> pd.Series:
		"""
		Fixture providing a pd.Series with overlapping rectangles and no NA values.

		Returns:
		--------
		pd.Series
			A pandas Series containing coordinates for three overlapping rectangles.
		"""
		return pd.Series({
			"X1_1": 0, "X2_1": 5, "Y1_1": 0, "Y2_1": 5,
			"X1_2": 3, "X2_2": 8, "Y1_2": 3, "Y2_2": 8,
			"X1_3": 2, "X2_3": 6, "Y1_3": 2, "Y2_3": 6
		})

	def test_get_central_scorer_valid_overlap(self, coord_fixture: pd.Series) -> None:
		"""
		Test get_central_scorer function with overlapping rectangles.

		Parameters:
		-----------
		coord_fixture : pd.Series
			A pandas Series containing coordinates for three overlapping rectangles.

		Returns:
		--------
		None
		"""
		result = get_central_scorer(coord_fixture)
		assert result == 3

	def test_get_central_scorer_with_na(self, coord_fixture: pd.Series) -> None:
		"""
		Test get_central_scorer function where one set of coordinates is NA.

		Parameters:
		-----------
		coord_fixture : pd.Series
			A pandas Series containing coordinates for rectangles, with one set containing NA values.

		Returns:
		--------
		None
		"""
		data = coord_fixture.copy()
		data["X1_2"], data["X2_2"], data["Y1_2"], data["Y2_2"] = [pd.NA] * 4
		result = get_central_scorer(data)
		assert result == 3

	def test_get_central_scorer_no_overlap(self, coord_fixture: pd.Series) -> None:
		"""
		Test get_central_scorer function with non-overlapping rectangles.

		Parameters:
		-----------
		coord_fixture : pd.Series
			A pandas Series containing coordinates for rectangles that do not overlap.

		Returns:
		--------
		None
		"""
		data = coord_fixture.copy()
		data.update({
			"X1_1": 0, "X2_1": 1, "Y1_1": 0, "Y2_1": 1,
			"X1_2": 2, "X2_2": 3, "Y1_2": 2, "Y2_2": 3,
			"X1_3": 4, "X2_3": 5, "Y1_3": 4, "Y2_3": 5
		})
		result = get_central_scorer(data)
		assert result is None

class TestSaveCroppedImages:
	@pytest.fixture
	def tmp_path(self, tmp_path_factory: pytest.TempPathFactory) -> Path:
		"""
		Fixture to provide a temporaty directory for testing

		Parameters:
		-----------
		tmp_path_factory : pytest.TempPathFactory
			The pytest fixture factory for creating temporary paths
		
		Returns:
		--------
		Path
			A temporary directory path
		"""
		return tmp_path_factory.mktemp("data")

	@pytest.fixture
	def data_fixture(self) -> pd.DataFrame:
		"""
		Fixture to provide a sample DataFrame with image information for testing.
		
		Returns:
		--------
		pd.DataFrame
			A DataFrame containing columns like 'Centre_Coords', 'X1_1', 'X2_1', 'Y1_1', 
			'Y2_1', 'Basename', 'Scorer1', 'Row', 'Col', 'Pos', and 'Median_Score'.
		"""
		data = pd.DataFrame({
			"Centre_Coords": [1, 1],
			"X1_1": [10, 15],
			"X2_1": [50, 55],
			"Y1_1": [10, 15],
			"Y2_1": [50, 55],
			"Basename": ["image1.jpg", "image1.jpg"],
			"Scorer1": ["scorer1", "scorer1"],
			"Row": [1, 2],
			"Col": [1, 2],
			"Pos": [1, 2],
			"Median_Score": [80, 90]
		})
		return data

	@patch("os.path.exists")
	@patch("cv2.imread")
	@patch("cv2.imwrite")
	@patch("builtins.print")
	def test_save_cropped_images(self, mock_print: MagicMock, mock_imwrite:
		 MagicMock, mock_imread: MagicMock, mock_exists: MagicMock, data_fixture: pd.DataFrame, tmp_path: Path) -> None:
		"""
		Test the `save_cropped_images` function for successful image reading, cropping, and saving.
		
		This test simulates reading and writing images using mocks and asserts that the correct functions 
		(imread, imwrite, and print) are called as expected.

		Parameters:
		-----------
		mock_print : MagicMock
			Mock for the `print` function.
		mock_imwrite : MagicMock
			Mock for the `cv2.imwrite` function.
		mock_imread : MagicMock
			Mock for the `cv2.imread` function.
		mock_exists : MagicMock
			Mock for the `os.path.exists` function.
		data_fixture : pd.DataFrame
			The fixture providing sample data for testing.

		Returns:
		--------
		None
		"""
		
		mock_exists.side_effect = lambda path: path.endswith("image1.jpg")
		mock_imread.return_value = np.ones((100, 100, 3), dtype=np.uint8)
		mock_imwrite.return_value = True

		out_folder = tmp_path
		save_cropped_images(data_fixture, out_folder)

		assert mock_imread.called
		assert mock_imwrite.called
		assert mock_print.called

	@patch("os.path.exists")
	@patch("cv2.imread")
	@patch("cv2.imwrite")
	@patch("builtins.print")

	def test_image_not_found(self, mock_print: MagicMock, mock_imwrite: MagicMock, mock_imread: MagicMock, mock_exists: MagicMock, data_fixture: pd.DataFrame, tmp_path: Path) -> None:
		"""
		Test the `save_cropped_images` function when images are not found.

		This test simulates the scenario where no image files are found for the given paths,
		and asserts that the appropriate print message indicating the missing image is called.

		Parameters:
		-----------
		mock_print : MagicMock
			Mock for the `print` function.
		mock_imwrite : MagicMock
			Mock for the `cv2.imwrite` function.
		mock_imread : MagicMock
			Mock for the `cv2.imread` function.
		mock_exists : MagicMock
			Mock for the `os.path.exists` function.
		data_fixture : pd.DataFrame
			The fixture providing sample data for testing.

		Returns:
		--------
		None
		"""
		mock_exists.side_effect = lambda path: False
		mock_imread.return_value = None
		mock_imwrite.return_value = True

		out_folder = tmp_path
		save_cropped_images(data_fixture, out_folder)

		assert not mock_imread.called
		assert mock_print.called
		assert any("Image not found for" in call[0][0] for call in mock_print.call_args_list)