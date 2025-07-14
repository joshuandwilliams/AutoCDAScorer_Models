import pandas as pd
import numpy as np
from typing import Dict, Union, List, Optional
from pathlib import Path
import re
import os
import cv2 as cv
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.colors import black

def read_CDAScorer_data(folder_path: Union[str, Path]) -> Dict[str, pd.DataFrame]:
	"""
	Read in the CDA datatables collected with CDAScorer by different scorers

	Parameters:
	-----------
	folder_path : Union[str, Path]
		The path to the folder containing the CSV files
	
	Returns:
	--------
	Dict[str, pd.DataFrame]
		A dictionary where each key is the name of a CSV file (without the extension)
		and the value is the corresponding dataframe.
	"""
	folder_path = Path(folder_path)
	if not folder_path.exists():
		raise FileNotFoundError(f"The folder '{folder_path}' does not exist.")
	
	csv_filepaths = list(folder_path.glob('*.csv'))
	if not csv_filepaths:
		raise ValueError(f"No CSV files were found in the folder '{folder_path}'")

	expected_columns = ['img', 'maxrow', 'maxcol', 'row', 'col', 'pos', 'score', 'x1', 'x2', 'y1', 'y2']
	dataframes = {}

	for csv_file in csv_filepaths:
		df = pd.read_csv(
			csv_file,
			dtype={
				'img': 'string',
				'row': 'int',
				'col': 'int',
				'pos': 'int',
				'maxrow': 'int',
				'maxcol': 'int',
				'score': 'int',
				'x1': 'int',
				'x2': 'int',
				'y1': 'int',
				'y2': 'int'
			}
		)
		if list(df.columns) != expected_columns:
			raise ValueError(f"CSV file '{csv_file}' has incorrect columns: {list(df.columns)}")
		
		scorer_name = csv_file.stem
		dataframes[scorer_name] = df
	
	return dataframes

def _fill_scores(cda_dataframes: Dict[str, pd.DataFrame], row: pd.Series, scorer_number: int) -> List[Union[int, None]]:
	"""
	Fill the scores and coordinates for a specific scorer.

	Parameters:
	-----------
	dataframes : Dict[str, pd.DataFrame]
		A dictionary where keys are scorer names and values are DataFrames containing the scoring data.
	
	row : pd.Series
		A row from the output DataFrame that contains the information about Basename, Row, Col, Pos, and Scorer.

	scorer_number : int
		The scorer number (1, 2, or 3) indicating which scorer's data to fill.

	Returns:
	--------
	List[Union[float, None]]
		A list containing the score and coordinates [score, x1, x2, y1, y2]. 
		If no match is found, returns [None, None, None, None, None].
	"""
	scorer_key = row.get(f'scorer{scorer_number}')
	
	if pd.isna(scorer_key) or scorer_key not in cda_dataframes:
		return [None, None, None, None, None]

	scorer_data = cda_dataframes[scorer_key]

	matching_row = scorer_data[
		(scorer_data['img'] == row['img']) &
		(scorer_data['row'] == row['row']) &
		(scorer_data['col'] == row['col']) &
		(scorer_data['pos'] == row['pos'])
	]
	if not matching_row.empty:
		return matching_row[['score', 'x1', 'x2', 'y1', 'y2']].values[0].tolist()
	return [None, None, None, None, None]

def combine_CDA_dataframes(cda_dataframes: Dict[str, pd.DataFrame], scorer_info: Union[str, Path]) -> pd.DataFrame:
	"""
	Combine the CDA dataframes into a single output table

	Parameters:
	-----------
	cda_dataframes : Dict[str, pd.DataFrame]
		The dataframe containing as keys the CSV filenames and as the values
		the contents of the CSVs.

	scorer_info : Union[str, Path]
		The path to the scoring assignment info
	
	Returns:
	--------
	pd.DataFrame
		A combined dataframe where all of the results are collated together,
		with data collected for the same CDA recorded in the same row
	"""
	for key in cda_dataframes:
		cda_dataframes[key]['img'] = cda_dataframes[key]['img'].apply(lambda x: re.split(r'[\\/]', x)[-1])

	combined_df = pd.concat(cda_dataframes.values(), ignore_index = True)

	unique_combinations = combined_df[['img', 'row', 'col', 'pos']].drop_duplicates().sort_values(by=['img', 'row', 'col', 'pos'])

	output_table = pd.DataFrame(unique_combinations)

	max_values = combined_df[['img', 'maxrow', 'maxcol']].drop_duplicates().set_index('img')
	max_values = max_values[~max_values.index.duplicated(keep='first')]
	output_table = output_table.join(max_values, on='img')

	randomised_info = pd.read_csv(scorer_info)
	output_table = output_table.merge(randomised_info[['basename', 'scorer1', 'scorer2', 'scorer3']], left_on='img', right_on='basename', how='left')

	for i in range(1, 4):
		output_table[[f'score{i}', f'x1_{i}', f'x2_{i}', f'y1_{i}', f'y2_{i}']] = output_table.apply(lambda row: pd.Series(_fill_scores(cda_dataframes, row, i)), axis=1)

	for i in range(1, 4):
		output_table[f'score{i}'] = output_table[f'score{i}'].fillna(pd.NA).astype('Int64')
		output_table[f'x1_{i}'] = output_table[f'x1_{i}'].fillna(pd.NA).astype('Int64')
		output_table[f'x2_{i}'] = output_table[f'x2_{i}'].fillna(pd.NA).astype('Int64')
		output_table[f'y1_{i}'] = output_table[f'y1_{i}'].fillna(pd.NA).astype('Int64')
		output_table[f'y2_{i}'] = output_table[f'y2_{i}'].fillna(pd.NA).astype('Int64')

	output_table.drop(columns=['img'], inplace=True)

	column_order = [
	'basename', 'maxrow', 'maxcol', 'row', 'col', 'pos',
	'scorer1', 'score1', 'x1_1', 'x2_1', 'y1_1', 'y2_1',
	'scorer2', 'score2', 'x1_2', 'x2_2', 'y1_2', 'y2_2',
	'scorer3', 'score3', 'x1_3', 'x2_3', 'y1_3', 'y2_3'
	]
	output_table = output_table.reindex(columns=column_order)

	output_table.columns = [col.capitalize() for col in output_table.columns]
	output_table.rename(columns={'Maxrow': 'MaxRow', 'Maxcol': 'MaxCol'}, inplace=True)
	return output_table

def clean_coordinates(row: pd.Series) -> pd.Series:
	"""
	Cleans the dataframe by:
	a) Removing annotations with coordinate side length 0 
	b) Fixing annotations where the x2 > x1 or y2 > y1 
	c) Checking that coordinates are square

	Parameters:
	-----------
	data : pd.DataFrame
		The row of the dataframe to be cleaned

	Returns:
	--------
	pd.Series
		The cleaned row with updated coordinates
	"""
	
	coord1 = [row["X1_1"], row["X2_1"], row["Y1_1"], row["Y2_1"]]
	coord2 = [row["X1_2"], row["X2_2"], row["Y1_2"], row["Y2_2"]]
	coord3 = [row["X1_3"], row["X2_3"], row["Y1_3"], row["Y2_3"]]
	coords = [coord1, coord2, coord3]        

	for index, coord_set in enumerate(coords):
		if pd.isna(coord_set).any() or coord_set[0] == coord_set[1] or coord_set[2] == coord_set[3]:
			coord_set[:] = [pd.NA] * 4
			row[f"X1_{index + 1}"], row[f"X2_{index + 1}"], row[f"Y1_{index + 1}"], row[f"Y2_{index + 1}"] = [pd.NA] * 4
			row[f"Score{index + 1}"] = pd.NA
			continue

		# b) Fixing annotations where x2 > x1 or y2 > y1
		coord_set[0], coord_set[1] = min(coord_set[0], coord_set[1]), max(coord_set[0], coord_set[1])
		coord_set[2], coord_set[3] = min(coord_set[2], coord_set[3]), max(coord_set[2], coord_set[3])
		row[f"X1_{index + 1}"], row[f"X2_{index + 1}"], row[f"Y1_{index + 1}"], row[f"Y2_{index + 1}"] = coord_set

		# c) Checking that coordinates are square
		if not abs((coord_set[1] - coord_set[0]) - (coord_set[3] - coord_set[2])) == 0:
			raise ValueError("Coord set is not square")

	return row

def clean_scores(data: pd.DataFrame) -> pd.DataFrame:
	"""
	Cleans the dataframe by removing:
	a) Rows where there are only zero or one annotations
	b) Rows where there are only two disagreeing score annotations

	Parameters:
	-----------
	data : pd.DataFrame
		The dataframe to be cleaned

	Returns:
	--------
	data : pd.DataFrame
		The cleaned dataframe
	"""
	missing_data_mask = data[["Score1", "Score2", "Score3"]].isnull().sum(axis=1) >= 2

	disagreement_mask = (
		data[["Score1", "Score2", "Score3"]].isnull().sum(axis=1) == 1
	) & (
		data[["Score1", "Score2", "Score3"]].apply(lambda row: len(set(row.dropna())) > 1, axis=1)
	)

	data = data[~(missing_data_mask | disagreement_mask)] # tilda = NOT
	data = data.reset_index(drop=True)
	return data

def add_median_column(data: pd.DataFrame) -> pd.DataFrame:
	"""
	Adds a median column to a dataframe with 3 scores

	Parameters:
	-----------
	data : pd.DataFrame
		The dataframe containing the 3 score columns (Score1, Score2, Score3) to be used for the median
	
	Returns:
	--------
	pd.DataFrame
		The same dataframe but with a Median_Score column
	"""
	data["Median_Score"] = data[["Score1", "Score2", "Score3"]].median(axis=1, skipna=True)
	return data

def _overlap(rect1: List[int], rect2: List[int]) -> bool:
	"""
	Check whether two rectangles overlap.

	Each rectangle is represented by a list of four coordinates [x1, x2, y1, y2]. (x1, y1) is the bottom left corner and (x2, y2) is the top right corner.

	Parameters:
	-----------
	rect1 : List[int]
		The first rectangle represented as [x1, x2, y1, y2]
	rect2 : List[int]
		The second rectangle represented as [x1, x2, y1, y2]
	
	Returns:
	--------
	bool
		True if rectangles overlap, else False.
	"""
	x_overlap = not (rect1[1] < rect2[0] or rect2[1] < rect1[0])
	y_overlap = not (rect1[3] < rect2[2] or rect2[3] < rect1[2])
	return x_overlap and y_overlap

def _find_overlapping_coordinates(coord_lists: List[List[int]]) -> Optional[List[List[int]]]:
	"""
	Find all sets of overlapping rectangles from a list of coordinate lists.

	Parameters:
	-----------
	coord_lists: List[int]
		A variable number of lists, each representing 4 coordinates of a rectangle [x1, x2, y1, y2]
	
	Returns:
	--------
	Optional[List[List[int]]]
		A list of overlapping rectangles. If no overlaps are found, returns None.
	"""
	overlapping_sets = []
	for i in range(len(coord_lists)):
		for j in range(i + 1, len(coord_lists)):
			if _overlap(coord_lists[i], coord_lists[j]):
				overlapping_sets.append(coord_lists[i])
				overlapping_sets.append(coord_lists[j])

	overlapping_sets = [list(coords) for coords in set(tuple(coords) for coords in overlapping_sets)]

	if overlapping_sets:
		return overlapping_sets
	else:
		return None

def _find_closest_coordinates(coord_lists: List[List[float]]) -> Optional[List[float]]:
	"""
	Find the closest rectangle coordinates to the center of the overlapping area.

	Parameters:
	-----------
	coord_lists : List[List[float]]
		A list of rectangles, where each rectangle is represented as [x1, x2, y1, y2].

	Returns:
	--------
	Optional[List[float]]
		The rectangle coordinates closest to the center of the overlapping area, or None if no overlapping area is found.
	"""
	
	overlapping_area = None
	overlapping_center = None

	coord_lists = _find_overlapping_coordinates(coord_lists)

	if coord_lists == None:
		return None

	for coords in coord_lists:
		if overlapping_area is None:
			overlapping_area = coords
		else:
			x1 = max(overlapping_area[0], coords[0])
			x2 = min(overlapping_area[1], coords[1])
			y1 = max(overlapping_area[2], coords[2])
			y2 = min(overlapping_area[3], coords[3])

			if x1 < x2 and y1 < y2:
				overlapping_area = [x1, x2, y1, y2]

	overlapping_center = [(overlapping_area[0] + overlapping_area[1]) / 2, (overlapping_area[2] + overlapping_area[3]) / 2]

	min_distance = float('inf')
	closest_coords = None
	largest_area = -1

	for coords in coord_lists:
		center = [(coords[0] + coords[1]) / 2, (coords[2] + coords[3]) / 2]
		distance = np.linalg.norm(np.array(center) - np.array(overlapping_center))
		
		area = (coords[1] - coords[0]) * (coords[3] - coords[2])
		
		if distance < min_distance:
			min_distance = distance
			closest_coords = coords
			largest_area = area
		elif distance == min_distance:
			if area > largest_area:
				closest_coords = coords
				largest_area = area

	return closest_coords

def get_central_scorer(row: pd.Series) -> Optional[int]:
	"""
	Find the index of the coordinate set closest to the centre of the overlapping area between all valid coordinate sets for that CDA.

	Parameters:
	-----------
	row : pd.Series
		A pandas Series representing a row of data with rectangle coordinates.
		The expected indices are "X1_1", "X2_1", "Y1_1", "Y2_1", "X1_2", "X2_2", 
		"Y1_2", "Y2_2", "X1_3", "X2_3", "Y1_3", "Y2_3".

	Returns:
	--------
	Optional[int]
		The index of the rectangle (1, 2, or 3) that is closest to the center of the overlapping area. Returns None if no overlapping rectangles are found.
	"""
	coord1 = [row["X1_1"], row["X2_1"], row["Y1_1"], row["Y2_1"]]
	coord2 = [row["X1_2"], row["X2_2"], row["Y1_2"], row["Y2_2"]]
	coord3 = [row["X1_3"], row["X2_3"], row["Y1_3"], row["Y2_3"]]
	coords = [coord1, coord2, coord3]
	coords = [coord for coord in coords if all(pd.notna(c) for c in coord)]

	closest_coords = _find_closest_coordinates(coords)
	if closest_coords is None:
		return None

	# Define coordinate lists and their corresponding indices
	coord_lists = [coord1, coord2, coord3]
	indices = [1, 2, 3]

	for coord, index in zip(coord_lists, indices):
		if not any(pd.isna(c) for c in coord) and closest_coords == coord:
			return index

def save_cropped_images(data: pd.DataFrame, out_folder: Path) -> None:
	"""
	Saves cropped images from the input DataFrame based on given coordinates and scorer information.

	Parameters:
	-----------
	data : pd.DataFrame
		DataFrame containing image information such as 'Centre_Coords', 'Basename', scorer columns, and crop coordinates.
	out_folder: Path
		Path to folder where cropped images will be saved
	
	Returns:
	--------
	None
		This function does not return anything but saves cropped images to the specified output directory.
	"""
	previous_file = None
	raw_img = None
	for _, row in data.iterrows():
		centre_coords = int(row['Centre_Coords'])
		coords = [row[f"X1_{centre_coords}"], row[f"X2_{centre_coords}"], row[f"Y1_{centre_coords}"], row[f"Y2_{centre_coords}"]]
		coords = [int(c) for c in coords]

		current_file = row["Basename"]

		if not current_file == previous_file:
			paths = [
				os.path.join("/Volumes/shared/Research-Groups/Mark-Banfield/Josh_Williams/Scoring_20Oct", row[f"Scorer{centre_coords}"], row["Basename"]),
				os.path.join("/Volumes/shared/Research-Groups/Mark-Banfield/Josh_Williams/Scoring_5May", row[f"Scorer{centre_coords}"], row["Basename"])
			]

			raw_img = None

			for path in paths:
				if os.path.exists(path):
					raw_img = cv.imread(path)
					break

			if raw_img is None:
				print(f"Image not found for {current_file}")
				continue
		
		cropped_img = raw_img[coords[2]:coords[3], coords[0]:coords[1]]

		filename = f"{os.path.splitext(row.Basename)[0]}_{row.Row}_{row.Col}_{row.Pos}.tif"
		out_path = os.path.join(out_folder, str(int(row['Median_Score'])), filename)
		cv.imwrite(out_path, cropped_img)

		print(f"Processed image {out_path}")
		previous_file = current_file