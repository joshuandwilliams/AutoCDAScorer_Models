from typing import List, Dict
import numpy as np
from sklearn.utils import shuffle
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.patches import Rectangle
from itertools import combinations

def generate_scorer_key(data: pd.DataFrame, random_state: int = 42) -> Dict[int, str]:
	"""
	Generates an anonymization key by shuffling the unique scorers from the 'Scorer1' column in the given dataset
	and assigning each a unique integer ID.

	Parameters:
	-----------
	data : pd.DataFrame
		A DataFrame containing a column 'Scorer1' with scorer names.
	random_state : int, optional (default=42)
		Random seed for shuffling to ensure reproducibility.

	Returns:
	--------
	Dict[int, str]
		A dictionary mapping anonymous integer IDs (1-based) to scorer names.
		The key is also saved to 'anonymisation_key.csv'.
	"""
	if 'Scorer1' not in data.columns or data['Scorer1'].isnull().all():
		raise ValueError("The 'Scorer1' column is missing or empty in the provided data.")

	scorer_list = data['Scorer1'].dropna().unique()
	shuffled_scorers = shuffle(scorer_list, random_state=random_state)
	key = {idx + 1: scorer for idx, scorer in enumerate(shuffled_scorers)}

	pd.DataFrame(list(key.items()), columns=['ID', 'Scorer_Name']).to_csv('anonymisation_key.csv', index=False)

	return key

def anonymise_scorers(all_scores: pd.DataFrame, key: Dict[int, str]) -> pd.DataFrame:
	"""
	Anonymizes scorer names in specified columns using a provided key.

	Parameters:
	-----------
	all_scores : pd.DataFrame
		A DataFrame containing scorer columns to be anonymized.
	key : Dict[int, str]
		A dictionary mapping integer IDs to scorer names.

	Returns:
	--------
	pd.DataFrame
		The DataFrame with anonymized scorer names.
	"""
	if not key:
		name_to_index = {}
	else:
		name_to_index = {name: idx for idx, name in key.items()}
	
	scorer_columns = ['Scorer1', 'Scorer2', 'Scorer3']
	
	for col in scorer_columns:
		all_scores[col] = all_scores[col].map(name_to_index).astype('Int64')
	
	return all_scores

def downsample_score_classes_csv(data: pd.DataFrame, seed: int = None, print_results: bool = False) -> pd.DataFrame:
	"""
	Standardizes class sizes by down-sampling the rows in `data` for each score class 
	(based on the 'Score' column) to match the number of rows in the smallest class.

	Parameters:
	----------
	data : pd.DataFrame
		Input DataFrame with columns 'Score', 'Scorer', and 'Median_Score'.
	seed : int, optional
		Seed for the random number generator to ensure reproducibility.
		Defaults to None, which results in a different sample each time.
	print_results: bool
		Boolean telling the function whether to print a counter of the downsampled score class sizes or not.

	Returns:
	-------
	pd.DataFrame
		A new DataFrame where each class (score) has the same number of rows as the smallest class.
	"""
	grouped = data.groupby('Median_Score')
	min_count = grouped.size().min()
	standardized_df = (
		grouped.apply(lambda x: x.sample(min_count, random_state=seed), include_groups=False)
		.reset_index()
	)
	
	if print_results:
		print(Counter(standardized_df['Median_Score']))

	return standardized_df

def generate_scorer_cms(data: pd.DataFrame, column: str = 'Median_Score') -> Dict[str, np.ndarray]:
	"""
	Generates confusion matrices for each scorer

	Parameters:
	-----------
	data : pd.DataFrame
		The input DataFrame containing columns 'Median_Score', 'Scorer', and 'Score'
	
	Returns:
	--------
	Dict[str, np.ndarray]
		A dictionary where the keys are the unique 'Scorer' values and the values are the corresponding confusion matrices comparing 'Median_Score' and 'Score'.
	"""
	confusion_matrices = {}
	data = data.dropna(subset=['Score'])
	unique_scorers = data['Scorer'].unique()
	for scorer in unique_scorers:
		scorer_data = data[data['Scorer'] == scorer]
		cm = confusion_matrix(scorer_data[column], scorer_data['Score'])
		confusion_matrices[scorer] = cm

	confusion_matrices = dict(sorted(confusion_matrices.items()))

	return confusion_matrices

def cm_accuracies(cm: np.ndarray, class_labels: List[str], print_results: bool = False) -> Dict[str, any]:
	"""
	Computes multiple accuracy metrics from the confusion matrix, including exact accuracy,
	within-one accuracy, and per-class accuracies.

	Parameters:
	----------
	cm : np.ndarray
		The confusion matrix.
	class_labels : List[str]
		The list of class labels.

	Returns:
	-------
	Dict[str, any]
		A dictionary containing exact accuracy, within-one accuracy, and per-class accuracies.
	"""
	if cm.size == 0:
		return {"exact_accuracy": 0, "within_one_accuracy": 0, "class_accuracies": {}}

	exact_accuracy = np.sum(np.diag(cm)) / np.sum(cm)
	
	correct_within_one = (np.sum(np.diag(cm)) 
						  + np.sum(np.diag(cm, k=1)) 
						  + np.sum(np.diag(cm, k=-1)))
	within_one_accuracy = correct_within_one / np.sum(cm)
	
	class_accuracies = {
		class_labels[i]: cm[i, i] / sum(cm[i, :]) if sum(cm[i, :]) > 0 else 0
		for i in range(cm.shape[0])
	}

	if print_results:
		print(f"Accuracy: {exact_accuracy}, Near-Miss: {within_one_accuracy}")
		print(f"Per-Class Accuracies: {dict(sorted(class_accuracies.items()))}")
	
	return {
		"exact_accuracy": exact_accuracy,
		"within_one_accuracy": within_one_accuracy,
		"class_accuracies": dict(sorted(class_accuracies.items()))
	}

def plot_confusion_matrix(conf_matrix: np.ndarray, labels: np.ndarray, axis_labels: tuple[str, str], output: str = "confusion_matrix.png") -> None:
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
	plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Greens)
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
			plt.text(j, i, str(conf_matrix[i][j]),
					 horizontalalignment='center',
					 color='white' if conf_matrix[i, j] > np.max(conf_matrix) / 2 else 'black')
			
			# Add a black border if it's on the diagonal (i == j)
			if i == j:
				ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='black', linewidth=1))

	plt.ylabel(axis_labels[0])
	plt.xlabel(axis_labels[1])
	plt.tight_layout()

	plt.savefig(output)
	plt.show()

def find_pairwise_agreement(row: pd.Series, precision: int = 0) -> pd.Series:
	"""
	Calculates the pairwise agreement percentage for Score1, Score2, and Score3 in a row.
	
	Parameters:
	----------
	row : pd.Series
		A row containing Score1, Score2, and Score3 values.
	
	precision : int, optional
		The tolerance level for agreement. Default is 0 for exact agreement, 
		higher values allow for agreement within the specified difference.
	
	Returns:
	-------
	float or pd.NA
		The pairwise agreement percentage for the row, or pd.NA if all scores are missing or only one score is non-missing.
	"""
	scores = row[['Score1', 'Score2', 'Score3']].dropna()
	if len(scores) < 2:
		return pd.NA

	score_pairs = [
		(scores.iloc[0], scores.iloc[1]),
		(scores.iloc[0], scores.iloc[2]) if len(scores) == 3 else None,
		(scores.iloc[1], scores.iloc[2]) if len(scores) == 3 else None
	]
	valid_pairs = [pair for pair in score_pairs if pair is not None]
	agreements = sum(abs(x - y) <= precision for x, y in valid_pairs)
	
	return (agreements / len(valid_pairs)) if valid_pairs else pd.NA

def extract_score_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts all valid score pairs (ScoreA, ScoreB) from each row in the DataFrame
    where both scores are non-missing.

    Parameters:
    ----------
    df : pd.DataFrame
        A DataFrame with columns Score1, Score2, and Score3.

    Returns:
    -------
    pd.DataFrame
        A DataFrame with columns ScoreA and ScoreB containing all valid score pairs.
    """
    pairs = []

    for _, row in df.iterrows():
        scores = row[['Score1', 'Score2', 'Score3']].dropna().tolist()
        row_pairs = list(combinations(scores, 2))
        pairs.extend(row_pairs)

    return pd.DataFrame(pairs, columns=["ScoreA", "ScoreB"])