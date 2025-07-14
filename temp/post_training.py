import os
import pandas as pd

def find_and_concatenate_results():
    dataframes = []

    for root, dirs, files in os.walk("."):
        for file in files:
            if file.startswith("results_") and file.endswith(".csv"):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                dataframes.append(df)


    all_results = pd.concat(dataframes, ignore_index=True)
    sorted_results = all_results.sort_values(by='best_vaf', ascending=False)
    output_file = "combined_sorted_results.csv"
    sorted_results.to_csv(output_file, index=False)