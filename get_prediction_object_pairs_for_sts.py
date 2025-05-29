import pandas as pd
import os
import random


"""
This script is used to get the prediction pairs obtained from the val set (). It will perform a clean up and verify that they are valid (based on length)
Afterwards it will generate a tsv with those object pairs. 
"""

def find_subdirectories_one_level(directory_path):
    """
    Finds the names of all subdirectories directly within the given directory.
    Goes only one level deep.

    Args:
        directory_path (str): The path to the directory to search.

    Returns:
        list: A list of strings, where each string is the path of a subdirectory.
              Returns an empty list if the directory does not exist or has no subdirectories.
    """
    subdirectories = []
    if not os.path.isdir(directory_path):
        print(f"Error: '{directory_path}' is not a valid directory.")
        return []

    try:
        for item_name in os.listdir(directory_path):
            item_path = os.path.join(directory_path, item_name)
            if os.path.isdir(item_path):
                subdirectories.append(item_path)
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

    return subdirectories

base_directories = find_subdirectories_one_level('results')

# --- Process each subdirectory once ---
for csv_item in base_directories: # Looping through each full path
    # Define paths for input and output files
    csv_file_path = os.path.join(csv_item, 'predictions.csv')
    object_pairs_tsv_file_path = os.path.join(csv_item, 'object_pairs_predictions.tsv')
    random_object_pairs_tsv_file_path = os.path.join(csv_item, 'object_pairs_predictions_random.tsv')

    # Check if the input CSV file exists
    if not os.path.exists(csv_file_path):
        print(f"Warning: '{csv_file_path}' not found. Skipping processing for this directory.")
        continue # Move to the next directory if the file is missing

    try:
        # Read the CSV file once
        inferences_item = pd.read_csv(csv_file_path,
                                      header=0,
                                      index_col=0)

        object_pairs = []
        obj_list = []
        obj_true_list = []

        # Iterate through rows to extract and clean data for both tasks
        for idx, inference in inferences_item.iterrows():
            inference_obj = str(inference['Obj']).replace('[start] ','').replace(' [end]','')
            inference_obj_true = str(inference['Obj_true']).replace('[start] ','').replace(' [end]','')

            if len(inference_obj) > 0 and len(inference_obj_true) > 0:
                # For the first TSV (original pairs)
                object_pairs.append((inference_obj, inference_obj_true))
                # For the second TSV (shuffled true objects)
                obj_list.append(inference_obj)
                obj_true_list.append(inference_obj_true)

        # --- Save the first TSV file (original object pairs) ---
        print(f"Saving object pairs to: {object_pairs_tsv_file_path}")
        pd.DataFrame(object_pairs).to_csv(object_pairs_tsv_file_path, sep='\t', header=None, index=None)

        # --- Save the second TSV file (randomized object pairs) ---
        random.shuffle(obj_true_list) # Shuffle the true objects list
        print(f"Saving random object pairs to: {random_object_pairs_tsv_file_path}")
        pd.DataFrame(zip(obj_list, obj_true_list)).to_csv(random_object_pairs_tsv_file_path, sep='\t', header=None, index=None)

    except KeyError as ke:
        print(f"Error: Missing expected column in '{csv_file_path}': {ke}. Skipping this directory.")
    except Exception as e:
        print(f"An unexpected error occurred while processing '{csv_file_path}': {e}. Skipping this directory.")
