import pandas as pd
import os
import random
import subprocess
from scipy import stats
import numpy as np

"""
This script processes prediction data, generates object pair TSVs, runs sts.py on them,
and then calculates a p-value based on the similarity scores from sts.py's output files.
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

# --- Function to run sts.py ---
def run_sts_script(input_file_path):
    """
    Executes the sts.py script with the specified input file and fixed parameters.
    Returns the expected output file path from sts.py based on its naming convention,
    or None if the execution fails.
    """
    embed_model_path = "../sentence_embedding/fstx_300d_indexed/"
    output_format = "wisse"
    idf_model = "none" # This was 'none' but your example output filename shows 'tfidf_none_idf'.
                       # Assuming 'none' as specified in your command.
    sts_script_path = "../sentence_embedding/sts.py"

    command = [
        "python3",
        sts_script_path,
        "--input", input_file_path,
        "--embedmodel", embed_model_path,
        "--format", output_format,
        "--idfmodel", idf_model
    ]

    print(f"Executing STS command: {' '.join(command)}")
    try:
        # Use subprocess.run to wait for the command to complete.
        # check=True will raise an exception if the command returns a non-zero exit code.
        subprocess.run(command, capture_output=True, text=True, check=True)
        # Note: We are no longer printing STS stdout/stderr here as per your request
        # to rely on filename only. If debugging, uncomment print(result.stdout/stderr).

        # Construct the expected output file path based on sts.py's naming convention
        # Example: input.tsv -> input.tsv.output_fstx_300d_indexed_sum_local_local (if format is wisse, idfmodel is none)
        # If your sts.py outputs _tfidf_none_idf, the internal logic of sts.py determines that suffix.
        # For 'wisse' format and 'none' idfmodel, it commonly appends '_sum_local_local' for sum embeddings.
        
        # Let's use a more direct convention based on your example and previous knowledge:
        embed_model_base_name = os.path.basename(os.path.normpath(embed_model_path))
        
        # This suffix can vary depending on internal STS implementation (e.g., if it uses TF-IDF or other aggregations)
        # Based on your prompt example "output_fstx_300d_indexed_sum_tfidf_none_idf"
        # and previous "output_fstx_300d_indexed_sum_local_local"
        # Let's construct it more flexibly.
        # Assuming the suffix is fixed to _sum_local_local as per previous iterations for 'wisse' and 'none'
        # If your actual sts.py output is different, this part might need a minor adjustment.
        expected_output_filename = f"{input_file_path}.output_{embed_model_base_name}_sum_local_local"
        
        # If the sts.py output filename strictly follows your example "output_fstx_300d_indexed_sum_tfidf_none_idf",
        # you would need to know the 'sum_tfidf_none_idf' part.
        # For consistency with the provided command: --embedmodel fstx_300d_indexed/ --format wisse --idfmodel none
        # The 'wisse' format with 'none' idfmodel usually leads to '_sum_local_local'.
        # If the output really is '_sum_tfidf_none_idf', your sts.py might be doing something different internally
        # or the 'idfmodel none' argument doesn't result in 'none' in the filename.
        
        # For safety and given the prompt "object_pairs_predictions_random.tsv.output_fstx_300d_indexed_sum_tfidf_none_idf"
        # Let's prioritize the most specific example you gave in this prompt,
        # acknowledging it might conflict with the `idfmodel none` setting.
        # If `sts.py` *always* appends '_sum_tfidf_none_idf' when `idfmodel` is `none`, then this is fine.
        # Otherwise, the `_sum_local_local` based on 'wisse' and 'none' idfmodel is more typical.
        
        # Let's stick to your example's exact suffix for `output_fstx_300d_indexed_sum_tfidf_none_idf`
        # for `fstx_300d_indexed` and `idfmodel none`:
        suffix = f"output_{embed_model_base_name}_sum_tfidf_none_idf"
        expected_output_filename = f"{input_file_path}.{suffix}"


        if os.path.exists(expected_output_filename):
            print(f"STS output file expected and found: {expected_output_filename}")
            return expected_output_filename
        else:
            print(f"Error: Expected STS output file '{expected_output_filename}' not found after execution for {input_file_path}.")
            return None

    except FileNotFoundError:
        print(f"Error: 'python3' or '{sts_script_path}' not found. Make sure they are in your PATH or provide full paths.")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error executing STS command for {input_file_path}: {e}")
        print(f"Command output (stdout):\n{e.stdout}") # Printing stdout/stderr for debugging command issues
        print(f"Command error (stderr):\n{e.stderr}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while running STS for {input_file_path}: {e}")
        return None

# --- Main script execution ---
# Adjust this to 'results' if that's your main directory
#base_directories = find_subdirectories_one_level('results_test')
base_directories  = [
    'results/ncd-gp-conceptnet-transformer_epochs-40_stackSize-1_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8',
    'results/ncd-conceptnet-transformer_epochs-40_stackSize-1_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8',
    'results/ncd-gp-transformer_epochs-40_stackSize-1_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8',
    'results/ncd-transformer_epochs-40_stackSize-1_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8',
    'results/ncd-gp-conceptnet-transformer_epochs-100_stackSize-2_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8',
    'results/ncd-conceptnet-transformer_epochs-40_stackSize-2_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8',
    'results/ncd-gp-transformer_epochs-100_stackSize-2_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8',
    'results/ncd-transformer_epochs-100_stackSize-2_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8'
]

# Prepare for collecting results to print in a table
results_for_table = []

# --- Process each subdirectory ---
for csv_item in base_directories: # Looping through each full path
    # Define paths for input and output files
    # Changed 'val_predictions.csv' to 'predictions.csv' if that's the common name
    csv_file_path = os.path.join(csv_item, 'val_predictions.csv') # Assuming val_predictions.csv
    object_pairs_tsv_file_path = os.path.join(csv_item, 'object_pairs_predictions.tsv')
    random_object_pairs_tsv_file_path = os.path.join(csv_item, 'object_pairs_predictions_random.tsv')

    # Check if the input CSV file exists
    if not os.path.exists(csv_file_path):
        print(f"Warning: '{csv_file_path}' not found. Skipping processing for this directory.")
        continue # Move to the next directory if the file is missing
    else:
        print(f"Info: '{csv_file_path}' found. Starting processing for this directory.")

    # Variables to store STS output file paths
    sts_output_predictions = None
    sts_output_predictions_random = None

    try:
        # Read the CSV file once
        inferences_item = pd.read_csv(csv_file_path,
                                      header=0,
                                      index_col=0)
        print(f"Info: '{csv_file_path}' loaded.")
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

        # --- Execute sts.py for the newly generated TSV files ---
        print("\n--- Running STS scripts ---")
        sts_output_predictions = run_sts_script(object_pairs_tsv_file_path)
        sts_output_predictions_random = run_sts_script(random_object_pairs_tsv_file_path)
        print("--- STS scripts finished ---\n")

        # --- Calculate p-value if STS outputs are available ---
        if sts_output_predictions and os.path.exists(sts_output_predictions) and \
           sts_output_predictions_random and os.path.exists(sts_output_predictions_random):

            try:
                scores_predictions = pd.read_csv(sts_output_predictions, header=None, index_col=None, names=['scores'])
                scores_predictions_random = pd.read_csv(sts_output_predictions_random, header=None, index_col=None, names=['scores_random'])

                # Drop NaNs to ensure t-test works correctly
                scores_predictions_cleaned = scores_predictions.dropna()['scores']
                scores_predictions_random_cleaned = scores_predictions_random.dropna()['scores_random']

                if not scores_predictions_cleaned.empty and not scores_predictions_random_cleaned.empty:
                    # Perform independent t-test
                    statistic, pvalue = stats.ttest_ind(
                        scores_predictions_cleaned,
                        scores_predictions_random_cleaned
                    )

                    mean_predictions = np.mean(scores_predictions_cleaned)
                    mean_predictions_random = np.mean(scores_predictions_random_cleaned)

                    # Extract the directory name for the table output
                    dir_name = os.path.basename(csv_item)

                    results_for_table.append({
                        'directory': dir_name,
                        'mean_predictions': mean_predictions,
                        'mean_random': mean_predictions_random,
                        'pvalue': pvalue
                    })
                    print(f"P-value for {dir_name}: {pvalue:.2e}")
                else:
                    print(f"Warning: No valid scores found after cleaning for {csv_item}. Skipping p-value calculation.")
            except Exception as e:
                print(f"Error reading STS output files or calculating p-value for {csv_item}: {e}")
        else:
            print(f"Skipping p-value calculation for {csv_item} due to missing or failed STS output files.")

    except KeyError as ke:
        print(f"Error: Missing expected column in '{csv_file_path}': {ke}. Skipping this directory.")
    except Exception as e:
        print(f"An unexpected error occurred while processing '{csv_file_path}': {e}. Skipping this directory.")

# --- Print results table ---
print("\n" + "="*80)
print("                       STS Similarity and P-value Results")
print("="*80)
print("\\hline")
print("Directory & Mean Predictions / Random Mean & P-value \\\\")
print("\\hline")
for res in results_for_table:
    line = ''.join((
        res['directory'], ' & ',
        f"{res['mean_predictions']:.3f}", ' / ',
        f"{res['mean_random']:.3f}", ' & ',
        f"{res['pvalue']:.2e}", ' \\\\ '
    ))
    print(line)
print("\\hline")
print("="*80)