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
    idf_model = "none" 
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
        
        # Let's use a more direct convention based on your example and previous knowledge:
        embed_model_base_name = os.path.basename(os.path.normpath(embed_model_path))
                
        # For safety and given the prompt "object_pairs_predictions_random.tsv.output_fstx_300d_indexed_sum_tfidf_none_idf"
        # Let's prioritize the most specific example you gave in this prompt,
        
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

#docker exec interesting_kapitsa sh -c "cd /workspace/open-ncd-kbc/results/open-ncd-kbc/ushzkb2o/ && find . -name '*.tsv' -print0 | tar -czf /tmp/ushzkb2o_tsv_files.tar.gz --null -T -"
#docker cp interesting_kapitsa:/tmp/ushzkb2o_tsv_files.tar.gz ./

#docker exec interesting_kapitsa sh -c "cd /workspace/open-ncd-kbc/results/baseline_att_lstm/vi0mllw3/ && find . -name '*.tsv' -print0 | tar -czf /tmp/vi0mllw3_tsv_files.tar.gz --null -T -"

#docker exec nostalgic_robinson sh -c "cd /workspace/open-ncd-kbc/results/open-ncd-kbc/r2rtc3jo/ && find . -name '*.tsv' -print0 | tar -czf /tmp/r2rtc3jo_tsv_files.tar.gz --null -T -"

#docker exec nostalgic_robinson sh -c "cd /workspace/open-ncd-kbc/results/baseline_att_lstm/brn82uha/ && find . -name '*.tsv' -print0 | tar -czf /tmp/brn82uha_tsv_files.tar.gz --null -T -"
#docker cp nostalgic_robinson:/tmp/r2rtc3jo_tsv_files.tar.gz ./
#docker cp nostalgic_robinson:/tmp/brn82uha_tsv_files.tar.gz ./

# base_directories = find_subdirectories_one_level('results/baseline_att_lstm/vi0mllw3') #
# base_directories = find_subdirectories_one_level('results/open-ncd-kbc')
# base_directories = find_subdirectories_one_level('results/baseline_att_lstm/brn82uha') # blue demon
base_directories = find_subdirectories_one_level('results/open-ncd-kbc/r2rtc3jo') # blue demon
results_for_table_file_name = os.path.join(base_directories, 'mean_sts_results_blue_demon.txt')
# testing directories to validate with a small set of results
# base_directories  = [
#     'results/ncd-gp-conceptnet-transformer_epochs-40_stackSize-1_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8',
#     'results/ncd-conceptnet-transformer_epochs-40_stackSize-1_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8',
#     'results/ncd-gp-transformer_epochs-40_stackSize-1_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8',
#     'results/ncd-transformer_epochs-40_stackSize-1_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8',
#     'results/ncd-gp-conceptnet-transformer_epochs-100_stackSize-2_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8',
#     'results/ncd-conceptnet-transformer_epochs-40_stackSize-2_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8',
#     'results/ncd-gp-transformer_epochs-100_stackSize-2_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8',
#     'results/ncd-transformer_epochs-100_stackSize-2_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8'
# ]

# Prepare for collecting results to print in a table
results_for_table = []

# --- Process each subdirectory ---
for csv_item in base_directories: # Looping through each full path
    # Define paths for input and output files
    # Changed 'val_predictions.csv' to 'predictions.csv' if that's the common name
    val_csv_file_path = os.path.join(csv_item, 'predictions.tsv') # Assuming val_predictions.csv # TODO: CHANGE TO PROPER FILE!!!
    val_object_pairs_tsv_file_path = os.path.join(csv_item, 'val_object_pairs_predictions.tsv')
    val_random_object_pairs_tsv_file_path = os.path.join(csv_item, 'val_object_pairs_predictions_random.tsv')

    # Also do the same for the test predictons:
    test_csv_file_path = os.path.join(csv_item, 'test_predictions.tsv') # Assuming val_predictions.csv # TODO: CHANGE TO PROPER FILE!!!
    test_object_pairs_tsv_file_path = os.path.join(csv_item, 'test_object_pairs_predictions.tsv')
    test_random_object_pairs_tsv_file_path = os.path.join(csv_item, 'test_object_pairs_predictions_random.tsv')

    # Check if the input CSV file exists
    if not os.path.exists(val_csv_file_path) or not os.path.exists(test_csv_file_path):
        print(f"Warning: '{val_csv_file_path}' or '{test_csv_file_path}' not found. Skipping processing for this directory.")
        continue # Move to the next directory if the file is missing
    else:
        print(f"Info: '{val_csv_file_path}' and '{test_csv_file_path}' found. Starting processing for this directory.")

    # Variables to store STS output file paths for validation predictions
    sts_output_val_predictions = None
    sts_output_val_predictions_random = None

    # Variables to store STS output file paths for validation predictions
    sts_output_test_predictions = None
    sts_output_test_predictions_random = None

    try:
        # Read the val CSV file once
        # val_inferences_item = pd.read_csv(val_csv_file_path,
        #                               header=0,
        #                               index_col=0)
        
        # # Read the test CSV file once
        # test_inferences_item = pd.read_csv(test_csv_file_path,
        #                               header=0,
        #                               index_col=0)
        
        # Read the val TSV file once
        val_inferences_item = pd.read_csv(val_csv_file_path,
                                        sep='\t',       # Specify tab as the separator
                                        header=0,       # Your TSV has a header row
                                        )
        # Read the test TSV file
        test_inferences_item = pd.read_csv(test_csv_file_path,
                                        sep='\t',       # Specify tab as the separator
                                        header=0,       # Your TSV has a header row
                                        )
            
        print(f"Info: '{val_csv_file_path}' and '{test_csv_file_path}' loaded.")
        
        val_object_pairs = []
        val_obj_list = []
        val_obj_true_list = []

        test_object_pairs = []
        test_obj_list = []
        test_obj_true_list = []

        # Iterate through rows to extract and clean data for both tasks
        for idx, inference in val_inferences_item.iterrows():
            inference_obj = str(inference['Obj']).replace('[start] ','').replace(' [end]','')
            inference_obj_true = str(inference['Obj_true']).replace('[start] ','').replace(' [end]','')

            if len(inference_obj) > 0 and len(inference_obj_true) > 0:
                # For the first TSV (original val pairs)
                val_object_pairs.append((inference_obj, inference_obj_true))
                # For the second TSV (shuffled true val objects)
                val_obj_list.append(inference_obj)
                val_obj_true_list.append(inference_obj_true)
        
        # Iterate through rows to extract and clean data for both tasks
        for idx, inference in test_inferences_item.iterrows():
            inference_obj = str(inference['Obj']).replace('[start] ','').replace(' [end]','')
            inference_obj_true = str(inference['Obj_true']).replace('[start] ','').replace(' [end]','')

            if len(inference_obj) > 0 and len(inference_obj_true) > 0:
                # For the first TSV (original test pairs)
                test_object_pairs.append((inference_obj, inference_obj_true))
                # For the second TSV (shuffled true test objects)
                test_obj_list.append(inference_obj)
                test_obj_true_list.append(inference_obj_true)

        # --- Save the first val TSV file (original object pairs) ---
        print(f"Saving object pairs to: {val_object_pairs_tsv_file_path}")
        pd.DataFrame(val_object_pairs).to_csv(val_object_pairs_tsv_file_path, sep='\t', header=None, index=None)

        # --- Save the second val TSV file (randomized object pairs) ---
        random.shuffle(val_obj_true_list) # Shuffle the true objects list
        print(f"Saving random object pairs to: {val_random_object_pairs_tsv_file_path}")
        pd.DataFrame(zip(val_obj_list, val_obj_true_list)).to_csv(val_random_object_pairs_tsv_file_path, sep='\t', header=None, index=None)

        # --- Save the first test TSV file (original object pairs) ---
        print(f"Saving object pairs to: {test_object_pairs_tsv_file_path}")
        pd.DataFrame(test_object_pairs).to_csv(test_object_pairs_tsv_file_path, sep='\t', header=None, index=None)

        # --- Save the second test TSV file (randomized object pairs) ---
        random.shuffle(test_obj_true_list) # Shuffle the true objects list
        print(f"Saving random object pairs to: {test_random_object_pairs_tsv_file_path}")
        pd.DataFrame(zip(test_obj_list, test_obj_true_list)).to_csv(test_random_object_pairs_tsv_file_path, sep='\t', header=None, index=None)

        # --- Execute sts.py for the newly generated val TSV files ---
        print("\n--- Running STS scripts for val set ---")
        sts_output_val_predictions = run_sts_script(val_object_pairs_tsv_file_path)
        sts_output_val_predictions_random = run_sts_script(val_random_object_pairs_tsv_file_path)

        # --- Execute sts.py for the newly generated test TSV files ---
        print("\n--- Running STS scripts for test set ---")
        sts_output_test_predictions = run_sts_script(test_object_pairs_tsv_file_path)
        sts_output_test_predictions_random = run_sts_script(test_random_object_pairs_tsv_file_path)
       

        print("--- STS scripts finished ---\n")

        # --- Calculate p-value if STS outputs are available ---
        if sts_output_val_predictions and os.path.exists(sts_output_val_predictions) and \
           sts_output_val_predictions_random and os.path.exists(sts_output_val_predictions_random) and \
            sts_output_test_predictions and os.path.exists(sts_output_test_predictions) and \
            sts_output_test_predictions_random and os.path.exists(sts_output_test_predictions_random):

            try:
                scores_val_predictions = pd.read_csv(sts_output_val_predictions, header=None, index_col=None, names=['scores'])
                scores_val_predictions_random = pd.read_csv(sts_output_val_predictions_random, header=None, index_col=None, names=['scores_random'])

                scores_test_predictions = pd.read_csv(sts_output_test_predictions, header=None, index_col=None, names=['scores'])
                scores_test_predictions_random = pd.read_csv(sts_output_test_predictions_random, header=None, index_col=None, names=['scores_random'])

                # Drop NaNs to ensure t-test works correctly
                scores_val_predictions_cleaned = scores_val_predictions.dropna()['scores']
                scores_val_predictions_random_cleaned = scores_val_predictions_random.dropna()['scores_random']
                
                scores_test_predictions_cleaned = scores_test_predictions.dropna()['scores']
                scores_test_predictions_random_cleaned = scores_test_predictions_random.dropna()['scores_random']

                if not scores_val_predictions_cleaned.empty and not scores_val_predictions_random_cleaned.empty and \
                    not scores_test_predictions_cleaned.empty and not scores_test_predictions_random_cleaned.empty:
                    # Perform independent t-test
                    val_statistic, val_pvalue = stats.ttest_ind(
                        scores_val_predictions_cleaned,
                        scores_val_predictions_random_cleaned
                    )

                    test_statistic, test_pvalue = stats.ttest_ind(
                        scores_test_predictions_cleaned,
                        scores_test_predictions_random_cleaned
                    )

                    mean_val_predictions = np.mean(scores_val_predictions_cleaned)
                    mean_val_predictions_random = np.mean(scores_val_predictions_random_cleaned)

                    mean_test_predictions = np.mean(scores_test_predictions_cleaned)
                    mean_test_predictions_random = np.mean(scores_test_predictions_random_cleaned)

                    # Calculate percentage differences
                    # Test predictions % difference
                    # Handle potential division by zero for random means, though unlikely with these values
                    if mean_test_predictions_random != 0:
                        test_percent_difference = ((mean_test_predictions - mean_test_predictions_random) / mean_test_predictions_random) * 100
                    else:
                        test_percent_difference = float('inf') # Or handle as appropriate, e.g., 0 or NaN

                    # Validation predictions % difference
                    if mean_val_predictions_random != 0:
                        val_percent_difference = ((mean_val_predictions - mean_val_predictions_random) / mean_val_predictions_random) * 100
                    else:
                        val_percent_difference = float('inf') # Or handle as appropriate

                    # Extract the directory name for the table output
                    dir_name = os.path.basename(csv_item)
                    current_result = {
                        'directory': dir_name,
                        'mean_test_predictions': mean_test_predictions,
                        'mean_test_random': mean_test_predictions_random,
                        'test_percent_difference': test_percent_difference, # Added
                        'test_pvalue': test_pvalue,
                        'mean_val_predictions': mean_val_predictions,
                        'mean_val_random': mean_val_predictions_random,
                        'val_percent_difference': val_percent_difference,   # Added
                        'val_pvalue': val_pvalue
                    }

                    results_for_table.append(current_result)

                    print(f"Directory: {dir_name}, "
                        f"Test Pred Mean: {mean_test_predictions:.3f}, "
                        f"Test Random Mean: {mean_test_predictions_random:.3f}, "
                        f"Test % Diff: {test_percent_difference:.2f}%, " # Added test % difference
                        f"Test P-value: {test_pvalue:.2e}, "
                        f"Val Pred Mean: {mean_val_predictions:.3f}, "
                        f"Val Random Mean: {mean_val_predictions_random:.3f}, "
                        f"Val % Diff: {val_percent_difference:.2f}%, " # Added validation % difference
                        f"Val P-value: {val_pvalue:.2e}"
                    )
                    # --- Save ONLY THE CURRENT result to CSV ---
                    # Create a DataFrame with just the current result
                    current_df = pd.DataFrame([current_result])
                    output_csv_path = os.path.join(csv_item, 'sts_pvalues_results.csv') #
                    # Save the current DataFrame to CSV
                    current_df.to_csv(output_csv_path, index=False)
                    print(f"Current result saved to: {output_csv_path}")
                else:
                    print(f"Warning: No valid scores found after cleaning for {csv_item}. Skipping p-value calculation.")

                print(f"\nResults successfully saved to {output_csv_path}")
            except Exception as e:
                print(f"Error reading STS output files or calculating p-value for {csv_item}: {e}")
        else:
            print(f"Skipping p-value calculation for {csv_item} due to missing or failed STS output files.")

    except KeyError as ke:
        print(f"Error: Missing expected column in '{val_csv_file_path}': {ke}. Skipping this directory.")
    except Exception as e:
        print(f"An unexpected error occurred while processing '{val_csv_file_path}': {e}. Skipping this directory.")

# --- Print results table ---
print("\n" + "="*80)
print("                       STS Similarity and P-value Results")
print("="*80)
print("\\hline")
# Updated header to exactly match your request:
print("Directory & Test $\\mu_{sts}$ (pred/rdn) & Gap \\% & Test p & Val. $\\mu_{sts}$ (pred/rdn) & Gap \\% & Val. p \\\\")
print("\\hline")
for res in results_for_table:
    line = ''.join((
        res['directory'], ' & ',
        f"{res['mean_test_predictions']:.3f}", ' / ',
        f"{res['mean_test_random']:.3f}", ' & ', # Combines pred/rdn for test
        f"{res['test_percent_difference']:.2f}\\%", ' & ', # Test % Diff
        f"{res['test_pvalue']:.2e}", ' & ', # Test p-value
        f"{res['mean_val_predictions']:.3f}", ' / ',
        f"{res['mean_val_random']:.3f}", ' & ', # Combines pred/rdn for validation
        f"{res['val_percent_difference']:.2f}\\%", ' & ', # Val % Diff
        f"{res['val_pvalue']:.2e}", ' \\\\ ' # Val p-value
    ))
    print(line)
print("\\hline")
print("="*80)

def save_sts_results_to_txt(results_for_table, filename="sts_results.txt"):
    """
    Saves STS similarity and p-value results in LaTeX formatting to a text file.

    Args:
        results_for_table (list of dict): A list of dictionaries, where each dictionary
                                          contains the results for a specific directory.
                                          Expected keys: 'directory',
                                          'mean_test_predictions', 'mean_test_random',
                                          'test_percent_difference', 'test_pvalue',
                                          'mean_val_predictions', 'mean_val_random',
                                          'val_percent_difference', 'val_pvalue'.
        filename (str): The name of the file to save the results to.
    """
    with open(filename, 'w') as f:
        f.write("="*80 + "\n")
        f.write("                       STS Similarity and P-value Results\n")
        f.write("="*80 + "\n")
        f.write("\\hline\n")
        f.write("Directory & Test $\\mu_{sts}$ (pred/rdn) & Gap \\% & Test p & Val. $\\mu_{sts}$ (pred/rdn) & Gap \\% & Val. p \\\\\n")
        f.write("\\hline\n")

        for res in results_for_table:
            line = ''.join((
                res['directory'], ' & ',
                f"{res['mean_test_predictions']:.3f}", ' / ',
                f"{res['mean_test_random']:.3f}", ' & ',
                f"{res['test_percent_difference']:.2f}\\%", ' & ',
                f"{res['test_pvalue']:.2e}", ' & ',
                f"{res['mean_val_predictions']:.3f}", ' / ',
                f"{res['mean_val_random']:.3f}", ' & ',
                f"{res['val_percent_difference']:.2f}\\%", ' & ',
                f"{res['val_pvalue']:.2e}", ' \\\\ \n'  # Added \n for newline in file
            ))
            f.write(line)
        f.write("\\hline\n")
        f.write("="*80 + "\n")

save_sts_results_to_txt(results_for_table, results_for_table_file_name)