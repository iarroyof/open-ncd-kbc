from os import name
import pandas as pd
from scipy import stats
import numpy as np

"""
This script will calculate the sts values using the word embeddings obtained using the sentence embedding repository script.
Afterwards it will format the results for latex.

SO overall the execution looks like

1. Obtain the predicctions using the val set and the specific model to get the sts values for.
2. run the get_prediction_object_pairs.py and the get_prediction_object_pairs_random.py with the correct
    csv file paths (csv_files variable) and the correct names csv names (Names used on the article).

3. Run the sentence embedding script (https://github.com/iarroyof/sentence_embedding/blob/master/sts.py) in the 
    https://github.com/iarroyof/sentence_embedding repository. This will output a file with the output_fstx_300d_indexed_sum_local_local suffix.

4. Run this script (get_sts_pvalue.py) so that it takes as input the outputs obtained from the sentence emebeddings and starts calculating the statistic_target, pvalue_target. 


Notes:

Since most of the the AIgroKB repo has been migrated to this repo (open-ncd-kbc), the paths for these set of scripts (the ones mentioned above) have changed.
Looking at the files, looks like the new correct path is now under open-ncd-kbc/results instead of open-ncd-kbc/results_final. To avoid overwrite, you can comment
out the current csv files and names and just add new ones. 

Check the val set to make sure you are performing inference on the correct set defined in the article


"""


csv_files  = [
    'results_final/ncd-gp-conceptnet-transformer_epochs-40_stackSize-1_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8/',
    'results_final/ncd-conceptnet-transformer_epochs-40_stackSize-1_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8/',
    'results_final/ncd-gp-transformer_epochs-40_stackSize-1_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8/',
    'results_final/ncd-transformer_epochs-40_stackSize-1_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8/',
    'results_final/ncd-gp-conceptnet-transformer_epochs-100_stackSize-2_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8/',
    'results_final/ncd-conceptnet-transformer_epochs-40_stackSize-2_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8/',
    'results_final/ncd-gp-transformer_epochs-100_stackSize-2_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8/',
    'results_final/ncd-transformer_epochs-100_stackSize-2_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8/'
]

csv_names = [
    'Conceptnet+OIE-GP+NCD', 'Conceptnet+NCD', 'OIE-GP+NCD', 'openNCDKB',
    'Conceptnet+OIE-GP+NCD,(N=2)', 'Conceptnet+NCD,(N=2)', 'OIE-GP+NCD,(N=2)', 'openNCDKB,(N=2)',
]


for num,csv_item in enumerate(csv_files):
    object_pairs = pd.read_csv(csv_item+'object_pairs.tsv.output_fstx_300d_indexed_sum_local_local',
            header=None,
            index_col=None,
            names=['object_pairs'])
    object_pairs_random = pd.read_csv(csv_item+'object_pairs_random.tsv.output_fstx_300d_indexed_sum_local_local',
            header=None,
            index_col=None,
            names=['object_pairs_random'])
    object_pairs_val = pd.read_csv(csv_item+'object_pairs_val.tsv.output_fstx_300d_indexed_sum_local_local',
            header=None,
            index_col=None,
            names=['object_pairs_val'])
    object_pairs_val_random = pd.read_csv(csv_item+'object_pairs_val_random.tsv.output_fstx_300d_indexed_sum_local_local',
            header=None,
            index_col=None,
            names=['object_pairs_val_random'])

    target_df = pd.concat([object_pairs, object_pairs_random], axis=1)
    source_df = pd.concat([object_pairs_val, object_pairs_val_random], axis=1)
    # print(target_df)
    # print(object_pairs)
    statistic_target, pvalue_target =stats.ttest_ind(target_df.dropna()['object_pairs'],
        target_df.dropna()['object_pairs_random'])
    # print('Target p value: ', pvalue_target)
    src_mean, src_rdn_mean = (np.mean(target_df.dropna()['object_pairs']),
        np.mean(target_df.dropna()['object_pairs_random']))
    statistic_source, pvalue_source =stats.ttest_ind(source_df.dropna()['object_pairs_val'],
        source_df.dropna()['object_pairs_val_random'])
    trg_mean, trg_rdn_mean = (np.mean(source_df.dropna()['object_pairs_val']),
        np.mean(source_df.dropna()['object_pairs_val_random']))

    print('\\hline')
    line = ''.join((csv_names[num], ' & ', f'{src_mean:.3f}', ' / ',
            f'{src_rdn_mean:.3f}' ,' & ', f'{pvalue_source:.2e}', ' & ',
            f'{trg_mean:.3f}', ' / ', f'{trg_rdn_mean:.3f}' ,' & ',
            f'{pvalue_target:.2e}', ' \\\\ ' ))
    print(line)
    
    

