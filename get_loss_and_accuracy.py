from os import name
import pandas as pd
from scipy import stats
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import string, re, os



csv_files  = [
    # 'results_final/ncd-gp-conceptnet-transformer_epochs-40_stackSize-1_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8/',
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

strip_chars = string.punctuation
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")

def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(
        lowercase, "[%s]" % re.escape(strip_chars), "")

for num,csv_item in enumerate(csv_files):

    print("Loading vectorizers")
    loaded_in_vect_model = tf.keras.models.load_model(csv_item+'in_vect_model',
                            custom_objects={ 
                                # 'TextVectorization':layers.experimental.preprocessing.TextVectorization,
                                'custom_standardization':custom_standardization},
                                compile=False)
    loaded_out_vect_model = tf.keras.models.load_model(csv_item+'out_vect_model',
                            custom_objects={ 
                                # 'TextVectorization':layers.experimental.preprocessing.TextVectorization,
                                'custom_standardization':custom_standardization},
                                compile=False)
    input_vectorizer = loaded_in_vect_model.layers[0]
    output_vectorizer = loaded_out_vect_model.layers[0]


    object_pairs = pd.read_csv(csv_item+'object_pairs.tsv',
            sep='\t',
            header=None,
            index_col=None,
            names=['obj','obj_true'])
    object_pairs_random = pd.read_csv(csv_item+'object_pairs_random.tsv',
            sep='\t',
            header=None,
            index_col=None,
            names=['obj','obj_true'])
    object_pairs_val = pd.read_csv(csv_item+'object_pairs_val.tsv',
            sep='\t',
            header=None,
            index_col=None,
            names=['obj','obj_true'])
    object_pairs_val_random = pd.read_csv(csv_item+'object_pairs_val_random.tsv',
            sep='\t',
            header=None,
            index_col=None,
            names=['obj','obj_true'])

    tokenized_obj_list = list()
    tokenized_obj_true_list = list()
    for num, objects in object_pairs.iterrows():
        # print(objects['obj'])
        # print(objects['obj_true'])
        tokenized_obj = input_vectorizer([objects['obj']]).numpy()[0]
        tokenized_obj_true = input_vectorizer([objects['obj_true']]).numpy()[0]
        tokenized_obj_list.append(tokenized_obj)
        tokenized_obj_true_list.append(tokenized_obj_true)
        if num ==200: 
            break 
    obj_one_hot = tf.one_hot(indices=tokenized_obj_list, depth=15000).numpy()
    obj_true_one_hot = tf.one_hot(indices=tokenized_obj_true_list, depth=15000).numpy()

    # y_true = [1, 2]
    # y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
    # Using 'auto'/'sum_over_batch_size' reduction type.
    scce = tf.keras.losses.CategoricalCrossentropy()
    print(scce(obj_one_hot, obj_true_one_hot,).numpy())
    print("sparse catergorica c entropy:", scce)
    print('first')
    
    
 
    # print(line)
    
    

