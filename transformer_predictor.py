from transformer_modules import *
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import string, re, os
import math, random
import logging
import argparse

from pdb import set_trace as st

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
config_file = 'predictor_configuration.json'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p')

def make_prediction(val_pair, n_demo, out_dir):
    if not (n_demo < 0 or isinstance(n_demo, str)):
        # Create duplicates of the same input query to make multiple predictions
        val_pairs = [val_pair] * n_demo
        save_predictions(pairs=val_pairs,
                    to_file_path=out_dir + 'val_predictions.csv')

        logging.info("KBC FOR VALIDATION SET WRITTEN TO {}".format(
                    out_dir + 'val_predictions.csv'))
        return None
    else:
        # Get a unique prediction for the imput query, and 
        inp = val_pair[0]
        return decode_sequence(inp)

(dataset_name,
n_epochs,
stack_size,
sequence_length,
max_features,
batch_size,
key_dim,
model_dim,
latent_dim,
num_heads,
out_dir) = get_config(config_file)
n_demo = 10
#n_demo = -1

if os.path.isdir(out_dir):
    logging.info("Loading Vectorizers")
else:
    logging.error("NO model trained and directory with specified parameters"
        " exists in: {}".format(config_file))
    exit()


logging.info("OBTAINING PREDICTION FOR INPUT QUERY...")

"""
The inputs 'Subject_Predicate' and 'Object' should be provided by a function
implementing the Allen AI's Semantic Role Labeling method. I set their values
for testing purposes:
"""
Subject_Predicate = "The cause of lung cancer can be"
Object = "DNA methylation"


prediction = make_prediction((Subject_Predicate, Object), n_demo, out_dir)
if prediction is None:
    pass
else:
    prediction = prediction.replace("[start] ", '').replace(" [end]", '')
    print(f"\n\n\n\nGiven sentence: {Subject_Predicate} {Object}")
    print(f"Generated sentence: {Subject_Predicate} {prediction}")
