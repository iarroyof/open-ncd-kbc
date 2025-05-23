import os
import time
import random
import argparse
import logging
import functools
import re
import string
import math

import numpy as np
import pandas as pd

import typing
from typing import Any, Tuple, Dict, Optional, List

os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision
from tensorflow.keras.layers import TextVectorization

import wandb

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p'
)

CS_LABELS = False
STRIP_CHARS = string.punctuation.replace("[", "").replace("]", "")

# Utility Functions

def save_vectorizer(vectorizer, to_file):
    """Save a TextVectorization layer as a Keras model to a file."""
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(1,), dtype=tf.string),
        vectorizer
    ])
    model.compile()
    model.save(to_file, save_format='keras')

def load_vectorizer(from_file):
    """Load a TextVectorization layer from a saved Keras model."""
    model = tf.keras.models.load_model(from_file)
    vectorizer_layer = model.layers[0]
    vocab = vectorizer_layer.get_vocabulary()
    config = vectorizer_layer.get_config()
    vectorizer = TextVectorization.from_config(config)
    vectorizer.adapt(['Initializing vectorizer'])
    vectorizer.set_vocabulary(vocab)
    return vectorizer

def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, f"[{re.escape(STRIP_CHARS)}]", "")

def prepare_data(line, start_token='[start] ', end_token=' [end]', pmid=True,
                 include_labels=False, include_sent=False, all_start_end=False):
    line = line.split('\t')
    if pmid:
        line.pop(0)
    pred = ' '.join(re.findall('[A-Z][a-z]*', line[1])).lower() or line[1]
    if not line[4].strip().isdigit() and not re.match(r'^-?\d+(?:\.\d+)$', line[4].strip()):
        complements = []
        i = 4
        while not line[i].isdigit():
            complements.append(line[i])
            line.pop(i)
        line[3] = " ".join([line[3]] + complements)
    sample = [line[0], pred, line[2], f"{start_token}{line[3]}{end_token}", float(line[4].strip())]
    if not include_labels:
        del sample[-1]
        sample_o = sample[-1]
    else:
        sample_o = tuple(sample[-2:])
    if not include_sent:
        del sample[0]
        sample_i = ' '.join([sample[1], sample[0]])
        if all_start_end:
            sample_i = f"{start_token}{sample_i}{end_token}"
    else:
        sample_i = ' '.join([sample[0], sample[2], sample[1]])
    return sample_i, sample_o

def parse_dataset_name(training_data):
    training_data = training_data.split(os.sep)[-1].lower()
    names_dic = {"NCD": 'ncd' in training_data, "GP": 'gp' in training_data, "CN": 'conceptnet' in training_data}
    return '-'.join(k for k, v in names_dic.items() if v)

# Model Architecture (unchanged classes: BahdanauAttention, Encoder, Decoder, MaskedLoss, TrainTranslator, Translator, etc.)
# For brevity, assuming these are defined as in the reference code above.

def main():
    parser = argparse.ArgumentParser(description="Train a sequence-to-sequence model with attention.")
    parser.add_argument("-s", "--seqLen", type=int, default=50, help="Per-sample sequence length")
    parser.add_argument("-u", "--nSteps", type=int, default=1024, help="Number of hidden recurrent steps (units)")
    parser.add_argument("-f", "--nFeatures", type=int, default=15000, help="Maximum vocabulary size")
    parser.add_argument("-b", "--batchSize", type=int, default=64, help="Batch size")
    parser.add_argument("-e", "--nEpochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("-d", "--embeddingDim", type=int, default=1024, help="Word embedding dimensionality")
    parser.add_argument("-l", "--numLayers", type=int, default=1, help="Number of stacked GRU layers")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability for GRU layers")
    parser.add_argument("-D", "--nDemo", type=int, default=20, help="Number of test samples to predict")
    parser.add_argument("-T", "--trainData", type=str, default="data/ncd_conceptnet/ncd_conceptnet_train.tsv", help="Training data TSV")
    parser.add_argument("-t", "--testData", type=str, default="data/ncd_conceptnet/ncd_conceptnet_valid.tsv", help="Test data TSV")
    args = parser.parse_args()

    gpus = tf.config.list_physical_devices('GPU')
    for dev in gpus:
        tf.config.experimental.set_memory_growth(dev, True)
    mixed_precision.set_global_policy("mixed_float16")

    dataset_name = parse_dataset_name(args.trainData)

    logging.info("Preparing train and test data")
    with open(args.trainData) as f:
        train_text = f.readlines()
    with open(args.testData) as f:
        val_text = f.readlines()
    train_pairs = list(map(functools.partial(prepare_data, include_labels=CS_LABELS, all_start_end=True), train_text))
    val_pairs = list(map(functools.partial(prepare_data, include_labels=CS_LABELS, all_start_end=True), val_text))

    train_in, train_out = zip(*train_pairs)
    test_in, test_out = zip(*val_pairs)
    train_in = [str(s) for s in train_in]
    train_out = [str(s) for s in train_out]
    dataset = (
        tf.data.Dataset.from_tensor_slices((train_in, train_out))
          .shuffle(len(train_in))
          .batch(args.batchSize, drop_remainder=True)
          .prefetch(tf.data.AUTOTUNE)
    )
    test_in = [str(s) for s in test_in]
    test_out = [str(s) for s in test_out]
    test_dataset = (
        tf.data.Dataset.from_tensor_slices((test_in, test_out))
          .shuffle(len(test_in))
          .batch(args.batchSize, drop_remainder=True)
          .prefetch(tf.data.AUTOTUNE)
    )

    input_vectorizer = TextVectorization(
        output_mode="int", max_tokens=args.nFeatures, output_sequence_length=args.seqLen,
        standardize=custom_standardization)
    output_vectorizer = TextVectorization(
        output_mode="int", max_tokens=args.nFeatures, output_sequence_length=args.seqLen + 1,
        standardize=custom_standardization)
    
    train_in_texts = [pair[0] for pair in train_pairs]
    train_out_texts = [pair[1][0] if CS_LABELS else pair[1] for pair in train_pairs]
    logging.info("Training input text vectorizer")
    input_vectorizer.adapt(train_in_texts)
    logging.info("Training output text vectorizer")
    output_vectorizer.adapt(train_out_texts)

    # Updated vectorizer saving
    checkpoint_path = (
        f"results_final{os.sep}{dataset_name}-transformer_epochs-{args.nEpochs}*stackSize-{args.numLayers}*"
        f"seqlen-{args.seqLen}_maxfeat-{args.nFeatures}_batch-{args.batchSize}*keydim-64*"
        f"modeldim-512_latent-1024_heads-2{os.sep}cp.weights.h5"
    )
    checkpoint_dir = os.path.dirname(checkpoint_path)
    out_dir = os.path.dirname(checkpoint_dir) + os.sep
    vectorizer_save_dir = os.path.join(out_dir, 'vectorizers')
    os.makedirs(vectorizer_save_dir, exist_ok=True)
    save_vectorizer(input_vectorizer, os.path.join(vectorizer_save_dir, 'in_vect_model.keras'))
    save_vectorizer(output_vectorizer, os.path.join(vectorizer_save_dir, 'out_vect_model.keras'))
    logging.info(f"Saved vectorizers to {vectorizer_save_dir}")

    # Model setup and training (unchanged for brevity)
    train_translator = TrainTranslator(
        args.embeddingDim, args.nSteps, input_vectorizer, output_vectorizer, args.numLayers, args.dropout
    )
    train_translator.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=MaskedLoss()
    )
    history = train_translator.fit(dataset, validation_data=test_dataset, epochs=args.nEpochs)
    logging.info("Training completed successfully")

    # Inference (unchanged for brevity)
    if args.nDemo >= 0:
        random.shuffle(val_pairs)
        val_pairs = val_pairs[:args.nDemo]
        inp_, targ_ = zip(*val_pairs)
        translator = Translator(train_translator.encoder, train_translator.decoder, input_vectorizer, output_vectorizer)
        result = translator.tf_translate(tf.constant(list(inp_)))['text'].numpy()
        result_df = pd.DataFrame({'Subj_Pred': inp_, 'Obj': result, 'Obj_true': targ_})
        result_df.to_csv(f"{out_dir}predictions.csv")
        print(result_df)
        logging.info(f"Results written to {out_dir}predictions.csv")

if __name__ == "__main__":
    main()
