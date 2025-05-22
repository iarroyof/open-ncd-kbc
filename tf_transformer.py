# prompt: In the above code cells we have two tensorflow scripts. 1. uses Transformer for a text to text generation task, however also uses tf version 2.6; 2. A tf script that uses the Attention GRU model for the same text to text generation task (same datasets). This later script is updated to latest versions of tf > 2.10. Use this updated version of the code as a reference to update the code in the fisrt cell, from tf 2.6 to tf > 2.10. Be very careful and detailed in comparing imports, file formats, and in avoiding attribute errors between versions

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import string, re, os
import math, random, functools
from joblib import Parallel, delayed
import logging
from pdb import set_trace as st
import argparse
import os
import time
import random
import functools
import re
import string
import math
import typing
from typing import Any, Tuple, Dict, Optional, List
from tensorflow.keras import mixed_precision
import wandb
from tensorflow.keras.layers import TextVectorization
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
"""
Code tha will be used to run the three different tests used for checking parameter/model transferability

TEST1: Train the transformer using conceptnet 600 and 70% from OpenIENCD

TEST2: Train the transformer using OpenIEGP and 70% from OpenIENCD

TEST3: Training the transformer using conceptnet 600, OpenIEGP and 70% OpenIENCD.

TODO:

    1. Add MDPI at the outputs of the predictions.
    2. Generate the input datasets similar to the way the ncd_conceptnet
    3. PMID seleccionar cuales se utilizaran para la encuesta

"""


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p')

cs_labels = False
pmid_val_labels = True

"""
Dataset names for results:
                conceptnet-ncd                                  (this only uses 60k from conceptnet)
                    data/ncd/openie5/ncd_oie5_conceptnet_train.tsv
                    data/ncd/openie5/ncd_oie5_conceptnet_test.tsv
                conceptnetFull-ncd
                    data/ncd/openie5/ncd_oie5_conceptnetFull_train.tsv
                    data/ncd/openie5/ncd_oie5_conceptnetFull_valid.tsv
                    data/tv_conceptnet_pmid.csv.csv
                oieGP-ncd
                    data/ncd/openie5/ncd_oie5_gp_oie_train.tsv
                    data/ncd/openie5/ncd_oie5_gp_oie_valid.tsv
                    neg_generator/oie_gp_shuffled_test.tsv  DONE
                conceptnetFull-oieGP-ncd
                    data/ncd/openie5/ncd_oie5_gp_oie_conceptnetFull_train.tsv
                    data/ncd/openie5/ncd_oie5_gp_oie_conceptnetFull_valid.tsv
                    data/tv_conceptnet_pmid.csv RUNNING
                    neg_generator/oie_gp_shuffled_test.tsv RUNNING
                ncd                                         (default value)
                    data/ncd/openie5/ncd_oie5_train.tsv
                    data/ncd/openie5/ncd_oie5_valid.tsv
                    data/ncd/openie5/ncd_oie5_test.tsv
"""


"""
Dataset names for results_final:
                ncd-conceptnet
                    data/ncd_conceptnet/ncd_conceptnet_train.tsv
                    data/ncd_conceptnet/ncd_conceptnet_valid.tsv
                ncd-gp            DONE
                    data/ncd_gp/ncd_gp_train.tsv
                    data/ncd_gp/ncd_gp_valid.tsv
                ncd-gp-conceptnet
                    data/ncd_gp_conceptnet/ncd_gp_conceptnet_train.tsv
                    data/ncd_gp_conceptnet/ncd_gp_conceptnet_valid.tsv
                ncd           DONE
                    data/ncd/openie5/ncd_oie5_train.tsv
                    data/ncd/openie5/ncd_oie5_valid.tsv
                    data/ncd/openie5/ncd_oie5_test.tsv
"""

""" TODO: add variable to detect train or val set. Only val set gets PMID.
"""


def prepare_data(
        line, start_token='[start] ', end_token=' [end]', include_pmid=False,
        include_labels=False, include_sent=False, all_start_end=False):
    """
    Prepares the data to be used by the model, for training and for validation
    - line: input sample from tsv
    - pmid: flag to determine whether to include pmid or not
    - include labels: whether to include the last two labels or not
    """
    line = line.split('\t')
    """ whether to remove pmid or not """
    if include_pmid:
        line_pmid = line[0] #save the pmid for later
        line.pop(0)
    else:
        line.pop(0) # sentence, pred, subject, object1, object2, .. objectN,  label, label

    """ Check whether the predicate is just full of empy chars or not
    and also to add spaces between the words (due to the conceptnet not being spaced)
    """
    pred = ' '.join(re.findall('[A-Z][a-z]*', line[1])).lower()
    if pred.isspace() or not pred:
        """ If is spaced, then just accept line as is"""
        pred = line[1]

    if not line[4].strip().isdigit(): # check if its not a digit (conceptnet dataset)
        if not re.match(r'^-?\d+(?:\.\d+)$', line[4].strip()):
            i = 4
            complements = [] #the additional objects
            while not line[i].isdigit():
                complements.append(line[i]) #append the additional object
                line.pop(i) #remove appendended oject

            line[3] = " ".join([line[3]] + complements) #join all the objects

    sample = [line[0], pred, line[2],
        start_token + line[3] + end_token, float(line[4].strip())] #create the sample

    if not include_labels:
        del sample[-1]
        sample_o = sample[-1]
    else:
        sample_o = tuple(sample[-2:])
    if not include_sent:
        del sample[0]
        sample_i = ' '.join([sample[1], sample[0]])
        if all_start_end: #whether start end token is also added to input
            sample_i = start_token + sample_i + end_token
    else:
        sample_i = ' '.join([sample[0], sample[2], sample[1]])

    if include_pmid:
        return  sample_i, sample_o, line_pmid
    else:
        return  sample_i, sample_o

@tf.keras.utils.register_keras_serializable()
def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(
        lowercase, "[%s]" % re.escape(strip_chars), "")


def format_dataset(in_phr, out_phr):

    in_phr = input_vectorizer(in_phr)
    out_phr = output_vectorizer(out_phr)

    return ({"encoder_inputs": in_phr,
            "decoder_inputs": out_phr[:, :-1],},
            out_phr[:, 1:])


def make_dataset(pairs, include_pmid=False):
    if include_pmid:
        in_phr_texts, out_phr_texts, _ = zip(*pairs)
    else:
        in_phr_texts, out_phr_texts = zip(*pairs)
    in_phr_texts = list(in_phr_texts)
    out_phr_texts = list(out_phr_texts)
    dataset = tf.data.Dataset.from_tensor_slices(
        (in_phr_texts, out_phr_texts))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset)

    return dataset.shuffle(2048).prefetch(16).cache()


def decode_sequence(input_sentence):
    tokenized_input_sentence = input_vectorizer([input_sentence])
    decoded_sentence = "[start]"

    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = output_vectorizer(
            [decoded_sentence])[:, :-1]
        predictions = transformer([tokenized_input_sentence,
                                    tokenized_target_sentence])
        try:
            sampled_token_index = np.argmax(predictions[0, i, :])
        except:
            logging.error("Invalid input to model: Invalid argument: slice "
                    "index {} of dimension 1 out of bounds.".format(i))
            continue
        try:
            sampled_token = out_phr_index_lookup[sampled_token_index]
        except KeyError:
            logging.error("KeyError: {}; output vocabulary length: {}".format(
                sampled_token_index, len(out_phr_index_lookup)))
            continue

        decoded_sentence += " " + sampled_token

        if sampled_token == "[end]":
            break

    return decoded_sentence


def sort_cols(columns):
    ends = np.unique([c[-2:] for c in columns])
    new_cols = []
    for e in ends:
        for c in columns:
            if c.endswith(e):
                new_cols.append(c)
    return new_cols


class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads,key_dim, **kwargs):
        super().__init__(**kwargs) # Updated super call
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            value_dim=key_dim,
            output_shape=embed_dim
        )
        self.dense_proj =  keras.Sequential(
            [layers.Dense(
                dense_dim, activation="relu",
                kernel_initializer='random_normal'),
                layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, mask=None):
        if mask is not None:
            # Ensure mask has the correct shape for attention
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
            # Expand mask for broadcasting: (batch_size, 1, 1, seq_len)
            padding_mask = tf.expand_dims(padding_mask, axis=1)
        else:
            padding_mask = None # No padding mask if input mask is None

        # MultiHeadAttention expects a mask of shape (batch_size, num_heads, seq_len, seq_len)
        # Or (batch_size, 1, seq_len, seq_len) which gets broadcast
        # The provided padding_mask is (batch_size, 1, 1, seq_len).
        # We need to tile it or reshape it depending on the attention mechanism's expectation.
        # Let's adjust the masking logic based on common Transformer implementations.
        # If mask is (batch_size, seq_len), attention_mask needs to be (batch_size, 1, seq_len, seq_len) or (batch_size, num_heads, seq_len, seq_len)
        # where a 1 at (i, j) means the i-th query token can attend to the j-th key token.
        # For self-attention in encoder, we usually just mask padding.
        # A (batch_size, 1, 1, seq_len) mask broadcast over query axis works for padding.
        # Let's keep the existing logic but ensure the mask type is float for `attention`.

        if mask is not None:
             # Create attention mask from padding mask: (batch_size, 1, 1, seq_len) -> (batch_size, 1, seq_len, seq_len) where it masks the value dimension (key)
            padding_mask_expanded = tf.cast(mask[:, tf.newaxis, tf.newaxis, :], dtype=tf.float32)
            attention_output = self.attention(
                query=inputs,
                value=inputs,
                key=inputs,
                attention_mask=padding_mask_expanded # Use the expanded mask
            )
        else:
             attention_output = self.attention(
                query=inputs,
                value=inputs,
                key=inputs,
                attention_mask=None # No mask
            )


        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)

        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "dense_dim": self.dense_dim,
            "num_heads": self.num_heads,
            "key_dim":  self.key_dim,
        })
        return config

    # Added `compute_output_shape` and `compute_mask` for potential TF 2.10+ compatibility
    # and explicit masking behavior.
    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, mask=None):
        # The mask from the input embedding should be propagated.
        return mask


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs) # Updated super call
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        # Position embeddings are fixed based on max sequence length
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        # We want to propagate the mask based on non-zero input tokens
        return tf.math.not_equal(inputs, 0)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "sequence_length": self.sequence_length,
            "vocab_size": self.vocab_size,
        })
        return config


class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads,key_dim, **kwargs):
        super().__init__(**kwargs) # Updated super call
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.key_dim = key_dim
        # Self-attention layer (masked)
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            value_dim=key_dim,
            output_shape=embed_dim
        )
        # Cross-attention layer (attends to encoder outputs)
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            value_dim=key_dim,
            output_shape=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [layers.Dense(
                latent_dim,
                activation="relu",
                kernel_initializer='random_normal'),
            layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "latent_dim": self.latent_dim,
            "key_dim": self.key_dim,
        })
        return config

    def call(self, inputs, encoder_outputs, mask=None):
        # inputs: target sequence (batch_size, target_seq_len, embed_dim)
        # encoder_outputs: output from encoder (batch_size, input_seq_len, embed_dim)
        # mask: padding mask for the target sequence (batch_size, target_seq_len)

        causal_mask = self.get_causal_attention_mask(inputs)

        # Combine padding mask and causal mask for the first attention layer (self-attention)
        self_attention_mask = causal_mask
        if mask is not None:
            # Ensure mask shape is compatible for broadcasting (batch_size, 1, target_seq_len)
            padding_mask_expanded = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
            # Combine causal mask and padding mask
            self_attention_mask = tf.minimum(causal_mask, padding_mask_expanded) # Use min as mask value is 0 or 1

        # Self-attention on the target sequence
        # Ensure the mask is float32 for the attention layer
        self_attention_mask = tf.cast(self_attention_mask, dtype=tf.float32)
        attention_output_1 = self.attention_1(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=self_attention_mask # Use the combined mask
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        # Cross-attention attends to encoder outputs
        # Here, the query comes from the decoder (out_1), and key/value from encoder_outputs
        # The mask for cross-attention should be the padding mask of the encoder_outputs.
        # The encoder outputs already have a padding mask propagated from the Encoder layer.
        # We need to access this mask. Assuming the encoder layer computes and propagates the mask:
        encoder_padding_mask = encoder_outputs._keras_mask if hasattr(encoder_outputs, '_keras_mask') else None

        cross_attention_mask = None
        if encoder_padding_mask is not None:
            # Create a mask that says where encoder outputs are NOT padded.
            # MultiHeadAttention expects a mask where 1 means "attend to this token".
            # So, the mask should be 1 for non-padding tokens and 0 for padding tokens in the encoder_outputs.
            # The shape needs to be compatible with (batch_size, 1, target_seq_len, input_seq_len).
            # We have (batch_size, input_seq_len) mask from encoder_outputs.
            # Expand it to (batch_size, 1, 1, input_seq_len) for broadcasting.
            encoder_padding_mask_expanded = tf.cast(encoder_padding_mask[:, tf.newaxis, tf.newaxis, :], dtype=tf.float32)
            cross_attention_mask = encoder_padding_mask_expanded


        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=cross_attention_mask, # Use the encoder padding mask
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        # This creates an upper triangular matrix of zeros and lower triangular matrix of ones
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        # Tile the mask to match the batch size
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1),
             tf.constant([1, 1],
             dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)

    # Added `compute_output_shape` and `compute_mask` for potential TF 2.10+ compatibility
    def compute_output_shape(self, inputs_shape, encoder_outputs_shape):
        return inputs_shape[0], inputs_shape[1], self.embed_dim

    def compute_mask(self, inputs, mask=None, encoder_outputs=None, encoder_outputs_mask=None):
        # The mask from the decoder input should be propagated.
        return mask


def build_transformer_encodec(sequence_length, max_features, model_dim, stack_size, latent_dim, num_heads, key_dim):
    # Encoder
    encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="encoder_inputs")
    # PositionalEmbedding computes and propagates the mask
    x = PositionalEmbedding(sequence_length, max_features, embed_dim=model_dim)(encoder_inputs) # dmodel
    encoder_outputs = x
    for n in range(stack_size): # Changed to iterate stack_size times
         # Pass the mask from the previous layer
        encoder_outputs = TransformerEncoder(embed_dim=model_dim, dense_dim=latent_dim, num_heads=num_heads, key_dim=key_dim)(encoder_outputs, mask=encoder_outputs._keras_mask if hasattr(encoder_outputs, '_keras_mask') else None)

    encoder = keras.Model(encoder_inputs, encoder_outputs, name="encoder")


    # Decoder
    decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="decoder_inputs")
    # PositionalEmbedding computes and propagates the mask for decoder inputs
    x = PositionalEmbedding(sequence_length + 1, max_features, embed_dim=model_dim)(decoder_inputs) # d model

    # The encoded sequence inputs will also be passed to the decoder
    encoded_seq_inputs = keras.Input(shape=(None, model_dim), name="decoder_state_inputs")
    # Ensure the mask from the encoder outputs is available for cross-attention
    encoded_seq_inputs._keras_mask = encoder_outputs._keras_mask if hasattr(encoder_outputs, '_keras_mask') else None

    decoder_outputs = x
    for n in range(stack_size): # Changed to iterate stack_size times
        # Pass decoder input mask and encoder output mask to the decoder layer
        decoder_outputs = TransformerDecoder(embed_dim=model_dim, latent_dim=latent_dim, num_heads=num_heads, key_dim=key_dim)(
            decoder_outputs, encoded_seq_inputs, mask=decoder_outputs._keras_mask if hasattr(decoder_outputs, '_keras_mask') else None)

    # Add dropout after the decoder stack
    decoder_outputs = layers.Dropout(0.1)(decoder_outputs) # Common dropout rate

    # Final dense layer for output vocabulary distribution
    decoder_outputs = layers.Dense(max_features, activation="softmax")(decoder_outputs)
    decoder = keras.Model([decoder_inputs, encoded_seq_inputs], decoder_outputs, name="decoder")

    # The full Transformer model combining encoder and decoder
    transformer_outputs = decoder([decoder_inputs, encoder_outputs])
    transformer = keras.Model(
        inputs=[encoder_inputs, decoder_inputs],
        outputs=transformer_outputs,
        name="transformer"
    )

    return transformer

def save_predictions(pairs, to_file_path):
    logging.info("OBTAINING PREDICTIONS TO {}".format(to_file_path))
    """ Generate predictions for test set """
    out_file = open(to_file_path, 'w')
    write_result_row = functools.partial(build_row_result, out_file=out_file)
    out_file.write('\t'.join(['Subj_Pred', 'Obj', 'Obj_true\n']))
    # Parallel processing might have issues writing to a single file handle.
    # It's safer to collect results and then write.
    results = Parallel(n_jobs=1)( # n_jobs=1 means no parallelization, which is safer for file writing
        delayed(build_row_result_single)(inp, out, pmid)
            for inp, out, pmid in pairs)
    for result_line in results:
         out_file.write(result_line + '\n')

    out_file.close()

def build_row_result_single(inp, out, pmid):
    """Helper function for parallel prediction, returns a single line."""
    translated = decode_sequence(inp)
    line = [inp, translated, out]
    return '\t'.join(line)


def load_vectorizer(from_file):
    """
    Load a saved TextVectorization layer from disk.
    Ensure loading uses the new .keras filenames:
    # When loading vectorizers, use:
    input_vectorizer = load_vectorizer(f"{vectorizer_path}in_vect_model.keras")
    output_vectorizer = load_vectorizer(f"{vectorizer_path}out_vect_model.keras")
    """
    # Load the Sequential model containing the TextVectorization layer
    model = tf.keras.models.load_model(from_file)

    # Get the TextVectorization layer from the model
    # Assuming it's the second layer (index 1) after the InputLayer (index 0)
    vectorizer_layer = model.layers[1]

    # Get the vocabulary and config from the loaded layer
    vocab = vectorizer_layer.get_vocabulary()
    config = vectorizer_layer.get_config()

    # Create a new TextVectorization layer from the config
    # Use from_config for TF 2.10+ compatibility
    vectorizer = TextVectorization.from_config(config)

    # Initialize vocabulary by adapting to a dummy string
    # This is often necessary after creating from_config before setting vocabulary
    vectorizer.adapt(['Initializing vectorizer'])

    # Set the loaded vocabulary
    vectorizer.set_vocabulary(vocab)

    return vectorizer

def save_vectorizer(vectorizer, to_file):
    """Save a TextVectorization layer to disk."""
    # Create a simple Sequential model containing the TextVectorization layer
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(1,), dtype=tf.string), # Input layer
        vectorizer # The TextVectorization layer
    ])
    # Compile the model (required for saving)
    model.compile()
    # Save the model in the native Keras format (.keras)
    model.save(to_file, save_format='keras')

# MAIN
parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument("-tf", "--trainFlag", action='store_true',
    help = "Whether to train (unspecified is False, so giving"
        " evaluation data is required, --validData, as train "
        "and test data will be ignored.")

parser.add_argument("-ev", "--evaluateFlag", action='store_true',
    help = "Whether to get evaluation metrics for validation data.")

parser.add_argument("-mp", "--predictFlag", action='store_true',
    help = "Whether to generate triples from test and validation data.")

parser.add_argument("-s", "--seqLen", type=int,
    default=30, help = "Per-sample sequence length")

parser.add_argument("-f", "--nFeatures", type=int,
    default=15000, help = "Maximum vocabulary size")

parser.add_argument("-b", "--batchSize", type=int,
    default=64, help = "Batch size")

parser.add_argument("-e", "--nEpochs", type=int,
    default=100, help = "Number of training epochs (training can stop earlier"
        " as improvements do not overmoce 0.005*loss within 10 epochs.)") # base = 30

parser.add_argument("-N", "--stackSize", type=int,
    default=1, help = "Stack size of encoder/decoders")

parser.add_argument("-kd", "--keyDim", type=int,
    default=0, help = "Key dimensionality; default is 0,"
        " interpreted as keyDim = modelDim/nHeads") #base = 64

parser.add_argument("-md", "--modelDim", type=int,
    default=512, help = "Embedding Dimensionality of input and output") #base = 512

parser.add_argument("-l", "--latentDim", type=int,
    default=2048, help = "Hidden embedding dimensionality. Inner layer"
        " dimensionality. Dense dim") #dff= 2048

parser.add_argument("-H", "--nHeads", type=int,
    default=8, help = "Number of attention heads") # h = 8

parser.add_argument("-D", "--nDemo", type=int,
    default=-1, help = "Number of predicted test samples to save as output"
        ". Only has effect when -mp parameter is enabled.")

parser.add_argument("-tnD", "--trainData", type=str,
    default="data/oie-gp_target/ncd_gp_conceptnet_train.tsv",
    help = "Training data (TSV file)")

parser.add_argument("-vD", "--validData", type=str,
    default="data/oie-gp_target/ncd_gp_valid.tsv",
    help = "Valid data (TSV file)")

parser.add_argument("-ttD", "--testData", type=str,
    default="data/oie-gp_target/ncd_gp_conceptnet_test.tsv",
    help = "Valid data (TSV file)")

parser.add_argument("-dN", "--datasetName", type=str,
    default="OIEGP",
    help = "Name used for directory naming")

parser.add_argument("-tN", "--testName", type=str,
    default="SD",
    help="Name used for the generated csv prediction file")

# Read arguments from command line
args = parser.parse_args()
logging.info("Working for:\n")
logging.info(pd.DataFrame([(arg, getattr(args, arg)) for arg in vars(args)]))

# Hyperparameters
train_flag = args.trainFlag
eval = args.evaluateFlag
to_predict = args.predictFlag
sequence_length = args.seqLen
max_features = args.nFeatures
batch_size = args.batchSize
n_epochs = args.nEpochs
stack_size= args.stackSize
key_dim = (int(args.modelDim/args.nHeads)
            if args.keyDim <= 0 or isinstance(args.keyDim, str)
            else args.keyDim)
model_dim = args.modelDim
latent_dim = args.latentDim
num_heads = args.nHeads
# Input data
training_data = args.trainData
validation_data = args.validData
test_data = args.testData
# Other settings
n_demo = args.nDemo
dataset_name = args.datasetName
test_name = args.testName

# Use SparseCategoricalCrossentropy with from_logits=False since the model output uses softmax
metric = "sparse_categorical_accuracy" # Use SparseCategoricalAccuracy for sparse targets
loss = "sparse_categorical_crossentropy"

# Define checkpoint path using the new Keras native format
checkpoint_path = (
    f"results_final{os.sep}{dataset_name}-transformer_epochs-{n_epochs}_stackSize-{stack_size}_"
    f"seqlen-{sequence_length}_maxfeat-{max_features}_batch-{batch_size}_keydim-{key_dim}_"
    f"modeldim-{model_dim}_latent-{latent_dim}_heads-{num_heads}{os.sep}cp.weights.h5" # Use .h5 or .weights.h5 for weights only
)

logging.info(checkpoint_path)
checkpoint_dir = os.path.dirname(checkpoint_path)
out_dir = os.path.dirname(checkpoint_dir) + os.sep # Adjusted out_dir to the parent of the checkpoint directory

# Create directories if they don't exist
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(out_dir, exist_ok=True) # Ensure out_dir exists


strip_chars = string.punctuation
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")

if train_flag:
    with open(training_data) as f:
        train_text = f.readlines()

    with open(test_data) as f:
        test_text = f.readlines()

    with open(validation_data) as f:
        val_text = f.readlines()

    """ They are not really pairs anymore because, pmid for
         each sentences is also returned """

    train_pairs = list(
        map(functools.partial(
            prepare_data, include_labels=cs_labels), train_text))

    test_pairs= list(
        map(functools.partial(
            prepare_data,
            include_labels=cs_labels,
            include_pmid=True), test_text))

    val_pairs= list(
        map(functools.partial(
            prepare_data,
            include_labels=cs_labels,
            include_pmid=True), val_text))

    train_in_texts = [pair[0] for pair in train_pairs]
    if cs_labels:
        train_out_texts = [pair[1][0] for pair in train_pairs]
    else:
        train_out_texts = [pair[1] for pair in train_pairs]

    # Updated TextVectorization usage for TF 2.10+
    input_vectorizer = TextVectorization(
        output_mode="int", max_tokens=max_features,
        output_sequence_length=sequence_length,
        standardize=custom_standardization)

    output_vectorizer = TextVectorization(
        output_mode="int", max_tokens=max_features,
        output_sequence_length=sequence_length+1, # +1 for the start token
        standardize=custom_standardization)

    logging.info("Adapting input text vectorizer...")
    input_vectorizer.adapt(train_in_texts)
    logging.info("Adapting output text vectorizer...")
    output_vectorizer.adapt(train_out_texts)

    #saving the vectorizers also
    vectorizer_save_dir = os.path.join(out_dir, 'vectorizers')
    os.makedirs(vectorizer_save_dir, exist_ok=True)
    save_vectorizer(
        vectorizer=input_vectorizer, to_file=os.path.join(vectorizer_save_dir, 'in_vect_model.keras')) # Use .keras format
    save_vectorizer(
        vectorizer=output_vectorizer, to_file=os.path.join(vectorizer_save_dir, 'out_vect_model.keras')) # Use .keras format
    logging.info(f"Saved vectorizers to {vectorizer_save_dir}")


    train_ds = make_dataset(train_pairs)
    # test_ds is used for validation during training in the original code
    # Let's use test_ds as the validation dataset for model.fit
    # And keep val_ds for separate evaluation/prediction after training
    val_ds_for_fit = make_dataset(test_pairs, include_pmid=pmid_val_labels) # Use test_pairs as validation for fit
    val_ds_for_eval_predict = make_dataset(val_pairs, include_pmid=pmid_val_labels) # Keep val_pairs for final eval/predict


else:
    # If not training, load model weights and vectorizers
    vectorizer_save_dir = os.path.join(out_dir, 'vectorizers')
    if os.path.isdir(vectorizer_save_dir):
        logging.info(f"Loading Vectorizers from {vectorizer_save_dir}")
        input_vectorizer =  load_vectorizer(os.path.join(vectorizer_save_dir, 'in_vect_model.keras')) # Load .keras format
        output_vectorizer = load_vectorizer(os.path.join(vectorizer_save_dir, 'out_vect_model.keras')) # Load .keras format
    else:
        logging.error("NO trained model weights or vectorizers found in: {}".format(out_dir))
        exit()

    with open(validation_data) as f:
        val_text = f.readlines()
    """ They are not really pairs anymore because, pmid for each
         sentences is also returned """
    val_pairs= list(
        map(functools.partial(
            prepare_data,
            include_labels=cs_labels,
            include_pmid=True), val_text))

    val_ds_for_eval_predict = make_dataset(val_pairs, include_pmid=pmid_val_labels)
    # If not training, we don't need val_ds_for_fit

# Update max_features based on adapted vocabularies
max_vocab = max([
        len(input_vectorizer.get_vocabulary()),
        len(output_vectorizer.get_vocabulary())])
if max_features > max_vocab:
    max_features = max_vocab
    logging.info(f"Adjusted max_features to max vocabulary size: {max_features}")


es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=10,
                                                min_delta=0.005,
                                                mode='auto',
                                                restore_best_weights=True,
                                                verbose=1)

# Define ModelCheckpoint callback to save weights only
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True, # Save only the model weights
    verbose=1
)

transformer = build_transformer_encodec(
    sequence_length=sequence_length, # Sequence length for inputs
    max_features=max_features,
    model_dim=model_dim,
    stack_size=stack_size,
    latent_dim=latent_dim,
    num_heads=num_heads,
    key_dim=key_dim
)
transformer.summary()

# Use Adam optimizer without specifying learning rate for default TF 2.10+ behavior
transformer.compile(optimizer=tf.keras.optimizers.Adam(),
                    loss=loss, # Using SparseCategoricalCrossentropy with from_logits=False (softmax output)
                    metrics=[metric]) # Using SparseCategoricalAccuracy


if train_flag:
    logging.info("Training Transformer Semantic EncoDec")
    history = transformer.fit(train_ds,
        epochs=n_epochs,
        validation_data=val_ds_for_fit, # Use test_ds as validation during fit
            callbacks=[ cp_callback, es_callback]) # Removed wandb callbacks if not used

    logging.info("TRAINED!!")
    rdf = pd.DataFrame(history.history)
    rdf.to_csv(os.path.join(out_dir, "history.csv")) # Save history in out_dir

    fig, axes = plt.subplots(2, 1)
    rdf[sort_cols(rdf.columns)].iloc[:, :2].plot(ax=axes[0])
    axes[0].grid(b=True,which='major',axis='both',linestyle='--')
    rdf[sort_cols(rdf.columns)].iloc[:, 2:].plot(ax=axes[1])
    axes[1].grid(b=True,which='major',axis='both',linestyle='--')
    plt.savefig(os.path.join(out_dir, 'history_plot.pdf')) # Save plot in out_dir

    # Model weights are already saved by cp_callback


else:
    # vectorizers have been loaded previously
    # Load model weights if not training
    logging.info(f"Loading model weights from {checkpoint_path}")
    try:
        transformer.load_weights(checkpoint_path)
        logging.info("Model weights loaded successfully.")
    except tf.errors.NotFoundError:
        logging.error(f"Error: Model weights file not found at {checkpoint_path}")
        exit()
    except Exception as e:
        logging.error(f"An error occurred loading model weights: {e}")
        exit()


# Prepare for prediction/evaluation
out_phr_vocab = output_vectorizer.get_vocabulary()
out_phr_index_lookup = dict(zip(range(len(out_phr_vocab)), out_phr_vocab))
# The length of the output sequence for decoding. Use sequence_length + 1 because output includes [start] and [end]
max_decoded_sentence_length = sequence_length + 1

val_file = os.path.join(out_dir, 'evaluation_on_validation_data.txt') # Save eval results in out_dir

if train_flag or eval or to_predict:
     # Need the raw pairs for prediction/evaluation, not just the batched dataset
    with open(validation_data) as f:
        val_text_raw = f.readlines()
    val_pairs_raw = list(
        map(functools.partial(
            prepare_data,
            include_labels=cs_labels,
            include_pmid=True), val_text_raw))

    if to_predict:
        logging.info("OBTAINING PREDICTIONS FOR VALIDATION SET")
        pairs_to_predict = val_pairs_raw # Use raw pairs for prediction

        if n_demo >= 0: # Apply n_demo if specified
            random.shuffle(pairs_to_predict)
            pairs_to_predict = pairs_to_predict[:n_demo]

        if pairs_to_predict:
             save_predictions(pairs=pairs_to_predict,
                 to_file_path=os.path.join(out_dir, f'{test_name}_predictions.tsv')) # Save predictions in out_dir with test_name
             logging.info("KBC FOR VALIDATION SET WRITTEN TO {}".format(
                 os.path.join(out_dir, f'{test_name}_predictions.tsv')))
        else:
             logging.warning("No validation samples to predict.")


    if eval:
        logging.info(
            'Validating Transformer Semantic EncoDec to {}'.format(val_file))
        # Evaluate using the dedicated validation dataset
        val_loss, val_acc = transformer.evaluate(val_ds_for_eval_predict)
        with open(val_file, 'w') as ev:
            line = 'Validation_loss: {}\nValidation_acc: {}\n'.format(
                val_loss, val_acc)
            ev.write(line)
        logging.info("Model evaluated. See results in {}".format(val_file))


logging.info("All tasks finished. See results in or load trained model "
    "from directory {}\n".format(out_dir))
