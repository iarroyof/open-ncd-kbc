import argparse
import logging
import os
import random
import re
import string
import functools
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
from joblib import Parallel, delayed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p'
)

# Constants
STRIP_CHARS = string.punctuation.replace("[", "").replace("]", "")
CS_LABELS = False
PMID_VAL_LABELS = True

# Utility Functions
def custom_standardization(input_string: tf.Tensor) -> tf.Tensor:
    """Standardize input strings by lowering case and removing punctuation."""
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, f"[{re.escape(STRIP_CHARS)}]", "")

def prepare_data(
    line: str,
    start_token: str = '[start] ',
    end_token: str = ' [end]',
    include_pmid: bool = False,
    include_labels: bool = False,
    include_sent: bool = False,
    all_start_end: bool = False
) -> Tuple[str, Any, Optional[str]]:
    """Prepare data for model input, handling PMID, labels, and tokenization."""
    parts = line.strip().split('\t')
    if include_pmid:
        pmid = parts.pop(0)
    else:
        parts.pop(0)  # Remove first column if not including PMID

    pred = ' '.join(re.findall('[A-Z][a-z]*', parts[1])).lower() or parts[1]
    complements = []
    i = 4
    while i < len(parts):
        try:
            float(parts[i])
            break  # Found the label
        except ValueError:
            complements.append(parts[i])
            parts.pop(i)  # Remove additional object
    else:
        raise ValueError(f"No label found in line: {line}")

    if complements:
        parts[3] = " ".join([parts[3]] + complements)

    label = float(parts[4].strip())
    sample = [parts[0], pred, parts[2], f"{start_token}{parts[3]}{end_token}", label]
    if not include_labels:
        sample.pop(-1)
        sample_o = sample[-1]
    else:
        sample_o = tuple(sample[-2:])

    if not include_sent:
        sample.pop(0)
        sample_i = ' '.join([sample[1], sample[0]])
        if all_start_end:
            sample_i = f"{start_token}{sample_i}{end_token}"
    else:
        sample_i = ' '.join([sample[0], sample[2], sample[1]])

    return (sample_i, sample_o, pmid) if include_pmid else (sample_i, sample_o)

def format_dataset(in_phr: tf.Tensor, out_phr: tf.Tensor) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
    """Format dataset for Transformer input-output pairs."""
    in_phr = input_vectorizer(in_phr)
    out_phr = output_vectorizer(out_phr)
    return ({"encoder_inputs": in_phr, "decoder_inputs": out_phr[:, :-1]}, out_phr[:, 1:])

def make_dataset(pairs: List[Tuple], include_pmid: bool = False) -> tf.data.Dataset:
    """Create a TensorFlow dataset from input-output pairs."""
    if include_pmid:
        in_texts, out_texts, _ = zip(*pairs)
    else:
        in_texts, out_texts = zip(*pairs)
    dataset = tf.data.Dataset.from_tensor_slices((list(in_texts), list(out_texts)))
    return dataset.batch(batch_size).map(format_dataset).shuffle(2048).prefetch(16).cache()

def decode_sequence(input_sentence: str, transformer: keras.Model, max_length: int) -> str:
    """Decode a sequence using the Transformer model."""
    tokenized_input = input_vectorizer([input_sentence])
    decoded = "[start]"
    for i in range(max_length):
        tokenized_target = output_vectorizer([decoded])[:, :-1]
        predictions = transformer([tokenized_input, tokenized_target])
        try:
            sampled_token_idx = np.argmax(predictions[0, i, :])
            sampled_token = out_phr_index_lookup[sampled_token_idx]
        except (IndexError, KeyError) as e:
            logging.error(f"Decoding error at step {i}: {e}")
            continue
        decoded += " " + sampled_token
        if sampled_token == "[end]":
            break
    return decoded

# Custom Layers
class PositionalEmbedding(layers.Layer):
    """Layer to add positional embeddings to token embeddings."""
    def __init__(self, sequence_length: int, vocab_size: int, embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_embeddings = layers.Embedding(vocab_size, embed_dim)
        self.position_embeddings = layers.Embedding(sequence_length, embed_dim)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs: tf.Tensor, mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        return tf.math.not_equal(inputs, 0)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"sequence_length": self.sequence_length, "vocab_size": self.vocab_size, "embed_dim": self.embed_dim})
        return config

import tensorflow as tf

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, model_dim, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.dense_proj = tf.keras.Sequential([
            tf.keras.layers.Dense(model_dim, activation="relu"),
            tf.keras.layers.Dense(model_dim)
        ])
        self.layernorm_1 = tf.keras.layers.LayerNormalization()
        self.layernorm_2 = tf.keras.layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, mask=None, training=False):
        # Handle the mask for attention
        if mask is not None:
            # Original mask shape: (batch_size, seq_len)
            # Reshape to (batch_size, 1, 1, seq_len) for broadcasting
            padding_mask = tf.cast(mask[:, tf.newaxis, tf.newaxis, :], dtype=tf.bool)
        else:
            padding_mask = None

        # Self-attention with the mask
        attention_output = self.attention(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=padding_mask
        )

        # Residual connection and normalization
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "dense_dim": self.dense_dim, "num_heads": self.num_heads, "key_dim": self.key_dim})
        return config

class TransformerDecoder(layers.Layer):
    """Transformer decoder layer with self-attention and cross-attention."""
    def __init__(self, embed_dim: int, latent_dim: int, num_heads: int, key_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.attention_1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, value_dim=key_dim)
        self.attention_2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, value_dim=key_dim)
        self.dense_proj = keras.Sequential([
            layers.Dense(latent_dim, activation="relu", kernel_initializer='random_normal'),
            layers.Dense(embed_dim)
        ])
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs: tf.Tensor, encoder_outputs: tf.Tensor, mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.float32)
            self_attention_mask = tf.minimum(causal_mask, padding_mask)
        else:
            self_attention_mask = tf.cast(causal_mask, dtype=tf.float32)

        attention_output_1 = self.attention_1(inputs, inputs, inputs, attention_mask=self_attention_mask)
        out_1 = self.layernorm_1(inputs + attention_output_1)

        encoder_mask = getattr(encoder_outputs, '_keras_mask', None)
        cross_attention_mask = tf.cast(encoder_mask[:, tf.newaxis, tf.newaxis, :], dtype=tf.float32) if encoder_mask is not None else None
        attention_output_2 = self.attention_2(out_1, encoder_outputs, encoder_outputs, attention_mask=cross_attention_mask)
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)

    def get_causal_attention_mask(self, inputs: tf.Tensor) -> tf.Tensor:
        batch_size, seq_len = tf.shape(inputs)[0], tf.shape(inputs)[1]
        i = tf.range(seq_len)[:, tf.newaxis]
        j = tf.range(seq_len)
        mask = tf.cast(i >= j, dtype="float32")  # Changed from "int32" to "float32"
        mask = tf.reshape(mask, (1, seq_len, seq_len))
        return tf.tile(mask, [batch_size, 1, 1])

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "latent_dim": self.latent_dim, "num_heads": self.num_heads, "key_dim": self.key_dim})
        return config

# Model Building
def build_transformer_encodec(
    sequence_length: int,
    max_features: int,
    model_dim: int,
    stack_size: int,
    latent_dim: int,
    num_heads: int,
    key_dim: int
) -> tf.keras.Model:
    """Build a Transformer model for text-to-text generation."""
    # Encoder
    encoder_inputs = tf.keras.Input(shape=(None,), dtype="int64", name="encoder_inputs")
    x = PositionalEmbedding(sequence_length, max_features, model_dim)(encoder_inputs)
    for _ in range(stack_size):
        x = TransformerEncoder(num_heads, key_dim, model_dim)(x)  # Removed latent_dim
    encoder_outputs = x

    # Decoder
    decoder_inputs = tf.keras.Input(shape=(None,), dtype="int64", name="decoder_inputs")
    x = PositionalEmbedding(sequence_length + 1, max_features, model_dim)(decoder_inputs)
    for _ in range(stack_size):
        x = TransformerDecoder(model_dim, latent_dim, num_heads, key_dim)(x, encoder_outputs)
    x = tf.keras.layers.Dropout(0.1)(x)
    outputs = tf.keras.layers.Dense(max_features, activation="softmax")(x)

    return tf.keras.Model([encoder_inputs, decoder_inputs], outputs, name="transformer")

# Prediction and Saving
def save_predictions(pairs: List[Tuple], filepath: str, transformer: keras.Model, max_length: int) -> None:
    """Save model predictions to a file."""
    logging.info(f"Generating predictions to {filepath}")
    with open(filepath, 'w') as f:
        f.write("Subj_Pred\tObj\tObj_true\n")
        results = Parallel(n_jobs=1)(delayed(lambda p: '\t'.join([p[0], decode_sequence(p[0], transformer, max_length), p[1]]))(p) for p in pairs)
        f.write('\n'.join(results) + '\n')

# Vectorizer Save and Load
def save_vectorizer(vectorizer: TextVectorization, path: str) -> None:
    """Save a TextVectorization layer as part of a Keras model."""
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(1,), dtype=tf.string),
        vectorizer
    ])
    model.save(path, save_format='tf')

def load_vectorizer(path: str) -> TextVectorization:
    """Load a TextVectorization layer from a saved Keras model."""
    model = tf.keras.models.load_model(path)
    return model.layers[0]

# Main Execution
def main():
    parser = argparse.ArgumentParser(description="Train or evaluate a Transformer model.")
    parser.add_argument("--trainFlag", "-tf", action="store_true", help="Train the model")
    parser.add_argument("--evaluateFlag", "-ev", action="store_true", help="Evaluate the model")
    parser.add_argument("--predictFlag", "-mp", action="store_true", help="Generate predictions")
    parser.add_argument("--seqLen", "-s", type=int, default=30, help="Sequence length")
    parser.add_argument("--nFeatures", "-f", type=int, default=15000, help="Max vocabulary size")
    parser.add_argument("--batchSize", "-b", type=int, default=64, help="Batch size")
    parser.add_argument("--nEpochs", "-e", type=int, default=100, help="Number of epochs")
    parser.add_argument("--stackSize", "-N", type=int, default=1, help="Stack size")
    parser.add_argument("--keyDim", "-kd", type=int, default=0, help="Key dimension (0 for modelDim/nHeads)")
    parser.add_argument("--modelDim", "-md", type=int, default=512, help="Model dimension")
    parser.add_argument("--latentDim", "-l", type=int, default=2048, help="Latent dimension")
    parser.add_argument("--nHeads", "-H", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--nDemo", "-D", type=int, default=-1, help="Number of demo predictions")
    parser.add_argument("--trainData", "-tnD", type=str, default="data/ncd_gp_conceptnet_toy/ncd_gp_conceptnet_train.tsv")
    parser.add_argument("--validData", "-vD", type=str, default="data/ncd_gp_conceptnet_toy/ncd_gp_conceptnet_valid.tsv")
    parser.add_argument("--testData", "-ttD", type=str, default="data/ncd_gp_conceptnet_toy/ncd_gp_conceptnet_valid.tsv")
    parser.add_argument("--datasetName", "-dN", type=str, default="TOY_DATA")
    parser.add_argument("--testName", "-tN", type=str, default="TOY_TEST")

    args = parser.parse_args()
    logging.info("Arguments:\n" + pd.DataFrame([(k, v) for k, v in vars(args).items()]).to_string())

    global batch_size, input_vectorizer, output_vectorizer, out_phr_index_lookup
    batch_size = args.batchSize

    # Hyperparameters
    key_dim = args.modelDim // args.nHeads if args.keyDim <= 0 else args.keyDim
    checkpoint_path = (
        f"results_final{os.sep}{args.datasetName}-transformer_epochs-{args.nEpochs}"
        f"*stackSize-{args.stackSize}*seqlen-{args.seqLen}_maxfeat-{args.nFeatures}"
        f"_batch-{args.batchSize}*keydim-{key_dim}*modeldim-{args.modelDim}"
        f"_latent-{args.latentDim}_heads-{args.nHeads}{os.sep}cp.weights.h5"
    )
    out_dir = os.path.dirname(os.path.dirname(checkpoint_path)) + os.sep
    os.makedirs(out_dir, exist_ok=True)

    vectorizer_dir = os.path.join(out_dir, 'vectorizers')
    os.makedirs(vectorizer_dir, exist_ok=True)

    # Data Preparation
    if args.trainFlag:
        with open(args.trainData) as f:
            train_text = f.readlines()
        with open(args.testData) as f:
            test_text = f.readlines()
        with open(args.validData) as f:
            val_text = f.readlines()

        train_pairs = [prepare_data(line, include_labels=CS_LABELS) for line in train_text]
        test_pairs = [prepare_data(line, include_labels=CS_LABELS, include_pmid=True) for line in test_text]
        val_pairs = [prepare_data(line, include_labels=CS_LABELS, include_pmid=True) for line in val_text]

        train_in_texts = [p[0] for p in train_pairs]
        train_out_texts = [p[1] for p in train_pairs]

        global input_vectorizer, output_vectorizer
        input_vectorizer = TextVectorization(max_tokens=args.nFeatures, output_mode="int", output_sequence_length=args.seqLen, standardize=custom_standardization)
        output_vectorizer = TextVectorization(max_tokens=args.nFeatures, output_mode="int", output_sequence_length=args.seqLen + 1, standardize=custom_standardization)
        input_vectorizer.adapt(train_in_texts)
        output_vectorizer.adapt(train_out_texts)

        # Save vectorizers
        save_vectorizer(input_vectorizer, os.path.join(vectorizer_dir, 'input_vectorizer'))
        save_vectorizer(output_vectorizer, os.path.join(vectorizer_dir, 'output_vectorizer'))
        logging.info(f"Saved vectorizers to {vectorizer_dir}")

        train_ds = make_dataset(train_pairs)
        val_ds = make_dataset(test_pairs, include_pmid=PMID_VAL_LABELS)
        eval_ds = make_dataset(val_pairs, include_pmid=PMID_VAL_LABELS)
    else:
        # Load vectorizers
        input_vectorizer = load_vectorizer(os.path.join(vectorizer_dir, 'input_vectorizer'))
        output_vectorizer = load_vectorizer(os.path.join(vectorizer_dir, 'output_vectorizer'))
        with open(args.validData) as f:
            val_text = f.readlines()
        val_pairs = [prepare_data(line, include_labels=CS_LABELS, include_pmid=True) for line in val_text]
        eval_ds = make_dataset(val_pairs, include_pmid=PMID_VAL_LABELS)

    # Adjust max_features
    max_features = min(args.nFeatures, max(len(input_vectorizer.get_vocabulary()), len(output_vectorizer.get_vocabulary())))
    logging.info(f"Adjusted max_features: {max_features}")

    # Model Setup
    transformer = build_transformer_encodec(args.seqLen, max_features, args.modelDim, args.stackSize, args.latentDim, args.nHeads, key_dim)
    transformer.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"])
    transformer.summary()

    if args.trainFlag:
        callbacks = [
            keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1),
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta=0.005, restore_best_weights=True, verbose=1)
        ]
        history = transformer.fit(train_ds, epochs=args.nEpochs, validation_data=val_ds, callbacks=callbacks)
        pd.DataFrame(history.history).to_csv(os.path.join(out_dir, "history.csv"))
        # Plotting
        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.savefig(os.path.join(out_dir, 'history_plot.pdf'))
    else:
        transformer.load_weights(checkpoint_path)

    # Prediction and Evaluation
    global out_phr_index_lookup
    out_phr_index_lookup = dict(enumerate(output_vectorizer.get_vocabulary()))
    max_decoded_length = args.seqLen + 1

    if args.predictFlag or args.evaluateFlag:
        if args.predictFlag:
            predict_pairs = val_pairs[:args.nDemo] if args.nDemo >= 0 else val_pairs
            random.shuffle(predict_pairs)
            save_predictions(predict_pairs, os.path.join(out_dir, f"{args.testName}_predictions.tsv"), transformer, max_decoded_length)

        if args.evaluateFlag:
            val_loss, val_acc = transformer.evaluate(eval_ds)
            with open(os.path.join(out_dir, 'evaluation.txt'), 'w') as f:
                f.write(f"Validation_loss: {val_loss}\nValidation_acc: {val_acc}\n")

    logging.info(f"Tasks completed. Results in {out_dir}")

if __name__ == "__main__":
    main()
