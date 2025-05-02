#!/usr/bin/env python3
"""
Attention-GRU Translator

Train a sequence-to-sequence model with stacked GRU layers and Bahdanau attention.
Supports variable number of layers and dropout.
"""

import argparse
import logging
import math
import os
import random
import re
import string
from typing import Any, List, NamedTuple, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ----------------------------------------------------------------------------
# Configuration and Logging
# ----------------------------------------------------------------------------

def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------

STRIP_CHARS = string.punctuation.replace("[", "").replace("]", "")

@tf.keras.utils.register_keras_serializable()
def custom_standardization(input_str: tf.Tensor) -> tf.Tensor:
    """Lowercase and strip punctuation."""
    lower = tf.strings.lower(input_str)
    return tf.strings.regex_replace(lower, f"[{re.escape(STRIP_CHARS)}]", "")


def parse_dataset_name(path: str) -> str:
    """Extracts a short name (e.g. ncd-gp) from a file path."""
    base = os.path.basename(path).lower()
    flags = {"ncd": "ncd" in base,
             "gp": "gp" in base,
             "cn": "conceptnet" in base}
    return "-".join([k for k, ok in flags.items() if ok])


def prepare_data(
    line: str,
    start_token: str = "[start] ",
    end_token: str = " [end]",
    pmid: bool = True,
    include_labels: bool = False,
    include_sent: bool = False,
    all_start_end: bool = False
) -> Tuple[str, Any]:
    """
    Parse a TSV line into input and (optionally) label.
    Returns (input_text, output_text_or_label).
    """
    cols = line.rstrip("\n").split("\t")
    if pmid:
        cols.pop(0)

    # Normalize predicate
    pred = " ".join(re.findall(r"[A-Z][a-z]*", cols[1])).lower() or cols[1]

    # Handle non-numeric label cleanup
    if not cols[4].strip().isdigit():
        i = 4
        extra = []
        while not cols[i].isdigit():
            extra.append(cols.pop(i))
        cols[3] = " ".join([cols[3]] + extra)

    raw = [cols[0], pred, cols[2], start_token + cols[3] + end_token]
    label = float(cols[4]) if include_labels else None

    if include_labels:
        out = (raw[-1], label)
    else:
        out = raw[-1]

    if include_sent:
        inp = f"{cols[0]} {cols[2]} {pred}"
    else:
        inp = f"{pred} {cols[0]}"
        if all_start_end:
            inp = f"{start_token}{inp}{end_token}"

    return inp, out

# ----------------------------------------------------------------------------
# Dataset Helpers
# ----------------------------------------------------------------------------

def format_pair(inp: tf.Tensor, out: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    return input_vectorizer(inp), output_vectorizer(out)


def make_dataset(pairs: List[Tuple[str, str]], batch_size: int) -> tf.data.Dataset:
    inputs, outputs = zip(*pairs)
    ds = tf.data.Dataset.from_tensor_slices((list(inputs), list(outputs)))
    ds = ds.batch(batch_size).map(format_pair)
    return ds.shuffle(2048).prefetch(tf.data.AUTOTUNE).cache()

# ----------------------------------------------------------------------------
# Shape Checker (for debugging)
# ----------------------------------------------------------------------------

class ShapeChecker:
    """Ensures tensor shapes match expected named dimensions."""
    def __init__(self):
        self.shapes = {}

    def __call__(
        self,
        tensor: tf.Tensor,
        names: Tuple[Any, ...],
        broadcast: bool = False
    ) -> None:
        if not tf.executing_eagerly():
            return
        names = (names,) if isinstance(names, str) else names
        shape = tf.shape(tensor)
        if tf.rank(tensor) != len(names):
            raise ValueError(f"Rank mismatch: found {tf.rank(tensor)}, expected {len(names)}")
        for idx, name in enumerate(names):
            dim = shape[idx]
            prev = self.shapes.get(name)
            if prev is None:
                self.shapes[name] = dim
            elif dim != prev and not (broadcast and dim == 1):
                raise ValueError(f"Dimension '{name}' mismatch: {dim} vs {prev}")

# ----------------------------------------------------------------------------
# NamedTuples for Decoder I/O
# ----------------------------------------------------------------------------

class DecoderInput(NamedTuple):
    new_tokens: tf.Tensor
    enc_output: tf.Tensor
    mask: tf.Tensor

class DecoderOutput(NamedTuple):
    logits: tf.Tensor
    attention_weights: tf.Tensor

# ----------------------------------------------------------------------------
# Model Components
# ----------------------------------------------------------------------------

class BahdanauAttention(layers.Layer):
    """Implements Bahdanau (additive) attention."""
    def __init__(self, units: int):
        super().__init__()
        self.W1 = layers.Dense(units, use_bias=False)
        self.W2 = layers.Dense(units, use_bias=False)
        self.attn = layers.AdditiveAttention()

    def call(
        self,
        query: tf.Tensor,
        value: tf.Tensor,
        mask: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        # Linear projections
        q = self.W1(query)
        k = self.W2(value)
        # Compute context and attention scores
        context, scores = self.attn(
            inputs=[q, value, k],
            mask=[tf.ones_like(q[..., 0], dtype=bool), mask],
            return_attention_scores=True
        )
        return context, scores

class Encoder(layers.Layer):
    """Stacked GRU encoder."""
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        enc_units: int,
        num_layers: int = 1,
        dropout: float = 0.0
    ):
        super().__init__()
        self.embedding = layers.Embedding(vocab_size, embed_dim)
        cells = [
            layers.GRUCell(enc_units, dropout=dropout, recurrent_dropout=dropout)
            for _ in range(num_layers)
        ]
        self.rnn = layers.RNN(cells, return_sequences=True, return_state=True)
        self._num_layers = num_layers

    def call(
        self,
        tokens: tf.Tensor,
        state: Any = None
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.embedding(tokens)
        outputs = self.rnn(x, initial_state=state)
        seq, states = outputs[0], outputs[1:]
        return seq, states[-1]

class Decoder(layers.Layer):
    """Stacked GRU decoder with attention."""
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        dec_units: int,
        num_layers: int = 1,
        dropout: float = 0.0
    ):
        super().__init__()
        self.embedding = layers.Embedding(vocab_size, embed_dim)
        cells = [
            layers.GRUCell(dec_units, dropout=dropout, recurrent_dropout=dropout)
            for _ in range(num_layers)
        ]
        self.rnn = layers.RNN(cells, return_sequences=True, return_state=True)
        self.attention = BahdanauAttention(dec_units)
        self.fc = layers.Dense(vocab_size)
        self._num_layers = num_layers

    def call(
        self,
        inp: DecoderInput,
        state: Any = None
    ) -> Tuple[DecoderOutput, tf.Tensor]:
        # Ensure state list matches number of layers
        if state is not None and not isinstance(state, (list, tuple)):
            state = [state] * self._num_layers
        x = self.embedding(inp.new_tokens)
        outputs = self.rnn(x, initial_state=state)
        seq, states = outputs[0], outputs[1:]
        context, attn_w = self.attention(seq, inp.enc_output, inp.mask)
        concat = tf.concat([context, seq], axis=-1)
        logits = self.fc(concat)
        return DecoderOutput(logits, attn_w), states[-1]

# ----------------------------------------------------------------------------
# Loss and Metrics
# ----------------------------------------------------------------------------

class MaskedLoss(keras.losses.Loss):
    """Sparse categorical crossentropy ignoring padding tokens."""
    def __init__(self):
        super().__init__(reduction='none')
        self.scce = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        mask = tf.cast(y_true != 0, tf.float32)
        loss = self.scce(y_true, y_pred) * mask
        return tf.reduce_sum(loss)

# ----------------------------------------------------------------------------
# Training Model
# ----------------------------------------------------------------------------

class TrainTranslator(keras.Model):
    """Wraps encoder, decoder, and custom training loops."""
    def __init__(
        self,
        embed_dim: int,
        units: int,
        in_processor,
        out_processor,
        num_layers: int,
        dropout: float,
        use_tf: bool = True
    ):
        super().__init__()
        self.encoder = Encoder(in_processor.vocabulary_size(), embed_dim, units, num_layers, dropout)
        self.decoder = Decoder(out_processor.vocabulary_size(), embed_dim, units, num_layers, dropout)
        self.in_proc = in_processor
        self.out_proc = out_processor
        self.use_tf = use_tf
        self.loss_fn = MaskedLoss()
        self.train_acc = keras.metrics.SparseCategoricalAccuracy()
        self.val_acc = keras.metrics.SparseCategoricalAccuracy()

    def compile(self, optimizer: keras.optimizers.Optimizer, **kwargs):
        super().compile(**kwargs)
        self.optimizer = optimizer

    def _step(self, batch, training: bool):
        inp, targ = batch
        inp_toks = self.in_proc(inp)
        targ_toks = self.out_proc(targ)
        mask = targ_toks != 0
        with tf.GradientTape() as tape:
            enc_seq, enc_state = self.encoder(inp_toks)
            state = enc_state
            loss = 0.0
            for t in range(targ_toks.shape[1] - 1):
                new_tokens = targ_toks[:, t:t+2]
                y_true = new_tokens[:, 1:2]
                dec_in = DecoderInput(new_tokens=new_tokens[:, :1], enc_output=enc_seq, mask=inp_toks!=0)
                dec_out, state = self.decoder(dec_in, state)
                loss += self.loss_fn(y_true, dec_out.logits)
                if training:
                    self.train_acc.update_state(y_true, dec_out.logits)
                else:
                    self.val_acc.update_state(y_true, dec_out.logits)
            loss /= tf.reduce_sum(tf.cast(mask, tf.float32))
        if training:
            grads = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss

    def train_step(self, data):
        loss = self._step(data, training=True)
        return {"loss": loss, "accuracy": self.train_acc.result()}

    def test_step(self, data):
        loss = self._step(data, training=False)
        return {"val_loss": loss, "val_accuracy": self.val_acc.result()}

# ----------------------------------------------------------------------------
# Inference Translator
# ----------------------------------------------------------------------------

class Translator(tf.Module):
    def __init__(self, encoder, decoder, in_proc, out_proc):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.in_proc = in_proc
        self.out_lookup = layers.experimental.preprocessing.StringLookup(
            vocabulary=out_proc.get_vocabulary(), mask_token='', invert=True)
        mask_ids = layers.experimental.preprocessing.StringLookup(
            vocabulary=out_proc.get_vocabulary(), mask_token='')( ['','[UNK]','[start]']).numpy()
        vocab_size = self.out_lookup.vocabulary_size()
        mask = np.zeros(vocab_size, bool)
        mask[mask_ids] = True
        self.mask = tf.constant(mask)
        self.start = self.out_lookup('[start]')
        self.end = self.out_lookup('[end]')

    def translate(self, inputs: List[str], max_len: int = 50, temp: float = 1.0) -> List[str]:
        batch = tf.constant(inputs)
        toks = self.in_proc(batch)
        enc_seq, enc_state = self.encoder(toks)
        state = enc_state
        token = tf.fill([tf.shape(batch)[0], 1], self.start)
        results = []
        for _ in range(max_len):
            dec_in = DecoderInput(new_tokens=token, enc_output=enc_seq, mask=toks!=0)
            dec_out, state = self.decoder(dec_in, state)
            logits = tf.where(self.mask, -np.inf, dec_out.logits)
            if temp == 0.0:
                token = tf.argmax(logits, axis=-1)
            else:
                token = tf.random.categorical(tf.squeeze(logits, 1)/temp, 1)
            results.append(token)
            if tf.reduce_all(token == self.end): break
        seq = tf.concat(results, axis=1)
        texts = self.out_lookup(seq)
        return tf.strings.reduce_join(texts, separator=' ', axis=1).numpy().tolist()

# ----------------------------------------------------------------------------
# Training and Evaluation Pipeline
# ----------------------------------------------------------------------------

def train_and_evaluate(args):
    # Load and prepare data
    with open(args.trainData) as f: train_lines = f.readlines()
    with open(args.testData)  as f: test_lines  = f.readlines()
    train_pairs = [prepare_data(l, include_labels=False, all_start_end=True) for l in train_lines]
    test_pairs  = [prepare_data(l, include_labels=False, all_start_end=True) for l in test_lines]

    train_ds = make_dataset(train_pairs, args.batchSize)
    val_ds   = make_dataset(test_pairs,  args.batchSize)

    # Vectorizers
    global input_vectorizer, output_vectorizer
    input_vectorizer = layers.TextVectorization(
        max_tokens=args.nFeatures,
        output_sequence_length=args.seqLen,
        standardize=custom_standardization
    )
    output_vectorizer = layers.TextVectorization(
        max_tokens=args.nFeatures,
        output_sequence_length=args.seqLen + 1,
        standardize=custom_standardization
    )
    input_vectorizer.adapt([p[0] for p in train_pairs])
    output_vectorizer.adapt([p[1] for p in train_pairs])

    # Model
    model = TrainTranslator(
        embed_dim=args.embeddingDim,
        units=args.nSteps,
        in_processor=input_vectorizer,
        out_processor=output_vectorizer,
        num_layers=args.numLayers,
        dropout=args.dropout
    )
    model.compile(optimizer=tf.optimizers.Adam())

    # Callbacks
    ckpt = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(args.resPath, "ckpt_{epoch}"),
        save_weights_only=True, verbose=1
    )

    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.nEpochs,
        callbacks=[ckpt]
    )

    # Save history
    pd.DataFrame(history.history).to_csv(os.path.join(args.resPath, "history.csv"))

    # Inference example
    translator = Translator(model.encoder, model.decoder, input_vectorizer, output_vectorizer)
    sample_inputs = [p[0] for p in random.sample(test_pairs, min(5, len(test_pairs)))]
    logging.info("Sample translations: %s", translator.translate(sample_inputs))

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    configure_logging()
    parser = argparse.ArgumentParser(description="Train Attention-GRU translator.")
    parser.add_argument("-s", "--seqLen",     type=int, default=50)
    parser.add_argument("-u", "--nSteps",     type=int, default=1024)
    parser.add_argument("-f", "--nFeatures",  type=int, default=15000)
    parser.add_argument("-b", "--batchSize",  type=int, default=64)
    parser.add_argument("-e", "--nEpochs",    type=int, default=40)
    parser.add_argument("-d", "--embeddingDim",type=int, default=1024)
    parser.add_argument("-l", "--numLayers",   type=int, default=1)
    parser.add_argument("--dropout",            type=float, default=0.0)
    parser.add_argument("-T", "--trainData",  type=str, required=True)
    parser.add_argument("-t", "--testData",   type=str, required=True)
    parser.add_argument("-rp","--resPath",    type=str, default=os.getcwd())
    args = parser.parse_args()
    train_and_evaluate(args)

if __name__ == "__main__":
    main()
