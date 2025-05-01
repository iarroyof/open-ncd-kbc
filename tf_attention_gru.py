import numpy as np
import pandas as pd
import typing
from typing import Any, Tuple
import tensorflow as tf
from keras.layers import TextVectorization
from keras import layers
import keras
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import functools, re, string, os, time, random
import argparse
from pdb import set_trace as st
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p')

cs_labels = False

def prepare_data(
        line, start_token='[start] ', end_token=' [end]', pmid=True,
        include_labels=False, include_sent=False, all_start_end=False):
    line = line.split('\t')

    if pmid:
        line.pop(0)

    pred = ' '.join(re.findall('[A-Z][a-z]*', line[1])).lower()
    if pred.isspace() or not pred:
        pred = line[1]

    if not line[4].strip().isdigit():
        if not re.match(r'^-?\d+(?:\.\d+)$', line[4].strip()):
            i = 4
            complements = []
            # Check index bounds when popping
            while i < len(line) and not line[i].strip().isdigit():
                complements.append(line[i])
                line.pop(i) # Pop from current index

            # Handle the case where line[3] might be the last element after popping
            if 3 < len(line):
                 line[3] = " ".join([line[3]] + complements)
            elif complements: # If line[3] was popped, just use complements
                 line[3] = " ".join(complements)
            # If both were empty, line[3] remains whatever it was (possibly empty string)


    # Ensure there are enough elements in line before accessing indices
    if len(line) < 5:
         logging.warning(f"Skipping malformed line: {line}")
         # Return None or raise an exception to handle malformed data
         return None

    sample = [line[0], pred, line[2],
        start_token + line[3] + end_token, float(line[4].strip())]

    # Ensure sample has expected elements before deleting
    if not include_labels:
        if len(sample) > 4: # Check if label exists before deleting
            del sample[-1]
        sample_o = sample[-1]
    else:
        if len(sample) > 3: # Ensure enough elements for tuple
            sample_o = tuple(sample[-2:])
        else:
            logging.warning(f"Skipping malformed line for cs_labels: {line}")
            return None


    if not include_sent:
        if len(sample) > 1: # Ensure sample[1] and sample[0] exist
            sample_i = ' '.join([sample[1], sample[0]])
            if all_start_end:
                sample_i = start_token + sample_i + end_token
        else:
            logging.warning(f"Skipping malformed line for include_sent=False: {line}")
            return None
    else:
        if len(sample) > 2: # Ensure sample[0], sample[2], sample[1] exist
            sample_i = ' '.join([sample[0], sample[2], sample[1]])
        else:
            logging.warning(f"Skipping malformed line for include_sent=True: {line}")
            return None


    return sample_i, sample_o

@keras.utils.register_keras_serializable()
def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(
        lowercase, "[%s]" % re.escape(strip_chars), "")

def format_dataset(in_phr, out_phr):
    # This function takes STRING tensors and vectorizes them into INT tensors
    # It should only be called once in the dataset pipeline's .map()
    in_phr = input_vectorizer(in_phr)
    out_phr = output_vectorizer(out_phr)
    return in_phr, out_phr

def make_dataset(pairs):
    # Filter out None values resulting from malformed lines
    pairs = [pair for pair in pairs if pair is not None]

    in_phr_texts, out_phr_texts = zip(*pairs)
    in_phr_texts = list(in_phr_texts)
    out_phr_texts = list(out_phr_texts)
    dataset = tf.data.Dataset.from_tensor_slices(
        (in_phr_texts, out_phr_texts))
    dataset = dataset.batch(batch_size)
    # Map applies format_dataset to each batch of strings
    dataset = dataset.map(format_dataset)
    # Cache after map to store the vectorized integer tensors
    # Prefetch after cache for performance
    return dataset.shuffle(2048).cache().prefetch(tf.data.AUTOTUNE)


def sort_cols(columns):
    ends = np.unique([c[-2:] for c in columns])
    new_cols = []
    for e in ends:
        for c in columns:
            if c.endswith(e):
                new_cols.append(c)
    return new_cols

class ShapeChecker:
    def __init__(self):
        self.shapes = {}

    def __call__(self, tensor, names, broadcast=False):
        # Only perform checks in eager mode
        if not tf.executing_eagerly():
            return

        if isinstance(names, str):
            names = (names,)

        shape = tf.shape(tensor)
        rank = tf.rank(tensor)

        if rank != len(names):
            raise ValueError(f'Rank mismatch:\n'
                             f'    found {rank}: {shape.numpy()}\n'
                             f'    expected {len(names)}: {names}\n')

        for i, name in enumerate(names):
            if isinstance(name, int):
                old_dim = name
            else:
                old_dim = self.shapes.get(name, None)
            new_dim = shape[i]

            # Allow broadcasting dimension 1 if specified
            if broadcast and new_dim == 1:
                continue

            if old_dim is None:
                self.shapes[name] = new_dim
                continue

            if new_dim != old_dim:
                raise ValueError(f"Shape mismatch for dimension: '{name}'\n"
                                 f"    found: {new_dim}\n"
                                 f"    expected: {old_dim}\n")

class Encoder(keras.layers.Layer):
    def __init__(self, input_vocab_size, embedding_dim, enc_units):
        super(Encoder, self).__init__()
        self.enc_units = enc_units
        self.input_vocab_size = input_vocab_size
        self.embedding = keras.layers.Embedding(self.input_vocab_size, embedding_dim)
        self.gru = keras.layers.GRU(self.enc_units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')

    def call(self, tokens, state=None):
        # Removed ShapeChecker calls inside layer methods
        # shape_checker = ShapeChecker()
        # shape_checker(tokens, ('batch', 's'))
        vectors = self.embedding(tokens)
        # shape_checker(vectors, ('batch', 's', 'embed_dim'))
        output, state = self.gru(vectors, initial_state=state)
        # shape_checker(output, ('batch', 's', 'enc_units'))
        # shape_checker(state, ('batch', 'enc_units'))
        return output, state

class BahdanauAttention(keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        # Keras’s AdditiveAttention will project query/value for us:
        self.attention = keras.layers.AdditiveAttention(use_scale=False)

    def call(self, query, value, mask):
        # query: (batch, 1, units)
        # value: (batch, S, units)
        # mask:  (batch, S) boolean
        context_vector, attention_weights = self.attention(
            [query, value],
            mask=[None, mask],                # only mask the `value` axis
            return_attention_scores=True
        )
        # context_vector: (batch, 1, units)
        # attention_weights: (batch, 1, S)
        return context_vector, attention_weights

class Decoder(keras.layers.Layer):
    def __init__(self, output_vocab_size, embedding_dim, dec_units):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.output_vocab_size = output_vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = keras.layers.Embedding(self.output_vocab_size, embedding_dim)
        self.gru = keras.layers.GRU(self.dec_units,
                                    return_sequences=True, # Keep True for attention
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')
        self.attention = BahdanauAttention(self.dec_units)
        self.Wc = keras.layers.Dense(dec_units, activation=tf.math.tanh, use_bias=False) # Typically Wc is used for the final context vector
        self.fc = keras.layers.Dense(self.output_vocab_size) # Final dense layer for logits

    # Nested classes remain indented inside the class
    class DecoderInput(typing.NamedTuple):
        new_tokens: Any # Shape: (batch, t) - t=1 during step-by-step decoding
        enc_output: Any # Shape: (batch, s, enc_units)
        mask: Any       # Shape: (batch, s) - boolean mask for enc_output padding


    class DecoderOutput(typing.NamedTuple):
        logits: Any            # Shape: (batch, t, output_vocab_size)
        attention_weights: Any # Shape: (batch, t, s)

    def call(self,
             inputs: 'Decoder.DecoderInput', # Use forward reference for type hint
             state=None) -> Tuple['Decoder.DecoderOutput', tf.Tensor]: # Use forward reference for type hint
        # Removed ShapeChecker calls inside layer methods
        # shape_checker = ShapeChecker()
        # shape_checker(inputs.new_tokens, ('batch', 't')) # inputs.new_tokens has shape (batch, 1)
        # shape_checker(inputs.enc_output, ('batch', 's', 'enc_units'))
        # shape_checker(inputs.mask, ('batch', 's'))

        # if state is not None:
        #     shape_checker(state, ('batch', 'dec_units'))

        # 1. Embed the new token
        vectors = self.embedding(inputs.new_tokens) # Shape: (batch, 1, embedding_dim)
        # shape_checker(vectors, ('batch', 't', 'embedding_dim'))

        # 2. Run the GRU on the embedded token
        # GRU takes (batch, seq_len, features) and initial state (batch, units)
        # returns output (batch, seq_len, units) and state (batch, units)
        # Since we process one token at a time (seq_len=1), rnn_output is (batch, 1, dec_units)
        rnn_output, state = self.gru(vectors, initial_state=state)
        # shape_checker(rnn_output, ('batch', 't', 'dec_units')) # t=1 here
        # shape_checker(state, ('batch', 'dec_units')) # GRU state

        # 3. Calculate attention weights and context vector
        # query is the current decoder GRU output (batch, 1, dec_units)
        # value is the encoder output (batch, s, enc_units)
        # mask is the input sequence mask (batch, s)
        context_vector, attention_weights = self.attention(
            query=rnn_output, value=inputs.enc_output, mask=inputs.mask)

        # Expected shapes from BahdanauAttention call:
        # context_vector: (batch, 1, enc_units) - Note: value_units from Attention = enc_units
        # attention_weights: (batch, 1, s)

        # shape_checker(context_vector, ('batch', 't', 'dec_units')) # This check might have been problematic, should be ('batch', 't', 'enc_units') based on context_vector shape, but t=1. Let's use the actual shape.
        # Let's check the expected shape based on AdditiveAttention docs again:
        # context_vector shape: (batch, query_seq_len, value_depth)
        # query_seq_len = 1, value_depth = enc_units
        # So context_vector shape is (batch, 1, enc_units)

        # Let's fix the shape check name or remove it if ShapeChecker is the issue
        # shape_checker(context_vector, ('batch', 't', 'enc_units')) # Corrected shape check name

        # shape_checker(attention_weights, ('batch', 't', 's')) # t=1

        # 4. Combine context vector and GRU output
        # Concatenate along the last dimension: (batch, 1, enc_units) + (batch, 1, dec_units)
        # Result shape: (batch, 1, enc_units + dec_units)
        context_and_rnn_output = tf.concat([context_vector, rnn_output], axis=-1)

        # 5. Pass through dense layer to get final attention-aware vector
        # This layer maps the concatenated vector to dec_units or similar size.
        # The Wc layer in Bahdanau is typically applied to the final combined vector.
        attention_vector = self.Wc(context_and_rnn_output) # Shape: (batch, 1, dec_units) ? Check Wc output size

        # Let's check Wc definition: keras.layers.Dense(dec_units, activation=tf.math.tanh, use_bias=False)
        # Yes, Wc maps the last dimension to dec_units. Input shape (batch, 1, enc_units + dec_units) -> Output shape (batch, 1, dec_units)

        # shape_checker(attention_vector, ('batch', 't', 'dec_units')) # t=1

        # 6. Final dense layer to predict logits over vocabulary
        # Input shape (batch, 1, dec_units) -> Output shape (batch, 1, output_vocab_size)
        logits = self.fc(attention_vector)
        # shape_checker(logits, ('batch', 't', 'output_vocab_size')) # t=1

        # Return logits and attention weights
        return self.DecoderOutput(logits, attention_weights), state


class MaskedLoss(keras.losses.Loss):
    def __init__(self):
        self.name = 'masked_loss'
        self.loss = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none') # Keep reduction='none' to apply mask

    def __call__(self, y_true, y_pred):
        # Removed ShapeChecker calls inside layer methods
        # shape_checker = ShapeChecker()
        # y_true shape: (batch, t) - t=1 in train_step
        # y_pred shape: (batch, t, logits) - t=1 in train_step
        # shape_checker(y_true, ('batch', 't'))
        # shape_checker(y_pred, ('batch', 't', 'logits'))

        loss = self.loss(y_true, y_pred) # loss shape: (batch, t) after reduction='none'
        # shape_checker(loss, ('batch', 't'))

        # Mask out loss for padding tokens (token 0)
        mask = tf.cast(y_true != 0, tf.float32) # mask shape: (batch, t)
        # shape_checker(mask, ('batch', 't'))

        loss *= mask # loss is now (batch, t), masked

        # Return the sum over the batch and time steps
        return tf.reduce_sum(loss)


class TrainTranslator(keras.Model):
    def __init__(self, embedding_dim, units,
                 input_text_processor,
                 output_text_processor,
                 use_tf_function=True):
        super().__init__()
        encoder = Encoder(input_text_processor.vocabulary_size(),
                          embedding_dim, units)
        decoder = Decoder(output_text_processor.vocabulary_size(),
                          embedding_dim, units)
        self.encoder = encoder
        self.decoder = decoder
        self.input_text_processor = input_text_processor
        self.output_text_processor = output_text_processor
        self.use_tf_function = use_tf_function
        self.shape_checker = ShapeChecker() # Keep the shape checker for preprocess/debugging outer logic

        # Keras metrics - these reset automatically at the start of each epoch
        self.train_loss_tracker = keras.metrics.Mean(name="loss")
        self.train_accuracy_tracker = keras.metrics.SparseCategoricalAccuracy(name="accuracy")
        self.test_loss_tracker = keras.metrics.Mean(name="loss") # For test_step if implemented manually
        self.test_accuracy_tracker = keras.metrics.SparseCategoricalAccuracy(name="accuracy") # For test_step

    @property
    def metrics(self):
        # List the metrics that Keras should track during .fit()
        # These should correspond to the metrics updated in train_step and test_step
        return [self.train_loss_tracker, self.train_accuracy_tracker,
                self.test_loss_tracker, self.test_accuracy_tracker]


    # The main call method, used by Keras for model building and functional API
    # It should handle the expected input type of the model (strings)
    # and typically performs a single forward pass or defines the structure.
    # The actual training loop is in train_step.
    def call(self, inputs, training=False):
        # Expect inputs as a tuple of string tensors (input_text, target_text)
        # The input signature of train_step handles this explicitly for the training loop.
        # For model building, Keras might pass raw data or dummy data.
        # Ensure preprocessing can handle string inputs.

        if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
            input_text, target_text = inputs
        elif isinstance(inputs, tf.Tensor) and inputs.dtype == tf.string:
             # If only input text is passed (e.g., during prediction or initial build)
             input_text = inputs
             # Create a dummy target_text with appropriate dtype (string)
             # Its content doesn't matter for shape inference
             batch_size = tf.shape(input_text)[0]
             dummy_target_text = tf.fill([batch_size], "")
             target_text = dummy_target_text
        else:
             raise ValueError(f"Unexpected input type to TrainTranslator.call: {type(inputs)}")

        # Preprocess string inputs to get tokenized integers and masks
        # self.shape_checker is used inside _preprocess
        input_tokens, input_mask, target_tokens, target_mask = self._preprocess(input_text, target_text)

        # Perform a single forward pass relevant for Keras model building
        # This involves encoding the input and performing one step of decoding
        # using the first token of the target sequence as input to the decoder.

        # Encoder forward pass
        enc_output, enc_state = self.encoder(input_tokens) # int64 -> float

        dec_state = enc_state # Initialize decoder state with encoder final state

        # Use the first token of the target sequence as the initial decoder input
        # This assumes target_tokens includes a start token at index 0
        initial_dec_token = target_tokens[:, :1] # Shape (batch, 1), int64

        # Prepare decoder input object
        decoder_input = Decoder.DecoderInput( # Use the nested class reference
            new_tokens=initial_dec_token, # Shape (batch, 1)
            enc_output=enc_output,      # Shape (batch, s, enc_units)
            mask=input_mask             # Shape (batch, s) - bool
        )

        # Perform one decoding step
        dec_result, _ = self.decoder(decoder_input, state=dec_state)
        # dec_result.logits shape: (batch, 1, output_vocab_size)

        # The model's output for Keras should be the decoder's logits
        # This is typically the output logits for the *next* token prediction.
        # Since we did one step with the start token, the logits predict the first actual target token.
        return dec_result.logits # Shape (batch, 1, output_vocab_size)


    def _preprocess(self, input_text, target_text):
        # This takes string tensors and returns integer tensors + masks
        # Use the model's shape checker here if desired
        self.shape_checker(input_text, ('batch',))
        self.shape_checker(target_text, ('batch',))

        # Apply TextVectorization layers
        input_tokens = self.input_text_processor(input_text)
        target_tokens = self.output_text_processor(target_text)

        # TextVectorization returns int64 by default if output_mode="int"
        self.shape_checker(input_tokens, ('batch', 's'))
        self.shape_checker(target_tokens, ('batch', 't_target')) # Use 't_target' for target sequence length

        # Create masks (True for non-padding tokens, False for padding token 0)
        input_mask = input_tokens != 0 # Boolean mask for input sequence padding
        target_mask = target_tokens != 0 # Boolean mask for target sequence padding

        self.shape_checker(input_mask, ('batch', 's'))
        self.shape_checker(target_mask, ('batch', 't_target'))

        return input_tokens, input_mask, target_tokens, target_mask

    # Use @tf.function here for efficient training loop
    # Specify input_signature to compile the function graph
    #@tf.function(input_signature=[[
    #    tf.TensorSpec(dtype=tf.string, shape=[None]),
    #    tf.TensorSpec(dtype=tf.string, shape=[None])
    #]])
    def train_step(self, inputs):
        # inputs will be a tuple of string tensors (input_text, target_text) due to input_signature
        input_text, target_text = inputs

        # Preprocess strings to get tokenized integers and masks
        # _preprocess uses the model's shape_checker if enabled
        (input_tokens, input_mask,
         target_tokens, target_mask) = self._preprocess(input_text, target_text)

        # Ensure target_tokens are int64 for loss calculation (usually already is)
        target_tokens = tf.cast(target_tokens, tf.int64)

        max_target_length = tf.shape(target_tokens)[1] # Dynamic target sequence length

        with tf.GradientTape() as tape:
            # Encoder forward pass
            enc_output, enc_state = self.encoder(input_tokens) # Input tokens are int64

            dec_state = enc_state # Initialize decoder state

            loss = tf.constant(0.0, dtype=tf.float32)
            total_tokens = tf.constant(0.0, dtype=tf.float32) # Counter for non-padding tokens

            # Decoder teacher forcing loop
            # Iterate from the first token of the target (usually [start]) up to the second-to-last token
            # because each step predicts the *next* token.
            # target_tokens[:, t:t+1] is the input token at time t
            # target_tokens[:, t+1:t+2] is the true next token at time t+1
            for t in tf.range(max_target_length - 1):
                # Use the current target token `target_tokens[:, t:t+1]` as the decoder input
                # Shape (batch, 1), int64
                new_tokens = target_tokens[:, t:t+1]

                # The target for the loss is the *next* token `target_tokens[:, t+1:t+2]`
                # Shape (batch, 1), int64
                y_true = target_tokens[:, t+1:t+2]

                # Prepare decoder input object for this single step
                decoder_input = Decoder.DecoderInput(
                    new_tokens=new_tokens, # Shape (batch, 1)
                    enc_output=enc_output, # Shape (batch, s, enc_units)
                    mask=input_mask        # Shape (batch, s)
                )

                # Perform one decoding step
                dec_result, dec_state = self.decoder(decoder_input, state=dec_state)
                # dec_result.logits shape: (batch, 1, output_vocab_size)

                # Calculate loss for this time step
                # self.loss is MaskedLoss, which takes (batch, t) and (batch, t, logits)
                # Here t=1 for both, so it works correctly
                step_loss = self.loss(y_true, dec_result.logits) # Should return (batch, 1) masked sum

                # Accumulate loss and token count across time steps in the batch
                # The MaskedLoss already returned the summed loss over batch and time=1, masked.
                loss += step_loss # Add the sum from this timestep

                # We need to count tokens across the time steps for the average loss and accuracy
                # The mask from the current y_true gives us which tokens are real
                step_mask = tf.cast(y_true != 0, tf.float32) # (batch, 1)
                total_tokens += tf.reduce_sum(step_mask) # Sum non-padding tokens for this timestep over the batch


                # Update training accuracy metric
                # accuracy.update_state expects (y_true, y_pred_logits)
                # y_true is (batch, 1), y_pred_logits is (batch, 1, vocab_size)
                self.train_accuracy_tracker.update_state(y_true, dec_result.logits)


            # Compute the average loss over the entire batch and all unmasked time steps
            # Avoid division by zero if a batch only contains padding (unlikely with start token, but safe)
            average_loss = tf.cond(total_tokens > 0,
                                   lambda: loss / total_tokens, # Divide total summed loss by total non-padding tokens
                                   lambda: tf.constant(0.0, dtype=tf.float32))

        # Get gradients and apply optimizer
        variables = self.trainable_variables
        gradients = tape.gradient(average_loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        # Update loss metric with the calculated average batch loss
        self.train_loss_tracker.update_state(average_loss)

        # Return metric results for this step
        return {'loss': self.train_loss_tracker.result(),
                'accuracy': self.train_accuracy_tracker.result()}


    # Use @tf.function here for efficient evaluation loop
    # Specify input_signature to compile the function graph
    @tf.function(input_signature=[[
        tf.TensorSpec(dtype=tf.string, shape=[None]),
        tf.TensorSpec(dtype=tf.string, shape=[None])
    ]])
    def test_step(self, inputs):
         # inputs will be a tuple of string tensors (input_text, target_text)
        input_text, target_text = inputs

        # Preprocess strings to get tokenized integers and masks
        (input_tokens, input_mask,
         target_tokens, target_mask) = self._preprocess(input_text, target_text)

        target_tokens = tf.cast(target_tokens, tf.int64)
        max_target_length = tf.shape(target_tokens)[1]

        enc_output, enc_state = self.encoder(input_tokens)
        dec_state = enc_state

        loss = tf.constant(0.0, dtype=tf.float32)
        total_tokens = tf.constant(0.0, dtype=tf.float32)

        # Decoder teacher forcing loop for evaluation
        # Iterate from the first token of the target (usually [start]) up to the second-to-last token
        for t in tf.range(max_target_length - 1):
            new_tokens = target_tokens[:, t:t+1] # Input token at time t
            y_true = target_tokens[:, t+1:t+2] # True token at time t+1

            decoder_input = Decoder.DecoderInput( # Use the nested class reference
                new_tokens=new_tokens, # Shape (batch, 1)
                enc_output=enc_output, # Shape (batch, s, enc_units)
                mask=input_mask        # Shape (batch, s)
            )

            dec_result, dec_state = self.decoder(decoder_input, state=dec_state)
            # dec_result.logits shape: (batch, 1, output_vocab_size)

            # Calculate loss for this time step
            step_loss = self.loss(y_true, dec_result.logits) # Should return (batch, 1) masked sum

            # Accumulate loss and token count
            loss += step_loss
            step_mask = tf.cast(y_true != 0, tf.float32)
            total_tokens += tf.reduce_sum(step_mask)

            # Update test accuracy metric
            self.test_accuracy_tracker.update_state(y_true, dec_result.logits)


        # Compute the average loss over the batch and time steps
        average_loss = tf.cond(total_tokens > 0,
                               lambda: loss / total_tokens,
                               lambda: tf.constant(0.0, dtype=tf.float32))

        # Update loss metric with the calculated average batch loss
        self.test_loss_tracker.update_state(average_loss)

        # Return metric results for this step
        return {'loss': self.test_loss_tracker.result(),
                'accuracy': self.test_accuracy_tracker.result()}


class BatchLogs(keras.callbacks.Callback):
    def __init__(self, key):
        self.key = key
        self.logs = []

    def on_train_batch_end(self, n, logs):
        self.logs.append(logs.get(self.key)) # Use .get() for safety


class Translator(tf.Module):
    def __init__(self, encoder, decoder, input_text_processor,
                 output_text_processor):
        super().__init__() # Important to call parent constructor for tf.Module
        self.encoder = encoder
        self.decoder = decoder
        self.input_text_processor = input_text_processor
        self.output_text_processor = output_text_processor
        self.output_token_string_from_index = (
            keras.layers.StringLookup(
                vocabulary=output_text_processor.get_vocabulary(),
                mask_token='',
                invert=True))
        index_from_string = keras.layers.StringLookup(
            vocabulary=output_text_processor.get_vocabulary(), mask_token='')
        # Ensure start and end tokens are treated as strings
        token_mask_ids = index_from_string(['', '[UNK]', '[start]']).numpy()
        token_mask = np.zeros([index_from_string.vocabulary_size()], dtype=bool)
        token_mask[np.array(token_mask_ids)] = True
        self.token_mask = tf.constant(token_mask) # Make token_mask a tf.constant
        self.start_token = tf.constant(index_from_string(tf.constant('[start]')), dtype=tf.int64) # Ensure int64
        self.end_token = tf.constant(index_from_string(tf.constant('[end]')), dtype=tf.int64) # Ensure int64


    def tokens_to_text(self, result_tokens):
        # Removed ShapeChecker calls inside method
        # shape_checker = ShapeChecker()
        # shape_checker(result_tokens, ('batch', 't'))
        # Ensure input to string lookup is int64
        result_text_tokens = self.output_token_string_from_index(tf.cast(result_tokens, tf.int64))
        # shape_checker(result_text_tokens, ('batch', 't'))
        result_text = tf.strings.reduce_join(result_text_tokens,
                                             axis=1, separator=' ')
        # shape_checker(result_text, ('batch'))
        result_text = tf.strings.strip(result_text)
        # shape_checker(result_text, ('batch',))
        return result_text

    def sample(self, logits, temperature):
        # Removed ShapeChecker calls inside method
        # logits shape: (batch, t, vocab) - in translate loop, t is 1
        # shape_checker = ShapeChecker()
        # shape_checker(logits, ('batch', 't', 'vocab'))
        # shape_checker(self.token_mask, ('vocab',))

        # Ensure token_mask broadcast correctly
        token_mask_expanded = self.token_mask[tf.newaxis, tf.newaxis, :]
        # shape_checker(token_mask_expanded, ('batch', 't', 'vocab'), broadcast=True)

        # Apply mask to logits (shape (batch, t, vocab))
        logits = tf.where(token_mask_expanded, -tf.float32.max, logits) # Use max float value for -inf

        # Squeeze the 't' dimension before categorical sampling if t is 1
        # Add a check in case t is not 1 (e.g. if batch sampling was added later)
        if tf.shape(logits)[1] == 1:
             logits = tf.squeeze(logits, axis=1) # Shape becomes (batch, vocab)
        # shape_checker(logits, ('batch', 'vocab') or ('batch', 't', 'vocab')) # Depends on squeeze


        if temperature == 0.0:
            # Argmax over the vocabulary dimension
            new_tokens = tf.argmax(logits, axis=-1) # Shape (batch,) if squeezed, (batch, t) if not
        else:
            # Categorical sampling over the vocabulary dimension
            new_tokens = tf.random.categorical(logits/temperature,
                                               num_samples=1) # Shape (batch, 1) if squeezed, (batch, t, 1) if not


        new_tokens = tf.cast(new_tokens, tf.int64) # Ensure int64 dtype
        # Reshape to (batch, 1) regardless of initial shape after sampling
        new_tokens = tf.reshape(new_tokens, (-1, 1))
        # shape_checker(new_tokens, ('batch', 't')) # t=1 here

        return new_tokens


    def translate(self,
                  input_text, *,
                  max_length=50,
                  return_attention=True,
                  temperature=1.0):
        batch_size = tf.shape(input_text)[0]
        # Preprocess input text
        # We only need input_tokens and input_mask for inference
        input_tokens = self.input_text_processor(input_text) # strings -> int64
        input_mask = input_tokens != 0 # boolean mask for input padding

        # Encoder forward pass
        enc_output, enc_state = self.encoder(input_tokens) # int64 -> float

        dec_state = enc_state # Initialize decoder state

        # Start decoding with the start token
        new_tokens = tf.fill([batch_size, 1], self.start_token) # int64, shape (batch, 1)

        result_tokens = []
        attention_weights_list = []

        # `done` flags to track completed sequences in the batch
        done = tf.zeros([batch_size, 1], dtype=tf.bool) # boolean, shape (batch, 1)

        # Decoding loop
        for _ in tf.range(max_length): # Use tf.range for graph mode compatibility
            # If all sequences in the batch are done, we can stop early even in graph mode
            # using tf.while_loop with a condition is more robust for graph mode early stopping,
            # but a simple range loop with masking completed sequences is easier to implement.
            # The check `tf.reduce_all(done)` below is only effective in eager mode.

            # Prepare decoder input object for this single step
            decoder_input = Decoder.DecoderInput(
                new_tokens=new_tokens,      # Current input token(s) to decoder (batch, 1)
                enc_output=enc_output,      # Encoder output (batch, s, enc_units)
                mask=input_mask             # Input mask (batch, s)
            )

            # Perform one decoding step
            dec_result, dec_state = self.decoder(decoder_input, state=dec_state)
            # dec_result.logits shape (batch, 1, output_vocab_size)
            # dec_result.attention_weights shape (batch, 1, s)

            # Sample the next token for each sequence in the batch
            sampled_tokens = self.sample(dec_result.logits, temperature) # int64, shape (batch, 1)

            # Store attention weights if requested
            if return_attention:
                 attention_weights_list.append(dec_result.attention_weights) # shape (batch, 1, s)

            # Check if sequences are done (sampled end token or already done)
            just_sampled_end = (sampled_tokens == self.end_token) # boolean, shape (batch, 1)
            done = done | just_sampled_end # Update done flags

            # Append the sampled token to the results.
            # If a sequence is done, append padding (token 0) instead of the sampled token.
            # This maintains consistent tensor shape in the list for tf.concat later.
            sampled_tokens = tf.where(done, tf.constant(0, dtype=tf.int64), sampled_tokens)
            result_tokens.append(sampled_tokens) # List of (batch, 1) tensors

            # Update new_tokens for the next decoder step.
            # Even if done, we feed the padding token (0) back into the decoder
            # for subsequent steps until max_length is reached.
            new_tokens = sampled_tokens

            # Optimization: if all sequences in the batch are done, stop early (only effective in eager mode)
            if tf.executing_eagerly() and tf.reduce_all(done):
                 break


        # Concatenate the list of tensors from each time step along the time axis
        # From list of (batch, 1) tensors to one (batch, max_length) tensor
        result_tokens = tf.concat(result_tokens, axis=-1)

        # Convert token IDs back to text strings
        result_text = self.tokens_to_text(result_tokens) # (batch,)

        response = {'text': result_text}

        # Concatenate attention weights if requested
        if return_attention and attention_weights_list: # Check if list is not empty
            # Concatenate attention weights: list of (batch, 1, s) -> (batch, max_length, s)
            attention_stack = tf.concat(attention_weights_list, axis=1)
            response['attention'] = attention_stack

        return response


    @tf.function(input_signature=[tf.TensorSpec(dtype=tf.string, shape=[None])])
    def tf_translate(self, input_text):
        # This is the graph-compatible translation function
        return self.translate(input_text)

def load_vectorizer(from_file):
    # Loading Keras models requires custom objects if they use custom layers or functions
    custom_objects = {"custom_standardization": custom_standardization,
                      "TextVectorization": TextVectorization} # Add TextVectorization itself
    loaded_vectorizer_model = keras.models.load_model(from_file, custom_objects=custom_objects)
    # Assuming the vectorizer is the second layer (index 1) in the Sequential model
    if len(loaded_vectorizer_model.layers) > 1:
        vectorizer_layer = loaded_vectorizer_model.layers[1]
        if isinstance(vectorizer_layer, TextVectorization):
            # Recreate the vectorizer object from the loaded layer config
            lconfig = vectorizer_layer.get_config()
            # The loaded config should already have 'standardize' pointing to custom_standardization
            # Ensure output_mode is correctly set if it's not in config or needs overriding
            lconfig['output_mode'] = 'int' # Assuming it was saved as int mode
            vectorizer = TextVectorization.from_config(lconfig)

            # Set the vocabulary explicitly
            lvocab = vectorizer_layer.get_vocabulary()
            vectorizer.set_vocabulary(lvocab)

            # Call adapt or call the vectorizer once to build its variables
            # Adapt with a dummy string is one way
            # vectorizer.adapt(tf.constant([""])) # Adapting might reset vocabulary, set it again if needed
            # A safer way after setting vocabulary is to just call it once
            _ = vectorizer(tf.constant(["dummy string"])) # Call once to build variables

            print(f"Vectorizer loaded successfully from path: {from_file}")
            return vectorizer
        else:
             raise TypeError(f"Layer 1 in the loaded model is not a TextVectorization layer: {type(vectorizer_layer)}")
    else:
         raise ValueError("Loaded model does not have enough layers to extract vectorizer.")


def save_vectorizer(vectorizer, to_file):
    # Keras recommends saving TextVectorization layers within a Functional or Sequential model
    # This ensures the vocabulary is saved correctly.
    vectorizer_model = keras.models.Sequential()
    # Input layer is needed when saving a Sequential model
    vectorizer_model.add(keras.Input(shape=(1,), dtype=tf.string))
    vectorizer_model.add(vectorizer)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(to_file), exist_ok=True)

    # Add .keras extension if not present
    if not to_file.endswith('.keras'):
        to_file += '.keras'

    # Save the model using the native Keras format
    try:
        vectorizer_model.save(to_file)
        print(f"Vectorizer model saved to path: {to_file}")
    except Exception as e:
        print(f"Error when trying to save vectorizer to path {to_file}: {e}")


def parse_dataset_name(training_data):
    training_data = training_data.split(os.sep)[-1].lower()
    names_dic = {
        "NCD": 'ncd' in training_data,
        "GP": 'gp' in training_data,
        "CN": 'conceptnet' in training_data}
    return '-'.join([k for k, v in names_dic.items() if v])

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seqLen", type=int,
                    default=50, help="Per-sample sequence length")
parser.add_argument("-u", "--nSteps", type=int,
                   default=1024, help="Number of hidden recurrent steps (units)")
parser.add_argument("-f", "--nFeatures", type=int,
                   default=15000, help="Maximum vocabulary size")
parser.add_argument("-b", "--batchSize", type=int,
                   default=64, help="Batch size")
parser.add_argument("-e", "--nEpochs", type=int,
                   default=40, help="Number of training epochs")
parser.add_argument("-d", "--embeddingDim", type=int,
                   default=1024, help="Word embedding dimensionality")
parser.add_argument("-D", "--nDemo", type=int,
                   default=-1, help="Number of predicted test samples to save as output")
parser.add_argument("-T", "--trainData", type=str,
                   default="data/ncd_conceptnet/ncd_conceptnet_train.tsv",
                   help="Training data (CSV file)")
parser.add_argument("-t", "--testData", type=str,
                   default="data/ncd_conceptnet/ncd_conceptnet_valid.tsv",
                   help="Test data (CSV file)")
parser.add_argument("-rp", "--resPath", type=str,
                   default=os.getcwd(),
                   help="Path where results, vectorizer and neural network models are stored.")
args = parser.parse_args()

sequence_length = args.seqLen
max_features = args.nFeatures
batch_size = args.batchSize
n_epochs = args.nEpochs
embedding_dim = args.embeddingDim
units = args.nSteps
training_data = args.trainData
testing_data = args.testData
n_demo = args.nDemo
results_path = os.path.normpath(args.resPath) + os.sep
dataset_name = parse_dataset_name(training_data)

# Initialize BatchLogs callbacks (optional, mostly for manual inspection)
train_loss_bl = BatchLogs('loss')
train_accu_bl = BatchLogs('accuracy')
test_loss_bl = BatchLogs('loss')
test_accu_bl = BatchLogs('accuracy')

strip_chars = string.punctuation
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")

with open(training_data) as f:
    train_text = f.readlines()

with open(testing_data) as f:
    val_text = f.readlines()

logging.info("Preparing train and test data")
# Filter out None values from prepare_data due to malformed lines
train_pairs = [pair for pair in map(functools.partial(
    prepare_data, include_labels=cs_labels, all_start_end=True), train_text) if pair is not None]
val_pairs = [pair for pair in map(functools.partial(
    prepare_data, include_labels=cs_labels, all_start_end=True), val_text) if pair is not None]

if not train_pairs:
    logging.error("No valid training data pairs found after preparation. Exiting.")
    exit()
if not val_pairs:
    logging.warning("No valid validation data pairs found after preparation.")


# Initialize and adapt vectorizers first
input_vectorizer = keras.layers.TextVectorization(
    output_mode="int", max_tokens=max_features,
    output_sequence_length=sequence_length, standardize=custom_standardization)

# Output sequence length should be sequence_length + 1 to accommodate [start] and [end] tokens
output_sequence_length = sequence_length + 1
output_vectorizer = keras.layers.TextVectorization(
    output_mode="int", max_tokens=max_features,
    output_sequence_length=output_sequence_length,
    standardize=custom_standardization)

train_in_texts = [pair[0] for pair in train_pairs]
# Ensure correct extraction of output text based on cs_labels
if cs_labels:
    # Assuming pair[1] is a tuple (text, label)
    train_out_texts = [pair[1][0] for pair in train_pairs]
else:
    # Assuming pair[1] is just the text
    train_out_texts = [pair[1] for pair in train_pairs]

logging.info("Training input text vectorizer")
# Adapt vectorizers on the list of strings
input_vectorizer.adapt(train_in_texts)
# Define save paths clearly
in_vect_save_path = os.path.join(results_path, "results",
                                 f"attentionGRU_{dataset_name}_seqlen-{sequence_length}_vectorizer",
                                 "in_vect_model")
save_vectorizer(vectorizer=input_vectorizer, to_file=in_vect_save_path)

logging.info("Training output text vectorizer")
output_vectorizer.adapt(train_out_texts)
out_vect_save_path = os.path.join(results_path, "results",
                                  f"attentionGRU_{dataset_name}_seqlen-{sequence_length}_vectorizer",
                                  "out_vect_model")
save_vectorizer(vectorizer=output_vectorizer, to_file=out_vect_save_path)

logging.info(f"Saved text vectorizers to {os.path.dirname(in_vect_save_path)}")


# Create datasets using make_dataset AFTER vectorizers are defined and adapted
#dataset = make_dataset(train_pairs)
#test_dataset = make_dataset(val_pairs)
raw_train = tf.data.Dataset.from_tensor_slices(
    # each element is (string, string)
    ([p[0] for p in train_pairs],
     [p[1] for p in train_pairs])
).batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

raw_val = tf.data.Dataset.from_tensor_slices(
    ([p[0] for p in val_pairs],
     [p[1] for p in val_pairs])
).batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

# Update max_features based on the actual vocabulary size
max_vocab = max([
        input_vectorizer.vocabulary_size(),
        output_vectorizer.vocabulary_size()])
# Use the minimum of requested max_features and actual max_vocab size
max_features = min(args.nFeatures, max_vocab)


# Define checkpoint path and ensure directory exists
checkpoint_path = os.path.join(results_path, "results",
                               f"attentionGRU_{dataset_name}_epochs-{n_epochs}_seqlen-{sequence_length}_maxfeat-{max_features}_batch-{batch_size}_embdim-{embedding_dim}_steps-{units}",
                               "cp.weights.h5")

print("Working results directory: " + os.path.dirname(checkpoint_path))
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)


# Keras ModelCheckpoint callback
cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                              save_weights_only=True,
                                              verbose=1)

# Instantiate the TrainTranslator model
#train_translator = TrainTranslator(
#    embedding_dim, units,
#    input_text_processor=input_vectorizer,
#    output_text_processor=output_vectorizer)
# then train with raw_train/raw_val:
history = train_translator.fit(
    raw_train,
    validation_data=raw_val,
    epochs=n_epochs,
    callbacks=[cp_callback, train_loss_bl, train_accu_bl]
)

# Build the model by calling it on a dummy batch of the expected string input type
# This is necessary before compiling or loading weights
logging.info("Building the model...")
# Get a sample batch of *string* data from the original pairs
# Need at least batch_size samples if available
sample_train_pairs = train_pairs[:batch_size] if len(train_pairs) >= batch_size else train_pairs
if sample_train_pairs:
    dummy_string_input = tf.constant([p[0] for p in sample_train_pairs])
    dummy_string_target = tf.constant([p[1] if not cs_labels else p[1][0] for p in sample_train_pairs])

    try:
        # Call the model once to build it using dummy string data
        # Pass inputs as a tuple matching the train_step signature
        _ = train_translator((dummy_string_input, dummy_string_target))
        logging.info("Model built successfully.")
    except Exception as e:
        logging.error(f"Error building the model: {e}")
        # Re-raise the exception if building fails critically
        # raise # Uncomment to stop execution on build error
else:
    logging.warning("Not enough training data to build the model with a full batch.")
    # Attempt to build with a minimal call if full batch is not possible
    try:
        dummy_string_input = tf.constant(["dummy input"])
        dummy_string_target = tf.constant(["dummy target"])
        _ = train_translator((dummy_string_input, dummy_string_target))
        logging.info("Model built successfully with minimal data.")
    except Exception as e:
        logging.error(f"Error building the model with minimal data: {e}")
        # Re-raise the exception if building fails critically
        # raise


# Compile the model with the custom loss and optimizer
train_translator.compile(
    optimizer=keras.optimizers.Adam(),
    loss=MaskedLoss() # Pass the instance of the custom loss
)

# If you were previously training and saved weights, uncomment to load them
# try:
#     train_translator.load_weights(checkpoint_path)
#     logging.info(f"Loaded weights from {checkpoint_path}")
# except tf.errors.NotFoundError:
#     logging.info("No existing weights found. Training from scratch.")
# except Exception as e:
#     logging.error(f"Error loading weights: {e}")


logging.info("Checking dataset output shapes before training loop...")
# The dataset pipeline already applies format_dataset and yields INT tensors.
# Just iterate and print the shape of the yielded tensors.
# tf.config.run_functions_eagerly(True) # Uncomment to debug dataset iteration in eager mode



logging.info("Training neural reasoning model...")
# Use the compiled model's fit method, passing the dataset directly
history = train_translator.fit(
    dataset,
    validation_data=test_dataset, # Keras will automatically call test_step on validation_data
    epochs=n_epochs,
    callbacks=[cp_callback, train_loss_bl, train_accu_bl] # Add BatchLogs if you want batch-level history
)

logging.info("Training finished.")
logging.info("Saving evaluation results...")

# Ensure the output directory for results exists
out_dir = os.path.dirname(checkpoint_path) + os.sep
os.makedirs(out_dir, exist_ok=True)

# Save training history
rdf = pd.DataFrame(history.history)
history_csv_path = os.path.join(out_dir, "history.csv")
rdf.to_csv(history_csv_path)
logging.info(f"Saved training history to {history_csv_path}")

# Save history plot
try:
    fig, axes = plt.subplots(2, 1)
    # Ensure columns exist before plotting
    # Use .get() or check for column existence
    if 'loss' in rdf.columns and 'val_loss' in rdf.columns:
         rdf[['loss', 'val_loss']].plot(ax=axes[0])
         axes[0].set_title('Loss')
    elif 'loss' in rdf.columns:
         rdf[['loss']].plot(ax=axes[0])
         axes[0].set_title('Loss')


    if 'accuracy' in rdf.columns and 'val_accuracy' in rdf.columns:
         rdf[['accuracy', 'val_accuracy']].plot(ax=axes[1])
         axes[1].set_title('Accuracy')
    elif 'accuracy' in rdf.columns:
        rdf[['accuracy']].plot(ax=axes[1])
        axes[1].set_title('Accuracy')

    plt.xlabel('Epoch')
    plt.tight_layout()
    history_plot_path = os.path.join(out_dir, 'history_plot.pdf')
    plt.savefig(history_plot_path)
    logging.info(f"Saved history plot to {history_plot_path}")
except Exception as e:
    logging.error(f"Error saving history plot: {e}")


# Perform inference using the trained model's weights
# Instantiate the Translator module for inference
# It reuses the trained encoder and decoder from the TrainTranslator model
translator = Translator(
    encoder=train_translator.encoder, # Use the trained encoder
    decoder=train_translator.decoder, # Use the trained decoder
    input_text_processor=input_vectorizer,
    output_text_processor=output_vectorizer,
)

# Load weights into the Translator module's sub-layers if they weren't shared directly
# (In this case, they are shared objects, so weights are already there from training)
# If you had saved the full TrainTranslator model and loaded it separately,
# you might need to explicitly get encoder/decoder from the loaded model.
# As is, this should work fine.


if n_demo > 0: # Only run demo if n_demo is positive
    # Prepare test data for inference
    val_pairs_for_demo = val_pairs
    if len(val_pairs) > n_demo:
        random.seed(42) # Use a fixed seed for reproducible demo samples
        val_pairs_for_demo = random.sample(val_pairs, n_demo)

    inp_ = [inp for inp, targ in val_pairs_for_demo]
    targ_ = [targ for inp, targ in val_pairs_for_demo]

    results = []
    logging.info(f"Now performing inferences on {len(inp_)} test samples using the trained model...")

    # Create a tf.data.Dataset for the inference input text
    inference_dataset = tf.data.Dataset.from_tensor_slices(tf.constant(inp_))
    # Batch the inference dataset
    inference_dataset = inference_dataset.batch(batch_size)

    # Iterate through the batched inference dataset
    for batch_input_text in inference_dataset:
        # Use the tf_translate function for graph mode inference
        # It takes a batch of string tensors
        try:
            result_batch = translator.tf_translate(batch_input_text)['text'].numpy()
            results.extend(result_batch.tolist()) # Extend the list with results from the batch
        except Exception as e:
             logging.error(f"Error during inference batch: {e}")
             # Optionally break or handle the error

    # Combine inputs, true targets, and predictions
    result_df = pd.DataFrame({'Subj_Pred': inp_, 'Obj_true': targ_, 'Obj_predicted': results})

    # Save predictions
    predictions_csv_path = os.path.join(out_dir, 'predictions.csv')
    result_df.to_csv(predictions_csv_path, index=False) # Save without index
    print(result_df)
    logging.info(f"See the predictions written to {predictions_csv_path}")
else:
    logging.info("n_demo is not positive, skipping inference demo.")
