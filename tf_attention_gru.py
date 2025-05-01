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
            while i < len(line) and not line[i].strip().isdigit():
                complements.append(line[i])
                line.pop(i)

            line[3] = " ".join([line[3]] + complements)

    sample = [line[0], pred, line[2],
        start_token + line[3] + end_token, float(line[4].strip())]
    if not include_labels:
        del sample[-1]
        sample_o = sample[-1]
    else:
        sample_o = tuple(sample[-2:])
    if not include_sent:
        del sample[0]
        sample_i = ' '.join([sample[1], sample[0]])
        if all_start_end:
            sample_i = start_token + sample_i + end_token
    else:
        sample_i = ' '.join([sample[0], sample[2], sample[1]])

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
        shape_checker = ShapeChecker()
        shape_checker(tokens, ('batch', 's'))
        vectors = self.embedding(tokens)
        shape_checker(vectors, ('batch', 's', 'embed_dim'))
        output, state = self.gru(vectors, initial_state=state)
        shape_checker(output, ('batch', 's', 'enc_units'))
        shape_checker(state, ('batch', 'enc_units'))
        return output, state

class BahdanauAttention(keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.W1 = keras.layers.Dense(units, use_bias=False)
        self.W2 = keras.layers.Dense(units, use_bias=False)
        self.attention = keras.layers.AdditiveAttention()

    def call(self, query, value, mask):
        shape_checker = ShapeChecker()
        shape_checker(query, ('batch', 't', 'query_units'))
        shape_checker(value, ('batch', 's', 'value_units'))
        shape_checker(mask, ('batch', 's')) # Mask is boolean for padding tokens
        w1_query = self.W1(query)
        shape_checker(w1_query, ('batch', 't', 'attn_units'))
        w2_key = self.W2(value)
        shape_checker(w2_key, ('batch', 's', 'attn_units'))

        # The attention layer expects masks as boolean tensors
        query_mask = tf.ones(tf.shape(query)[:-1], dtype=bool) # Assuming query does not need masking per time step
        value_mask = mask # Use the provided input mask

        context_vector, attention_weights = self.attention(
            inputs=[w1_query, value, w2_key],
            mask=[query_mask, value_mask],
            return_attention_scores=True,
        )
        shape_checker(context_vector, ('batch', 't', 'value_units'))
        shape_checker(attention_weights, ('batch', 't', 's'))
        return context_vector, attention_weights

class Decoder(keras.layers.Layer):
    def __init__(self, output_vocab_size, embedding_dim, dec_units):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.output_vocab_size = output_vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = keras.layers.Embedding(self.output_vocab_size, embedding_dim)
        self.gru = keras.layers.GRU(self.dec_units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')
        self.attention = BahdanauAttention(self.dec_units)
        self.Wc = keras.layers.Dense(dec_units, activation=tf.math.tanh, use_bias=False)
        self.fc = keras.layers.Dense(self.output_vocab_size)

    class DecoderInput(typing.NamedTuple):
        new_tokens: Any # Shape: (batch, t)
        enc_output: Any # Shape: (batch, s, enc_units)
        mask: Any       # Shape: (batch, s) - boolean mask for enc_output padding


    class DecoderOutput(typing.NamedTuple):
        logits: Any            # Shape: (batch, t, output_vocab_size)
        attention_weights: Any # Shape: (batch, t, s)

    def call(self,
             inputs: DecoderInput,
             state=None) -> Tuple[DecoderOutput, tf.Tensor]:
        shape_checker = ShapeChecker()
        shape_checker(inputs.new_tokens, ('batch', 't'))
        shape_checker(inputs.enc_output, ('batch', 's', 'enc_units'))
        shape_checker(inputs.mask, ('batch', 's'))

        if state is not None:
            shape_checker(state, ('batch', 'dec_units'))

        vectors = self.embedding(inputs.new_tokens)
        shape_checker(vectors, ('batch', 't', 'embedding_dim'))

        # In a typical seq2seq decoder loop, 't' is 1.
        # The GRU call should match the shape expectations.
        # If inputs.new_tokens has shape (batch, 1), vectors will be (batch, 1, embedding_dim)
        # gru will return (batch, 1, dec_units) for rnn_output
        rnn_output, state = self.gru(vectors, initial_state=state)
        shape_checker(rnn_output, ('batch', 't', 'dec_units'))
        shape_checker(state, ('batch', 'dec_units'))

        # Pass the correct shapes and mask to attention
        context_vector, attention_weights = self.attention(
            query=rnn_output, value=inputs.enc_output, mask=inputs.mask)

        # context_vector should have the same 't' dimension as the query (rnn_output)
        shape_checker(context_vector, ('batch', 't', 'dec_units'))
        shape_checker(attention_weights, ('batch', 't', 's'))

        # Concatenate context vector and GRU output
        context_and_rnn_output = tf.concat([context_vector, rnn_output], axis=-1)
        # The Wc layer should process the concatenated tensor
        attention_vector = self.Wc(context_and_rnn_output) # Should be (batch, t, dec_units)

        shape_checker(attention_vector, ('batch', 't', 'dec_units'))

        # Final output layer
        logits = self.fc(attention_vector)
        shape_checker(logits, ('batch', 't', 'output_vocab_size'))

        return self.DecoderOutput(logits, attention_weights), state


class MaskedLoss(keras.losses.Loss):
    def __init__(self):
        self.name = 'masked_loss'
        self.loss = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')

    def __call__(self, y_true, y_pred):
        shape_checker = ShapeChecker()
        # y_true is the actual target token (batch, 1)
        # y_pred is the logits from the decoder output (batch, 1, vocab_size)
        shape_checker(y_true, ('batch', 't')) # t should be 1 in the training loop step
        shape_checker(y_pred, ('batch', 't', 'logits')) # t should be 1 in the training loop step

        # Ensure y_true has the same time dimension as y_pred for loss calculation
        # y_true needs to be reshaped to (batch, t) to match y_pred's shape before loss calculation
        # However, the loop passes y_true as (batch, 1), which is already correct if y_pred is (batch, 1, logits)
        loss = self.loss(y_true, y_pred) # loss will be (batch, t) after reduction=none
        shape_checker(loss, ('batch', 't'))

        # Mask out loss for padding tokens (token 0)
        mask = tf.cast(y_true != 0, tf.float32) # mask will be (batch, t)
        shape_checker(mask, ('batch', 't'))

        loss *= mask # loss is now (batch, t), masked

        # Return the sum over the batch and time steps, averaged later by non-masked elements
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
        self.shape_checker = ShapeChecker()
        # Metrics reset automatically at the start of each epoch
        self.train_loss_tracker = keras.metrics.Mean(name="loss")
        self.train_accuracy_tracker = keras.metrics.SparseCategoricalAccuracy(name="accuracy")
        self.test_loss_tracker = keras.metrics.Mean(name="loss")
        self.test_accuracy_tracker = keras.metrics.SparseCategoricalAccuracy(name="accuracy")


    @property
    def metrics(self):
        # List the metrics here so they are tracked by .fit()
        return [self.train_loss_tracker, self.train_accuracy_tracker]

    # No need for a separate test_metrics property, Keras handles it
    # if you use .evaluate() or validation_data in .fit()

    # Removed the @tf.function decorator from the main call method
    # because it handles mixed string and int inputs for initial build.
    # The decorated _tf_train_step and _tf_test_step are used for the main loops.
    def call(self, inputs, training=False):
        # This call method is primarily used for model building by Keras
        # It should accept the raw data input type (strings)
        # And output something Keras can use to infer output shapes.
        # The actual training loop will use _train_step which handles tokenized data.

        # Expect inputs as a tuple of string tensors during .fit (if use_tf_function is True)
        # Or directly as string tensors during initial build if not using tf.function
        # or during evaluation if _test_step is used manually.

        if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
            input_text, target_text = inputs
        else:
             # Fallback for initial build or direct call with only input text
             # Keras might pass only x during model building
             input_text = inputs
             # Create a dummy target_text with appropriate dtype (string)
             # Its content doesn't matter for shape inference
             dummy_target_text = tf.fill(tf.shape(input_text), "")
             target_text = dummy_target_text


        # Preprocess string inputs to get tokenized integers and masks
        input_tokens, input_mask, target_tokens, target_mask = self._preprocess(input_text, target_text)

        # Perform a single forward pass relevant for Keras model building
        # This needs to represent the sequence prediction process structure.
        # A common way is to pass the encoder output and the start token
        # through the decoder for one step.

        enc_output, enc_state = self.encoder(input_tokens)

        # Use the first token of the target sequence as the initial decoder input
        # This assumes target_tokens includes a start token at index 0
        initial_dec_token = target_tokens[:, :1] # Shape (batch, 1)

        dec_state = enc_state # Initialize decoder state with encoder final state

        decoder_input = Decoder.DecoderInput( # Use the nested class reference
            new_tokens=initial_dec_token,
            enc_output=enc_output,
            mask=input_mask # Pass the input mask to attention
        )

        # Perform one decoding step
        dec_result, _ = self.decoder(decoder_input, state=dec_state)

        # The model's output for Keras should be the decoder's logits
        # This is typically the output logits for the *next* token prediction.
        # Since we did one step with the start token, the logits predict the first actual target token.
        return dec_result.logits # Shape (batch, 1, output_vocab_size)


    def _preprocess(self, input_text, target_text):
        # This takes string tensors and returns integer tensors + masks
        self.shape_checker(input_text, ('batch',))
        self.shape_checker(target_text, ('batch',))

        # Apply TextVectorization layers
        input_tokens = self.input_text_processor(input_text)
        target_tokens = self.output_text_processor(target_text)

        # TextVectorization returns int64 by default if output_mode="int"
        self.shape_checker(input_tokens, ('batch', 's'))
        self.shape_checker(target_tokens, ('batch', 't'))

        # Create masks (True for non-padding tokens, False for padding token 0)
        input_mask = input_tokens != 0 # Boolean mask for input sequence padding
        target_mask = target_tokens != 0 # Boolean mask for target sequence padding

        self.shape_checker(input_mask, ('batch', 's'))
        self.shape_checker(target_mask, ('batch', 't'))

        return input_tokens, input_mask, target_tokens, target_mask

    # Use @tf.function here for efficient training loop
    # Specify input_signature to compile the function graph
    @tf.function(input_signature=[[
        tf.TensorSpec(dtype=tf.string, shape=[None]),
        tf.TensorSpec(dtype=tf.string, shape=[None])
    ]])
    def train_step(self, inputs):
        # inputs will be a tuple of string tensors (input_text, target_text) due to input_signature
        input_text, target_text = inputs

        # Preprocess strings to get tokenized integers and masks
        (input_tokens, input_mask,
         target_tokens, target_mask) = self._preprocess(input_text, target_text)

        # Ensure target_tokens are int64 for loss calculation
        target_tokens = tf.cast(target_tokens, tf.int64)

        max_target_length = tf.shape(target_tokens)[1]

        with tf.GradientTape() as tape:
            # Encoder forward pass
            enc_output, enc_state = self.encoder(input_tokens)
            self.shape_checker(enc_output, ('batch', 's', 'enc_units'))
            self.shape_checker(enc_state, ('batch', 'enc_units'))

            dec_state = enc_state # Initialize decoder state

            loss = tf.constant(0.0)
            total_tokens = tf.constant(0.0) # Counter for non-padding tokens

            # Decoder teacher forcing loop
            # Iterate from the first token of the target (usually [start]) up to the second-to-last token
            # because each step predicts the *next* token.
            for t in tf.range(max_target_length - 1):
                # Use the current target token `target_tokens[:, t:t+1]` as the decoder input
                # The target for the loss is the *next* token `target_tokens[:, t+1:t+2]`
                new_tokens = target_tokens[:, t:t+1] # Current input token to decoder
                y_true = target_tokens[:, t+1:t+2] # Next token is the true target

                decoder_input = Decoder.DecoderInput( # Use the nested class reference
                    new_tokens=new_tokens,
                    enc_output=enc_output,
                    mask=input_mask # Pass the input mask to attention
                )

                dec_result, dec_state = self.decoder(decoder_input, state=dec_state)

                # dec_result.logits shape: (batch, 1, output_vocab_size)
                # y_true shape: (batch, 1)

                # Calculate loss for this time step
                step_loss = self.loss(y_true, dec_result.logits) # Should be (batch, 1) after reduction=none

                # Mask the loss based on the true target token
                mask = tf.cast(y_true != 0, tf.float32) # (batch, 1)
                step_loss *= mask # (batch, 1), masked

                loss += tf.reduce_sum(step_loss) # Sum masked loss over batch
                total_tokens += tf.reduce_sum(mask) # Count non-padding tokens

                # Update training metric
                # accuracy.update_state expects (y_true, y_pred_logits) where y_true is integer indices
                # y_true should be (batch, 1) and y_pred_logits should be (batch, 1, vocab_size)
                self.train_accuracy_tracker.update_state(y_true, dec_result.logits)


            # Compute the average loss
            # Avoid division by zero if a batch only contains padding (unlikely but safe)
            average_loss = tf.cond(total_tokens > 0,
                                   lambda: loss / total_tokens,
                                   lambda: tf.constant(0.0))

        # Get gradients and apply optimizer
        variables = self.trainable_variables
        gradients = tape.gradient(average_loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        # Update loss metric manually since reduction='none' was used
        # The Mean metric updates based on the *value* passed to update_state, not gradients
        # We pass the calculated average_loss for the batch
        self.train_loss_tracker.update_state(average_loss)


        # Return metrics for this step
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

        # Ensure target_tokens are int64 for loss calculation
        target_tokens = tf.cast(target_tokens, tf.int64)

        max_target_length = tf.shape(target_tokens)[1]

        enc_output, enc_state = self.encoder(input_tokens)
        dec_state = enc_state

        loss = tf.constant(0.0)
        total_tokens = tf.constant(0.0)

        # Decoder teacher forcing loop for evaluation
        for t in tf.range(max_target_length - 1):
            new_tokens = target_tokens[:, t:t+1]
            y_true = target_tokens[:, t+1:t+2]

            decoder_input = Decoder.DecoderInput( # Use the nested class reference
                new_tokens=new_tokens,
                enc_output=enc_output,
                mask=input_mask # Pass the input mask to attention
            )

            dec_result, dec_state = self.decoder(decoder_input, state=dec_state)

            step_loss = self.loss(y_true, dec_result.logits)
            mask = tf.cast(y_true != 0, tf.float32)
            step_loss *= mask

            loss += tf.reduce_sum(step_loss)
            total_tokens += tf.reduce_sum(mask)

            # Update test metric
            self.test_accuracy_tracker.update_state(y_true, dec_result.logits)

        # Compute the average loss
        average_loss = tf.cond(total_tokens > 0,
                               lambda: loss / total_tokens,
                               lambda: tf.constant(0.0))

        # Update loss metric manually
        self.test_loss_tracker.update_state(average_loss)

        # Return metrics for this step
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
        self.start_token = index_from_string(tf.constant('[start]'))
        self.end_token = index_from_string(tf.constant('[end]'))

    def tokens_to_text(self, result_tokens):
        shape_checker = ShapeChecker()
        shape_checker(result_tokens, ('batch', 't'))
        # Ensure input to string lookup is int64
        result_text_tokens = self.output_token_string_from_index(tf.cast(result_tokens, tf.int64))
        shape_checker(result_text_tokens, ('batch', 't'))
        result_text = tf.strings.reduce_join(result_text_tokens,
                                             axis=1, separator=' ')
        shape_checker(result_text, ('batch'))
        result_text = tf.strings.strip(result_text)
        shape_checker(result_text, ('batch',))
        return result_text

    def sample(self, logits, temperature):
        # logits shape: (batch, t, vocab) - in translate loop, t is 1
        shape_checker = ShapeChecker()
        shape_checker(logits, ('batch', 't', 'vocab'))
        shape_checker(self.token_mask, ('vocab',))

        # Ensure token_mask broadcast correctly
        token_mask_expanded = self.token_mask[tf.newaxis, tf.newaxis, :]
        shape_checker(token_mask_expanded, ('batch', 't', 'vocab'), broadcast=True)

        # Apply mask to logits
        logits = tf.where(token_mask_expanded, -np.inf, logits)

        # Squeeze the 't' dimension before categorical sampling
        logits = tf.squeeze(logits, axis=1) # Shape becomes (batch, vocab)
        shape_checker(logits, ('batch', 'vocab'))

        if temperature == 0.0:
            new_tokens = tf.argmax(logits, axis=-1) # Shape (batch,)
        else:
            new_tokens = tf.random.categorical(logits/temperature,
                                               num_samples=1) # Shape (batch, 1)

        # Ensure the output shape is (batch, 1)
        new_tokens = tf.cast(new_tokens, tf.int64) # Ensure int64 dtype
        new_tokens = tf.reshape(new_tokens, (-1, 1)) # Reshape to (batch, 1)
        shape_checker(new_tokens, ('batch', 't'))

        return new_tokens


    def translate(self,
                  input_text, *,
                  max_length=50,
                  return_attention=True,
                  temperature=1.0):
        batch_size = tf.shape(input_text)[0]
        input_tokens = self.input_text_processor(input_text) # strings -> int64
        input_mask = input_tokens != 0 # boolean mask for input padding

        enc_output, enc_state = self.encoder(input_tokens) # int64 -> float

        dec_state = enc_state # float

        # Start decoding with the start token
        new_tokens = tf.fill([batch_size, 1], self.start_token) # int64, shape (batch, 1)

        result_tokens = []
        attention_weights_list = [] # Renamed for clarity

        # `done` flags to track completed sequences
        done = tf.zeros([batch_size, 1], dtype=tf.bool) # boolean, shape (batch, 1)

        for _ in tf.range(max_length): # Use tf.range for graph mode compatibility
            decoder_input = Decoder.DecoderInput( # Use the nested class reference
                new_tokens=new_tokens,      # Current input token(s) to decoder (batch, 1)
                enc_output=enc_output,      # Encoder output (batch, s, enc_units)
                mask=input_mask             # Input mask (batch, s)
            )

            dec_result, dec_state = self.decoder(decoder_input, state=dec_state)

            # dec_result.logits shape (batch, 1, output_vocab_size)
            # dec_result.attention_weights shape (batch, 1, s)

            # Sample the next token
            sampled_tokens = self.sample(dec_result.logits, temperature) # int64, shape (batch, 1)

            # Store attention weights if requested
            if return_attention:
                 attention_weights_list.append(dec_result.attention_weights) # shape (batch, 1, s)

            # Check if sequences are done (sampled end token or already done)
            just_done = (sampled_tokens == self.end_token) # boolean, shape (batch, 1)
            done = done | just_done # boolean, shape (batch, 1)

            # Append the sampled token to the results
            # If a sequence is done, append padding (token 0) instead of the sampled token
            # This is crucial for consistent shape during concat later
            sampled_tokens = tf.where(done, tf.constant(0, dtype=tf.int64), sampled_tokens)
            result_tokens.append(sampled_tokens) # List of (batch, 1) tensors

            # Update new_tokens for the next decoder step
            new_tokens = sampled_tokens

            # Optimization: if all sequences in the batch are done, stop early in eager mode
            if tf.executing_eagerly() and tf.reduce_all(done):
                 break

        # Concatenate results and attention weights along the time axis
        result_tokens = tf.concat(result_tokens, axis=-1) # (batch, max_length)
        result_text = self.tokens_to_text(result_tokens) # (batch,)

        response = {'text': result_text}

        if return_attention:
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
    custom_objects = {"custom_standardization": custom_standardization}
    loaded_vectorizer_model = keras.models.load_model(from_file, custom_objects=custom_objects)
    vectorizer_layer = loaded_vectorizer_model.layers[1] # Assuming layer 0 is Input, layer 1 is TextVectorization

    # Recreate the vectorizer object from the loaded layer
    lconfig = vectorizer_layer.get_config()
    # The 'output_mode' might be set during saving, remove if necessary for recreation
    # lconfig.pop('output_mode', None) # Remove output_mode if it causes issues during recreation
    # Ensure the custom standardization function is passed if it was used
    lconfig['standardize'] = custom_standardization # Ensure custom function is linked

    # Explicitly set the output mode again if needed, or rely on the loaded config
    # lconfig['output_mode'] = "int" # Or whatever mode was used

    vectorizer = TextVectorization.from_config(lconfig)

    # The from_config method doesn't load vocabulary, we need to set it explicitly
    lvocab = vectorizer_layer.get_vocabulary()
    vectorizer.set_vocabulary(lvocab)

    # Need to call adapt or call the vectorizer once to build its variables
    # Adapting with a dummy string is one way
    vectorizer.adapt(tf.constant(["", "a"])) # Adapt with empty string and a placeholder

    print(f"Vectorizer loaded from path: {from_file}")
    return vectorizer


def save_vectorizer(vectorizer, to_file):
    # Keras recommends saving TextVectorization layers within a Functional or Sequential model
    # This ensures the vocabulary is saved correctly.
    vectorizer_model = keras.models.Sequential()
    # Input layer is needed when saving a Sequential model
    vectorizer_model.add(keras.Input(shape=(1,), dtype=tf.string))
    vectorizer_model.add(vectorizer)

    # Compile is not strictly necessary for saving, but harmless.
    # vectorizer_model.compile()

    # Ensure the directory exists
    os.makedirs(os.path.dirname(to_file), exist_ok=True)

    # Add .keras extension if not present
    if not to_file.endswith('.keras'):
        to_file += '.keras'

    # Save the model
    try:
        # Using the native Keras format (.keras) which is recommended
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

# Initialize BatchLogs callbacks
# Note: Keras metrics tracked by the model are generally preferred for .fit()
# BatchLogs can be useful for inspecting batch-level metrics explicitly.
train_loss_bl = BatchLogs('loss')
train_accu_bl = BatchLogs('accuracy')
test_loss_bl = BatchLogs('loss') # Note: these won't work directly with validation_data in .fit
test_accu_bl = BatchLogs('accuracy') # You would need a custom training loop or callback for test batch logs

strip_chars = string.punctuation
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "") # Corrected variable name

with open(training_data) as f:
    train_text = f.readlines()

with open(testing_data) as f:
    val_text = f.readlines()

logging.info("Preparing train and test data")
# Ensure prepare_data handles potential index errors if lines are not correctly formatted
try:
    train_pairs = list(
        map(functools.partial(
            prepare_data,
            include_labels=cs_labels, all_start_end=True), train_text))
    val_pairs = list(
        map(functools.partial(
            prepare_data,
            include_labels=cs_labels, all_start_end=True), val_text))
except IndexError as e:
    logging.error(f"Error processing data file: {e}. Check file format.")
    exit() # Exit if data loading fails


# Initialize and adapt vectorizers first
input_vectorizer = keras.layers.TextVectorization(
    output_mode="int", max_tokens=max_features,
    output_sequence_length=sequence_length, standardize=custom_standardization)

# Output sequence length should be sequence_length + 1 to accommodate [start] and [end] tokens
# and predict the sequence including [end]
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
# make_dataset now correctly batches strings and then maps format_dataset
# to tokenize them into integers.
dataset = make_dataset(train_pairs)
test_dataset = make_dataset(val_pairs)

# Update max_features based on the actual vocabulary size
max_vocab = max([
        len(input_vectorizer.get_vocabulary()),
        len(output_vectorizer.get_vocabulary())])
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
train_translator = TrainTranslator(
    embedding_dim, units,
    input_text_processor=input_vectorizer,
    output_text_processor=output_vectorizer)

# Build the model by calling it on a dummy batch or the first batch from the dataset
# This is necessary before compiling or loading weights
logging.info("Building the model...")
# Get a sample batch from the dataset
for example_input_batch, example_target_batch in dataset.take(1):
    # Pass the raw string batch to the model's call method for building
    # The call method handles preprocessing internally for building
    # Need to get the original string tensors before format_dataset was applied
    # A simpler way is to pass dummy tensors of the correct dtype (string)
    # Or rely on .fit() or .compile(run_eagerly=True) to build it.
    # Let's get original strings from pairs and pass them to the model call for building
    dummy_string_input = tf.constant([p[0] for p in train_pairs[:batch_size]])
    dummy_string_target = tf.constant([p[1] if not cs_labels else p[1][0] for p in train_pairs[:batch_size]])

    # The model's call method is designed to take string inputs for building
    _ = train_translator((dummy_string_input, dummy_string_target)) # Call the model once to build it
    logging.info("Model built.")
    break # Only need one batch to build


# Compile the model with the custom loss and optimizer
# Ensure the loss is the MaskedLoss instance
train_translator.compile(
    optimizer=keras.optimizers.Adam(),
    loss=MaskedLoss()
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
# Remove the redundant call to format_dataset here.
# The dataset pipeline already applies format_dataset.
# Just iterate and print the shape of the yielded tensors.
#tf.config.run_functions_eagerly(True) # Keep eager for debugging dataset output if needed
for in_phr, out_phr in dataset.take(1):
    print("Dataset batch yielded - Input shape:", in_phr.shape, "Target shape:", out_phr.shape)
    # Optional: Check dtypes - should be int64
    print("Dataset batch yielded - Input dtype:", in_phr.dtype, "Target dtype:", out_phr.dtype)
#tf.config.run_functions_eagerly(False) # Turn eager off for training if desired


logging.info("Training neural reasoning model...")
# Use the compiled model's fit method
history = train_translator.fit(
    dataset,
    validation_data=test_dataset,
    epochs=n_epochs,
    callbacks=[cp_callback] # Add other callbacks like EarlyStopping if needed
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
    if 'loss' in rdf.columns and 'val_loss' in rdf.columns:
         rdf[['loss', 'val_loss']].plot(ax=axes[0])
    if 'accuracy' in rdf.columns and 'val_accuracy' in rdf.columns:
         rdf[['accuracy', 'val_accuracy']].plot(ax=axes[1])
    plt.tight_layout()
    history_plot_path = os.path.join(out_dir, 'history_plot.pdf')
    plt.savefig(history_plot_path)
    logging.info(f"Saved history plot to {history_plot_path}")
except Exception as e:
    logging.error(f"Error saving history plot: {e}")


# Perform inference using the trained model's weights
# Instantiate the Translator module for inference
translator = Translator(
    encoder=train_translator.encoder, # Use the trained encoder from train_translator
    decoder=train_translator.decoder, # Use the trained decoder from train_translator
    input_text_processor=input_vectorizer,
    output_text_processor=output_vectorizer,
)

# Load weights into the Translator module's sub-layers if they weren't shared directly
# (In this case, they are shared objects, so weights are already there)
# If you had saved the full TrainTranslator model and loaded it separately,
# you might need to explicitly get encoder/decoder from the loaded model.
# As is, this should work fine.

if not (n_demo < 0 or isinstance(n_demo, str)):
    # Prepare test data for inference
    if len(val_pairs) > n_demo:
        random.seed(42) # Use a fixed seed for reproducible demo samples
        val_pairs_subset = random.sample(val_pairs, n_demo)
    else:
        val_pairs_subset = val_pairs

    inp_ = [inp for inp, targ in val_pairs_subset]
    targ_ = [targ for inp, targ in val_pairs_subset]

    results = []
    logging.info(f"Now performing inferences on {len(inp_)} test samples using the trained model...")

    # Process inference in batches
    # Create a tf.data.Dataset for the inference data
    inference_dataset = tf.data.Dataset.from_tensor_slices(tf.constant(inp_))
    inference_dataset = inference_dataset.batch(batch_size)

    for batch_input_text in inference_dataset:
        # Use the tf_translate function for graph mode inference
        result = translator.tf_translate(batch_input_text)['text'].numpy()
        results.extend(result.tolist()) # Extend the list with results from the batch

    # Combine inputs, true targets, and predictions
    result_df = pd.DataFrame({'Subj_Pred': inp_, 'Obj_true': targ_, 'Obj_predicted': results})

    # Save predictions
    predictions_csv_path = os.path.join(out_dir, 'predictions.csv')
    result_df.to_csv(predictions_csv_path, index=False) # Save without index
    print(result_df)
    logging.info(f"See the predictions written to {predictions_csv_path}")
else:
    logging.info("n_demo is set to -1 or not a number, skipping inference demo.")
