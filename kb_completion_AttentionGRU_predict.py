import numpy as np
import pandas as pd
import typing
from typing import Any, Tuple
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import functools, re, string, os, time, random
import argparse
from pdb import set_trace as st
from tqdm import tqdm
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
            while not line[i].isdigit():
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

    return  sample_i, sample_o

"""
In the case of immiting the following decorator 'keras serializable', this error
is going to raise:
raise RuntimeError(
RuntimeError: Unable to restore a layer of class TextVectorization. Layers of class TextVectorization require that the class be provided to the model loading code, either by registering the class using @keras.utils.register_keras_serializable on the class def and including that file in your program, or by passing the class in a keras.utils.CustomObjectScope that wraps this load call.
"""
@tf.keras.utils.register_keras_serializable()
def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(
        lowercase, "[%s]" % re.escape(strip_chars), "")


def format_dataset(in_phr, out_phr):
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
    dataset = dataset.map(format_dataset)
    return dataset.shuffle(2048).prefetch(16).cache()


def sort_cols(columns):
    ends = np.unique([c[-2:] for c in columns])
    new_cols = []
    for e in ends:
        for c in columns:
            if c.endswith(e):
                new_cols.append(c)
    return new_cols


#@title Shape checker
class ShapeChecker():
  def __init__(self):
    # Keep a cache of every axis-name seen
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

      if (broadcast and new_dim == 1):
        continue

      if old_dim is None:
        # If the axis name is new, add its length to the cache.
        self.shapes[name] = new_dim
        continue

      if new_dim != old_dim:
        raise ValueError(f"Shape mismatch for dimension: '{name}'\n"
                         f"    found: {new_dim}\n"
                         f"    expected: {old_dim}\n")

class Encoder(tf.keras.layers.Layer):
  def __init__(self, input_vocab_size, embedding_dim, enc_units):
    super(Encoder, self).__init__()
    self.enc_units = enc_units
    self.input_vocab_size = input_vocab_size

    # The embedding layer converts tokens to vectors
    self.embedding = tf.keras.layers.Embedding(self.input_vocab_size,
                                               embedding_dim)

    # The GRU RNN layer processes those vectors sequentially.
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   # Return the sequence and state
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, tokens, state=None):
    shape_checker = ShapeChecker()
    shape_checker(tokens, ('batch', 's'))

    # 2. The embedding layer looks up the embedding for each token.
    vectors = self.embedding(tokens)
    shape_checker(vectors, ('batch', 's', 'embed_dim'))

    # 3. The GRU processes the embedding sequence.
    #    output shape: (batch, s, enc_units)
    #    state shape: (batch, enc_units)
    output, state = self.gru(vectors, initial_state=state)
    shape_checker(output, ('batch', 's', 'enc_units'))
    shape_checker(state, ('batch', 'enc_units'))

    # 4. Returns the new sequence and its state.
    return output, state


class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super().__init__()
    # For Eqn. (4), the  Bahdanau attention
    self.W1 = tf.keras.layers.Dense(units, use_bias=False)
    self.W2 = tf.keras.layers.Dense(units, use_bias=False)

    self.attention = tf.keras.layers.AdditiveAttention()

  def call(self, query, value, mask):
    shape_checker = ShapeChecker()
    shape_checker(query, ('batch', 't', 'query_units'))
    shape_checker(value, ('batch', 's', 'value_units'))
    shape_checker(mask, ('batch', 's'))

    # From Eqn. (4), `W1@ht`.
    w1_query = self.W1(query)
    shape_checker(w1_query, ('batch', 't', 'attn_units'))

    # From Eqn. (4), `W2@hs`.
    w2_key = self.W2(value)
    shape_checker(w2_key, ('batch', 's', 'attn_units'))

    query_mask = tf.ones(tf.shape(query)[:-1], dtype=bool)
    value_mask = mask

    context_vector, attention_weights = self.attention(
        inputs = [w1_query, value, w2_key],
        mask=[query_mask, value_mask],
        return_attention_scores = True,
    )
    shape_checker(context_vector, ('batch', 't', 'value_units'))
    shape_checker(attention_weights, ('batch', 't', 's'))

    return context_vector, attention_weights


class Decoder(tf.keras.layers.Layer):
  def __init__(self, output_vocab_size, embedding_dim, dec_units):
    super(Decoder, self).__init__()
    self.dec_units = dec_units
    self.output_vocab_size = output_vocab_size
    self.embedding_dim = embedding_dim

    # For Step 1. The embedding layer convets token IDs to vectors
    self.embedding = tf.keras.layers.Embedding(self.output_vocab_size,
                                               embedding_dim)

    # For Step 2. The RNN keeps track of what's been generated so far.
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

    # For step 3. The RNN output will be the query for the attention layer.
    self.attention = BahdanauAttention(self.dec_units)

    # For step 4. Eqn. (3): converting `ct` to `at`
    self.Wc = tf.keras.layers.Dense(dec_units, activation=tf.math.tanh,
                                    use_bias=False)

    # For step 5. This fully connected layer produces the logits for each
    # output token.
    self.fc = tf.keras.layers.Dense(self.output_vocab_size)


class DecoderInput(typing.NamedTuple):
  new_tokens: Any
  enc_output: Any
  mask: Any

class DecoderOutput(typing.NamedTuple):
  logits: Any
  attention_weights: Any


def call(self,
         inputs: DecoderInput,
         state=None) -> Tuple[DecoderOutput, tf.Tensor]:
  shape_checker = ShapeChecker()
  shape_checker(inputs.new_tokens, ('batch', 't'))
  shape_checker(inputs.enc_output, ('batch', 's', 'enc_units'))
  shape_checker(inputs.mask, ('batch', 's'))

  if state is not None:
    shape_checker(state, ('batch', 'dec_units'))

  # Step 1. Lookup the embeddings
  vectors = self.embedding(inputs.new_tokens)
  shape_checker(vectors, ('batch', 't', 'embedding_dim'))

  # Step 2. Process one step with the RNN
  rnn_output, state = self.gru(vectors, initial_state=state)

  shape_checker(rnn_output, ('batch', 't', 'dec_units'))
  shape_checker(state, ('batch', 'dec_units'))

  # Step 3. Use the RNN output as the query for the attention over the
  # encoder output.
  context_vector, attention_weights = self.attention(
      query=rnn_output, value=inputs.enc_output, mask=inputs.mask)
  shape_checker(context_vector, ('batch', 't', 'dec_units'))
  shape_checker(attention_weights, ('batch', 't', 's'))

  # Step 4. Eqn. (3): Join the context_vector and rnn_output
  #     [ct; ht] shape: (batch t, value_units + query_units)
  context_and_rnn_output = tf.concat([context_vector, rnn_output], axis=-1)

  # Step 4. Eqn. (3): `at = tanh(Wc@[ct; ht])`
  attention_vector = self.Wc(context_and_rnn_output)
  shape_checker(attention_vector, ('batch', 't', 'dec_units'))

  # Step 5. Generate logit predictions:
  logits = self.fc(attention_vector)
  shape_checker(logits, ('batch', 't', 'output_vocab_size'))

  return DecoderOutput(logits, attention_weights), state


class MaskedLoss(tf.keras.losses.Loss):
  def __init__(self):
    self.name = 'masked_loss'
    self.loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

  def __call__(self, y_true, y_pred):
    shape_checker = ShapeChecker()
    shape_checker(y_true, ('batch', 't'))
    shape_checker(y_pred, ('batch', 't', 'logits'))

    # Calculate the loss for each item in the batch.
    loss = self.loss(y_true, y_pred)
    shape_checker(loss, ('batch', 't'))

    # Mask off the losses on padding.
    mask = tf.cast(y_true != 0, tf.float32)
    shape_checker(mask, ('batch', 't'))
    loss *= mask

    # Return the total.
    return tf.reduce_sum(loss)


class TrainTranslator(tf.keras.Model):
  def __init__(self, embedding_dim, units,
               input_text_processor,
               output_text_processor,
               use_tf_function=True):
    super().__init__()
    # Build the encoder and decoder
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

  def train_step(self, inputs):
    self.shape_checker = ShapeChecker()
    if self.use_tf_function:
      return self._tf_train_step(inputs)
    else:
      return self._train_step(inputs)


  def test_step(self, inputs):
    self.shape_checker = ShapeChecker()
    if self.use_tf_function:
      return self._tf_test_step(inputs)
    else:
      return self._test_step(inputs)


def _preprocess(self, input_text, target_text):
  self.shape_checker(input_text, ('batch',))
  self.shape_checker(target_text, ('batch',))

  # Convert the text to token IDs
  input_tokens = self.input_text_processor(input_text)
  target_tokens = self.output_text_processor(target_text)
  self.shape_checker(input_tokens, ('batch', 's'))
  self.shape_checker(target_tokens, ('batch', 't'))

  # Convert IDs to masks.
  input_mask = input_tokens != 0
  self.shape_checker(input_mask, ('batch', 's'))

  target_mask = target_tokens != 0
  self.shape_checker(target_mask, ('batch', 't'))

  return input_tokens, input_mask, target_tokens, target_mask

def _test_step(self, inputs):
  input_text, target_text = inputs

  (input_tokens, input_mask,
   target_tokens, target_mask) = self._preprocess(input_text, target_text)

  max_target_length = tf.shape(target_tokens)[1]

  enc_output, enc_state = self.encoder(input_tokens)
  dec_state = enc_state
  loss = tf.constant(0.0)

  for t in tf.range(max_target_length-1):
      # Pass in two tokens from the target sequence:
      # 1. The current input to the decoder.
      # 2. The target for the decoder's next prediction.
      new_tokens = target_tokens[:, t:t+2]
      y, y_pred, dec_state = self._loop_step(
                                             new_tokens, input_mask,
                                             enc_output, dec_state)
      loss = loss + self.loss(y, y_pred)
      self.test_metric.update_state(y, y_pred)

    # Average the loss over all non padding tokens.
  average_loss = loss / tf.reduce_sum(tf.cast(target_mask, tf.float32))
  average_metric = self.test_metric.result()

  return {'loss': average_loss, 'accuracy': average_metric}

def _train_step(self, inputs):
  input_text, target_text = inputs

  (input_tokens, input_mask,
   target_tokens, target_mask) = self._preprocess(input_text, target_text)

  max_target_length = tf.shape(target_tokens)[1]

  with tf.GradientTape() as tape:
    # Encode the input
    enc_output, enc_state = self.encoder(input_tokens)
    self.shape_checker(enc_output, ('batch', 's', 'enc_units'))
    self.shape_checker(enc_state, ('batch', 'enc_units'))

    # Initialize the decoder's state to the encoder's final state.
    # This only works if the encoder and decoder have the same number of
    # units.
    dec_state = enc_state
    loss = tf.constant(0.0)
    #metric = tf.constant(0.0)

    for t in tf.range(max_target_length-1):
      # Pass in two tokens from the target sequence:
      # 1. The current input to the decoder.
      # 2. The target for the decoder's next prediction.
      new_tokens = target_tokens[:, t:t+2]
      y, y_pred, dec_state = self._loop_step(
                                             new_tokens, input_mask,
                                             enc_output, dec_state)
      loss = loss + self.loss(y, y_pred)
      self.train_metric.update_state(y, y_pred)

    # Average the loss over all non padding tokens.
    average_loss = loss / tf.reduce_sum(tf.cast(target_mask, tf.float32))
    average_metric = self.train_metric.result() # / tf.reduce_sum(tf.cast(target_mask, tf.float32))

  # Apply an optimization step
  variables = self.trainable_variables
  gradients = tape.gradient(average_loss, variables)
  self.optimizer.apply_gradients(zip(gradients, variables))

  # Return a dict mapping metric names to current value
  return {'loss': average_loss, 'accuracy': average_metric}

def _train_step_(self, inputs):
    input_text, target_text = inputs

    (input_tokens, input_mask,
    target_tokens, target_mask) = self._preprocess(input_text, target_text)

    max_target_length = tf.shape(target_tokens)[1]
    loss = tf.constant(0.0)
    #seq_outputs = []
    with tf.GradientTape() as tape:

        for t in range(max_target_length-1):  #tf.range(max_target_length-1):
        #metric = tf.constant(0.0)

    # Encode the input
            enc_output, enc_state = self.encoder(input_tokens)
            self.shape_checker(enc_output, ('batch', 's', 'enc_units'))
            self.shape_checker(enc_state, ('batch', 'enc_units'))

    # Initialize the decoder's state to the encoder's final state.
    # This only works if the encoder and decoder have the same number of
    # units.
            dec_state = enc_state
            dec_state_ = enc_state
      # Pass in two tokens from the target sequence:
      # 1. The current input to the decoder.
      # 2. The target for the decoder's next prediction.
            new_tokens = target_tokens[:, t:t+2]
            y, y_pred, dec_state = self._loop_step(
                                             new_tokens, input_mask,
                                             enc_output, dec_state)
            loss = loss + self.loss(y, y_pred)
            #seq_output.append((y, y_pred))
            self.metric.update_state(y, y_pred)


    # Average the loss over all non padding tokens.
    average_loss = loss / tf.reduce_sum(tf.cast(target_mask, tf.float32))
    average_metric = self.metric.result()

  # Apply an optimization step
    variables = self.trainable_variables
    gradients = tape.gradient(average_loss, variables)
    self.optimizer.apply_gradients(zip(gradients, variables))

  # Return a dict mapping metric names to current value
    return {'loss': average_loss, 'accuracy': average_metric}


def _loop_step(self, new_tokens, input_mask, enc_output, dec_state):
  input_token, target_token = new_tokens[:, 0:1], new_tokens[:, 1:2]

  # Run the decoder one step.
  decoder_input = DecoderInput(new_tokens=input_token,
                               enc_output=enc_output,
                               mask=input_mask)

  dec_result, dec_state = self.decoder(decoder_input, state=dec_state)
  self.shape_checker(dec_result.logits, ('batch', 't1', 'logits'))
  self.shape_checker(dec_result.attention_weights, ('batch', 't1', 's'))
  self.shape_checker(dec_state, ('batch', 'dec_units'))

  # `self.loss` returns the total for non-padded tokens
  y = target_token
  y_pred = dec_result.logits

  #if loss_metric:
  #  out_eval = self.loss(y, y_pred)
  #else:
  #  self.metric.update_state(y, y_pred)
  #  out_eval = None
  return y, y_pred, dec_state
  #return out_eval, dec_state


@tf.function(input_signature=[[tf.TensorSpec(dtype=tf.string, shape=[None]),
                               tf.TensorSpec(dtype=tf.string, shape=[None])]])
def _tf_train_step(self, inputs):
  return self._train_step(inputs)


@tf.function(input_signature=[[tf.TensorSpec(dtype=tf.string, shape=[None]),
                               tf.TensorSpec(dtype=tf.string, shape=[None])]])
def _tf_test_step(self, inputs):
  return self._test_step(inputs)


class BatchLogs(tf.keras.callbacks.Callback):
  def __init__(self, key):
    self.key = key
    self.logs = []

  def on_train_batch_end(self, n, logs):
    self.logs.append(logs[self.key])


class Translator(tf.Module):

  def __init__(self, encoder, decoder, input_text_processor,
               output_text_processor):
    self.encoder = encoder
    self.decoder = decoder
    self.input_text_processor = input_text_processor
    self.output_text_processor = output_text_processor

    self.output_token_string_from_index = (
        tf.keras.layers.experimental.preprocessing.StringLookup(
            vocabulary=output_text_processor.get_vocabulary(),
            mask_token='',
            invert=True))

    # The output should never generate padding, unknown, or start.
    index_from_string = tf.keras.layers.experimental.preprocessing.StringLookup(
        vocabulary=output_text_processor.get_vocabulary(), mask_token='')
    token_mask_ids = index_from_string(['', '[UNK]', '[start]']).numpy()

    token_mask = np.zeros([index_from_string.vocabulary_size()], dtype=bool)
    token_mask[np.array(token_mask_ids)] = True
    self.token_mask = token_mask

    self.start_token = index_from_string(tf.constant('[start]'))
    self.end_token = index_from_string(tf.constant('[end]'))

def tokens_to_text(self, result_tokens):
  shape_checker = ShapeChecker()
  shape_checker(result_tokens, ('batch', 't'))
  result_text_tokens = self.output_token_string_from_index(result_tokens)
  shape_checker(result_text_tokens, ('batch', 't'))

  result_text = tf.strings.reduce_join(result_text_tokens,
                                       axis=1, separator=' ')
  shape_checker(result_text, ('batch'))

  result_text = tf.strings.strip(result_text)
  shape_checker(result_text, ('batch',))

  return result_text

def sample(self, logits, temperature):
  shape_checker = ShapeChecker()
  # 't' is usually 1 here.
  shape_checker(logits, ('batch', 't', 'vocab'))
  shape_checker(self.token_mask, ('vocab',))

  token_mask = self.token_mask[tf.newaxis, tf.newaxis, :]
  shape_checker(token_mask, ('batch', 't', 'vocab'), broadcast=True)

  # Set the logits for all masked tokens to -inf, so they are never chosen.
  logits = tf.where(self.token_mask, -np.inf, logits)

  if temperature == 0.0:
    new_tokens = tf.argmax(logits, axis=-1)
  else:
    logits = tf.squeeze(logits, axis=1)
    new_tokens = tf.random.categorical(logits/temperature,
                                        num_samples=1)

  shape_checker(new_tokens, ('batch', 't'))

  return new_tokens


def translate_unrolled(self,
                       input_text, *,
                       max_length=50,
                       return_attention=True,
                       temperature=1.0):
  batch_size = tf.shape(input_text)[0]
  input_tokens = self.input_text_processor(input_text)
  enc_output, enc_state = self.encoder(input_tokens)

  dec_state = enc_state
  new_tokens = tf.fill([batch_size, 1], self.start_token)

  result_tokens = []
  attention = []
  done = tf.zeros([batch_size, 1], dtype=tf.bool)

  for _ in range(max_length):
    dec_input = DecoderInput(new_tokens=new_tokens,
                             enc_output=enc_output,
                             mask=(input_tokens!=0))

    dec_result, dec_state = self.decoder(dec_input, state=dec_state)

    attention.append(dec_result.attention_weights)

    new_tokens = self.sample(dec_result.logits, temperature)

    # If a sequence produces an `end_token`, set it `done`
    done = done | (new_tokens == self.end_token)
    # Once a sequence is done it only produces 0-padding.
    new_tokens = tf.where(done, tf.constant(0, dtype=tf.int64), new_tokens)

    # Collect the generated tokens
    result_tokens.append(new_tokens)

    if tf.executing_eagerly() and tf.reduce_all(done):
      break

  # Convert the list of generates token ids to a list of strings.
  result_tokens = tf.concat(result_tokens, axis=-1)
  result_text = self.tokens_to_text(result_tokens)

  if return_attention:
    attention_stack = tf.concat(attention, axis=1)
    return {'text': result_text, 'attention': attention_stack}
  else:
    return {'text': result_text}

@tf.function(input_signature=[tf.TensorSpec(dtype=tf.string, shape=[None])])
def tf_translate(self, input_text):
  return self.translate(input_text)

def sort_cols(columns):
    ends = np.unique([c[-2:] for c in columns])
    new_cols = []
    for e in ends:
        for c in columns:
            if c.endswith(e):
                new_cols.append(c)
    return new_cols

def load_vectorizer(from_file):
    loaded_vectorizer_model =  tf.keras.models.load_model(from_file)
    lvocab = loaded_vectorizer_model.layers[0].get_vocabulary()
    lconfig = loaded_vectorizer_model.layers[0].get_config()
    """ PASSING THIS parameter destrois the output tensor becoming it either into a
        Ragged Tensor or an unpadded Eager Tensor. No aparent reason for that,
        so a bug in TF2.6 TextVectorization class. Delete it before continue"""
    del(lconfig['output_mode'])
    vectorizer = layers.experimental.preprocessing.TextVectorization(**lconfig)

    vectorizer.adapt(['Creating new TextVectorization for Python function'])
    vectorizer.set_vocabulary(lvocab)

    return vectorizer

def save_vectorizer(vectorizer, to_file):
    vectorizer_model = tf.keras.models.Sequential()
    vectorizer_model.add(tf.keras.Input(shape=(1,), dtype=tf.string))
    vectorizer_model.add(vectorizer)
    vectorizer_model.compile()

    vectorizer_model.save(to_file, save_format='tf')

def create_dataset(pairs):
#    inp = [inp for inp, targ in pairs]
#    targ = [targ for inp, targ in pairs]
    #inp, targ = zip(*pairs)
    BUFFER_SIZE = len(pairs)
    dataset = tf.data.Dataset.from_tensor_slices(pairs).shuffle(BUFFER_SIZE)
    return dataset.batch(batch_size)


# MAIN
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Adding optional argument
#parser.add_argument("-tf", "--trainFlag", action='store_true',
#    help = "Whether to train (unspecified is False, so giving"
#        " evaluation data is required, --testData, as train "
#        "data will be ignored.")
parser.add_argument("-s", "--seqLen", type=int,
    default=70, help = "Per-sample sequence length")
parser.add_argument("-u", "--nSteps", type=int,
    default=1024, help = "Number of hidden recurrent steps (units)")
parser.add_argument("-f", "--nFeatures", type=int,
    default=15000, help = "Maximum vocabulary size")
parser.add_argument("-b", "--batchSize", type=int,
    default=64, help = "Batch size")
parser.add_argument("-e", "--nEpochs", type=int,
    default=2, help = "Number of training epochs")
parser.add_argument("-d", "--embeddingDim", type=int,
    default=256, help = "Word embedding dimensionality")
parser.add_argument("-D", "--nDemo", type=int,
    default=100, help = "Number of predicted test samples to save as output")
#parser.add_argument("-T", "--trainData", type=str,
#    default="data/ncd/openie5/ncd_oie5_conceptnet_train.tsv",
#    help = "Training data (CSV file)")
parser.add_argument("-t", "--testData", type=str,
    default="data/ncd/openie5/ncd_oie5_conceptnet_test.tsv",
    help = "Test data (CSV file)")
parser.add_argument("-rp", "--resPath", type=str,
    #default="/media/vitrion/ST1000-VTR/",
    default=os.getcwd(),
    help = "Path where results, vectorizer and neural network models are stored.")
#parser.add_argument("-i", "--index", type=str,
#    default="023", help = "Number of experiment in the grid.")
# Read arguments from command line
args = parser.parse_args()
# Hyperparameters
sequence_length = args.seqLen
max_features = args.nFeatures
batch_size = args.batchSize
n_epochs = args.nEpochs
embedding_dim = args.embeddingDim
units = args.nSteps
# Input data
testing_data = args.testData
# Other settings
n_demo = args.nDemo
#train_flag = args.trainFlag
results_path = os.path.normpath(args.resPath) + os.sep
#idx = args.index

train_loss = BatchLogs('loss')
train_accu = BatchLogs('accuracy')
test_loss = BatchLogs('loss')
test_accu = BatchLogs('accuracy')

Decoder.call = call

TrainTranslator._preprocess = _preprocess
TrainTranslator._train_step = _train_step
TrainTranslator._test_step = _test_step
TrainTranslator._loop_step = _loop_step
TrainTranslator._tf_train_step = _tf_train_step
TrainTranslator._tf_test_step = _tf_test_step
TrainTranslator.train_metric = keras.metrics.SparseCategoricalAccuracy()
TrainTranslator.test_metric = keras.metrics.SparseCategoricalAccuracy()

Translator.tokens_to_text = tokens_to_text
Translator.sample = sample
Translator.translate = translate_unrolled
Translator.tf_translate = tf_translate

with open(testing_data) as f:
    val_text = f.readlines()

val_pairs = list(
    map(functools.partial(
        prepare_data,
        include_labels=cs_labels, all_start_end=True), val_text))

checkpoint_path = ("results"+os.sep+"CSRncdKBC-attentionGRU_"
#checkpoint_path = ("results"+os.sep+ f"attentionGRU-CSOIEGP_index-{idx}_"
    f"epochs-{n_epochs}_seqlen-{sequence_length}_maxfeat-{max_features}_batch"
    f"-{batch_size}_embdim-{embedding_dim}_steps-{units}"+os.sep+"cp.ckpt") # results/CSRncdKBC-..._steps{}/cp.ckpt
#  f"-{batch_size}_modeldim-{embedding_dim}_units-{units}"+os.sep+"cp.ckpt")
#st()
print("Working results directory: " + checkpoint_path)
checkpoint_dir = os.path.dirname(checkpoint_path)
out_dir = results_path + os.sep.join(checkpoint_path.split(os.sep)[:2]) + os.sep
# out_dir = results/CSRncdKBC-..._steps{}/
checkpoint_path = results_path + checkpoint_path

test_dataset = create_dataset(val_pairs)

strip_chars = string.punctuation
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")

input_vectorizer = load_vectorizer(results_path + "results"
        + f'{os.sep}attentionGRU-CSOIEGP_seqlen-{sequence_length}_vectorizer'
        + f'{os.sep}in_vect_model')
        #out_dir+'in_vect_model')
output_vectorizer = load_vectorizer(results_path + "results"
        + f'{os.sep}attentionGRU-CSOIEGP_seqlen-{sequence_length}_vectorizer'
        + f'{os.sep}out_vect_model')


train_translator = TrainTranslator(
    embedding_dim, units,
    input_text_processor=input_vectorizer,
    output_text_processor=output_vectorizer)

# Configure the loss and optimizer
train_translator.compile(
    optimizer=tf.optimizers.Adam(),
    loss=MaskedLoss()
)

train_translator.load_weights(checkpoint_path)

translator = Translator(
    encoder=train_translator.encoder,
    decoder=train_translator.decoder,
    input_text_processor=input_vectorizer,
    output_text_processor=output_vectorizer,
)

if not (n_demo < 0 or isinstance(n_demo, str)):
    random.shuffle(val_pairs)
    val_pairs = val_pairs[:n_demo]

inp_, targ_ = zip(*val_pairs)

results = []
logging.info("Now performing inferences using the trained model...")
try: 
    for chunk in tqdm(np.array_split(inp_, len(inp_)/batch_size),
            desc='Reasoning…', ascii=False, ncols=int(len(inp_)/batch_size)):
        result = translator.tf_translate(tf.constant(chunk))['text'].numpy()
        results.append(result.tolist())

    result = sum(results, [])

except ValueError:
    result = translator.tf_translate(tf.constant(inp_))['text'].numpy()

result = pd.DataFrame({'Subj_Pred': inp_, 'Obj': result, 'Obj_true': targ_})
result.to_csv(out_dir + 'predictions.csv')
print(result)
logging.info("See the natural language predicions written "
    f"to {out_dir + 'predictions.csv'}")
