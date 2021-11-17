import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import  plot_model
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import string, re, os
import math, random, functools
#from pdb import set_trace as st
import argparse


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


def make_dataset(pairs):
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
            print("Invalid input to model: Invalid argument: slice index {} of dimension 1 out of bounds.".format(i))
            continue
        try:
            sampled_token = out_phr_index_lookup[sampled_token_index]
        except KeyError:
            print("KeyError: {}; output vocabulary length: {}".format(
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
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, mask=None):
        if mask is not None:
            padding_mask = tf.cast(
                mask[:, tf.newaxis, tf.newaxis, :], dtype="int32")
        attention_output = self.attention(
            query=inputs, value=inputs, key=inputs, attention_mask=padding_mask
        )
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
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
        return tf.math.not_equal(inputs, 0)


class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [layers.Dense(
                latent_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
            padding_mask = tf.minimum(padding_mask, causal_mask)

        attention_output_1 = self.attention_1(
            query=inputs, value=inputs, key=inputs, attention_mask=causal_mask
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1),
             tf.constant([1, 1],
             dtype=tf.int32)],
            axis=0,
        )

        return tf.tile(mask, mult)


# MAIN
parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument("-s", "--seqLen", type=int,
    default=70, help = "Per-sample sequence length")
parser.add_argument("-f", "--nFeatures", type=int,
    default=15000, help = "Maximum vocabulary size")
parser.add_argument("-b", "--batchSize", type=int,
    default=64, help = "Batch size")
parser.add_argument("-e", "--nEpochs", type=int,
    default=2, help = "Number of training epochs")
parser.add_argument("-d", "--embeddingDim", type=int,
    default=256, help = "Word embedding dimensionality")
parser.add_argument("-l", "--latentDim", type=int,
    default=2048, help = "Hidden embedding dimensionality")
parser.add_argument("-H", "--nHeads", type=int,
    default=8, help = "Number of attention heads")
parser.add_argument("-D", "--nDemo", type=int,
    default=100, help = "Number of predicted test samples to save as output")
parser.add_argument("-T", "--trainData", type=str,
    default="data/ncd/openie5/ncd_oie5_conceptnet_train.tsv",
    help = "Training data (CSV file)")
parser.add_argument("-t", "--testData", type=str,
    default="data/ncd/openie5/ncd_oie5_conceptnet_test.tsv",
    help = "Test data (CSV file)")

# Read arguments from command line
args = parser.parse_args()
# Hyperparameters
sequence_length = args.seqLen
max_features = args.nFeatures
batch_size = args.batchSize
n_epochs = args.nEpochs
embedding_dim = args.embeddingDim
latent_dim = args.latentDim
num_heads = args.nHeads
# Input data
training_data = args.trainData
testing_data = args.testData
# Other settings
n_demo = args.nDemo

metric = "accuracy"
loss = "sparse_categorical_crossentropy"

with open(training_data) as f:
    train_text = f.readlines()

with open(testing_data) as f:
    val_text = f.readlines()


train_pairs = list(
    map(functools.partial(prepare_data, include_labels=cs_labels), train_text))
val_pairs = list(
    map(functools.partial(prepare_data, include_labels=cs_labels), val_text))

strip_chars = string.punctuation
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")

input_vectorizer = layers.experimental.preprocessing.TextVectorization(
    output_mode="int", max_tokens=max_features,
    output_sequence_length=sequence_length, standardize=custom_standardization)

output_vectorizer = layers.experimental.preprocessing.TextVectorization(
    output_mode="int", max_tokens=max_features,
    output_sequence_length=sequence_length+1,
    standardize=custom_standardization)

train_in_texts = [pair[0] for pair in train_pairs]
if cs_labels:
    train_out_texts = [pair[1][0] for pair in train_pairs]
else:
    train_out_texts = [pair[1] for pair in train_pairs]

input_vectorizer.adapt(train_in_texts)
output_vectorizer.adapt(train_out_texts)

max_vocab = max([
        len(input_vectorizer.get_vocabulary()),
        len(output_vectorizer.get_vocabulary())])
if max_features > max_vocab:
    max_features = max_vocab

train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)

checkpoint_path = ("results/CSRncdKBC-transformer_epochs"
    "-{}_seqlen-{}_maxfeat-{}_batch-{}_embdim-{}_latent-{}_heads-{}/cp.ckpt".format(
    n_epochs,
    sequence_length,
    max_features,
    batch_size,
    embedding_dim,
    latent_dim,
    num_heads
))
print(checkpoint_path)
checkpoint_dir = os.path.dirname(checkpoint_path)
out_dir = '/'.join(checkpoint_path.split('/')[:2]) + '/'
# Create a callback that saves the model's weights
cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="encoder_inputs")
x = PositionalEmbedding(sequence_length, max_features, embedding_dim)(encoder_inputs)
encoder_outputs = TransformerEncoder(embedding_dim, latent_dim, num_heads)(x)
encoder = keras.Model(encoder_inputs, encoder_outputs)

decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="decoder_inputs")
encoded_seq_inputs = keras.Input(shape=(None, embedding_dim), name="decoder_state_inputs")
x = PositionalEmbedding(sequence_length, max_features, embedding_dim)(decoder_inputs)
x = TransformerDecoder(embedding_dim, latent_dim, num_heads)(x, encoded_seq_inputs)
x = layers.Dropout(0.5)(x)
decoder_outputs = layers.Dense(max_features, activation="softmax")(x)
decoder = keras.Model([decoder_inputs, encoded_seq_inputs], decoder_outputs)

decoder_outputs = decoder([decoder_inputs, encoder_outputs])
transformer = keras.Model(
    [encoder_inputs, decoder_inputs], decoder_outputs, name="transformer"
)

transformer.summary()

transformer.compile("rmsprop", loss=loss, metrics=[metric])

history = transformer.fit(train_ds,
	epochs=n_epochs,
	validation_data=val_ds,
        callbacks=[cp_callback])

os.listdir(checkpoint_dir)

rdf = pd.DataFrame(history.history)
rdf.to_csv(out_dir + "history.csv")

fig, axes = plt.subplots(2, 1)
rdf[sort_cols(rdf.columns)].iloc[:, :2].plot(ax=axes[0])
rdf[sort_cols(rdf.columns)].iloc[:, 2:].plot(ax=axes[1])
plt.savefig(out_dir + 'history_plot.pdf')

#plot_model(transformer, to_file=out_dir + "architecture.pdf", show_shapes=True)

out_phr_vocab = output_vectorizer.get_vocabulary()
out_phr_index_lookup = dict(zip(range(len(out_phr_vocab)), out_phr_vocab))
max_decoded_sentence_length = 20

if not (n_demo < 0 or isinstance(n_demo, str)):
    random.shuffle(val_pairs)
    val_pairs = val_pairs[:n_demo]

test_result = []
for inp, out in val_pairs:
    translated = decode_sequence(inp)
    test_result.append({'Subj_Pred': inp,
                        'Obj': translated,
                        'Obj_true': out})

out_df = pd.DataFrame(test_result)
out_df.to_csv(out_dir + 'predictions.csv')
print(out_df)
