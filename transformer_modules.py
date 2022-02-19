import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import  plot_model
import numpy as np
import json
import string, re, os
import logging
import functools
from joblib import Parallel, delayed

from pdb import set_trace as st


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p')

config_file = 'predictor_configuration.json'
strip_chars = string.punctuation
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")

cs_labels = False
pmid_val_labels = True
metric = "accuracy"
loss = "sparse_categorical_crossentropy"

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
        super(TransformerEncoder, self).__init__(**kwargs)
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
            padding_mask = tf.cast(
                mask[:, tf.newaxis, tf.newaxis, :], dtype="int32")
        attention_output = self.attention(
            query=inputs, value=inputs, key=inputs, attention_mask=padding_mask
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
        super(TransformerDecoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            value_dim=key_dim,
            output_shape=embed_dim
        )
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

def build_transformer_encodec(sequence_length, max_features, model_dim, stack_size, latent_dim, num_heads, key_dim):
    encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="encoder_inputs")
    x = PositionalEmbedding(sequence_length, max_features, embed_dim=model_dim)(encoder_inputs) #dmodel
    for n in range(0, stack_size-1):
        x = TransformerEncoder(embed_dim=model_dim,dense_dim=latent_dim, num_heads=num_heads,key_dim=key_dim)(x)
    encoder_outputs = TransformerEncoder(embed_dim=model_dim, dense_dim=latent_dim, num_heads=num_heads,key_dim=key_dim)(x)
    encoder = keras.Model(encoder_inputs, encoder_outputs)

    decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="decoder_inputs")
    encoded_seq_inputs = keras.Input(shape=(None, model_dim), name="decoder_state_inputs")
    x = PositionalEmbedding(sequence_length, max_features, embed_dim=model_dim)(decoder_inputs) #d model

    for n in range(0, stack_size):
        x = TransformerDecoder(embed_dim=model_dim,latent_dim=latent_dim, num_heads=num_heads,key_dim=key_dim)(x, encoded_seq_inputs)
    x = layers.Dropout(0.5)(x)
    decoder_outputs = layers.Dense(max_features, activation="softmax")(x)
    decoder = keras.Model([decoder_inputs, encoded_seq_inputs], decoder_outputs)

    decoder_outputs = decoder([decoder_inputs, encoder_outputs])
    transformer = keras.Model(
        [encoder_inputs, decoder_inputs], decoder_outputs, name="transformer"
    )

    return transformer

def save_predictions(pairs, to_file_path):
    logging.info("OBTAINING PREDICTIONS TO {}".format(to_file_path))
    """ Generate predictions for test set """
    out_file = open(to_file_path, 'w')
    write_result_row = functools.partial(build_row_result, out_file=out_file)
    out_file.write('\t'.join(['Subj_Pred', 'Obj', 'Obj_true\n']))
    
    Parallel(n_jobs=1)(
        delayed(write_result_row)(inp, out)
            for inp, out in pairs)
    out_file.close()


def build_row_result(inp, out, out_file):
    translated = decode_sequence(inp)
    line = [inp, translated, out]
    out_file.write('\t'.join(line) + '\n')

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


def get_config(config_file):
    with open(config_file) as f:
        config = f.read()
        js = json.loads(config)

    dataset_name = js["dataset_name"]
    max_features = js["max_features"]
    model_dim = int(js["model_dim"])
    num_heads = int(js["num_heads"])
    n_epochs = int(js["n_epochs"])
    stack_size = int(js["stack_size"])
    sequence_length = int(js["sequence_length"])
    max_features = int(js["max_features"])
    batch_size = int(js["batch_size"])
    model_dim = int(js["model_dim"])
    latent_dim = int(js["latent_dim"])
    num_heads = int(js["num_heads"])
    key_dim = int(float(model_dim)/float(num_heads))

    checkpoint_path = ("results_final/{}-transformer_epochs"
            "-{}_stackSize-{}_seqlen-{}_maxfeat-{}_batch-{}_keydim"
            "-{}_modeldim-{}_latent-{}_heads-{}/cp.ckpt".format(
            dataset_name,
            n_epochs,
            stack_size,
            sequence_length,
            max_features,
            batch_size,
            key_dim,
            model_dim,
            latent_dim,
            num_heads)
        )

    checkpoint_dir = os.path.dirname(checkpoint_path)
    logging.info(checkpoint_path)
    out_dir = checkpoint_dir + '/'

    return (dataset_name,
            n_epochs,
            stack_size,
            sequence_length,
            max_features,
            batch_size,
            key_dim,
            model_dim,
            latent_dim,
            num_heads,
            out_dir
            )

def get_max_vocab():
    return max([
        len(input_vectorizer.get_vocabulary()),
        len(output_vectorizer.get_vocabulary())])

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
            out_dir
            ) = get_config(config_file)
input_vectorizer =  load_vectorizer(out_dir+'in_vect_model')
output_vectorizer = load_vectorizer(out_dir+'out_vect_model')

max_vocab = get_max_vocab()

if max_features != max_vocab:
    max_features = max_vocab

transformer = build_transformer_encodec(
        sequence_length,
        max_features,
        model_dim,
        stack_size,
        latent_dim,
        num_heads,
        key_dim)
transformer.summary()
transformer.compile(optimizer="Adam", loss=loss, metrics=[metric])

transformer.load_weights(out_dir+'transformer_model_weights/model')

out_phr_vocab = output_vectorizer.get_vocabulary()
out_phr_index_lookup = dict(zip(range(len(out_phr_vocab)), out_phr_vocab))
max_decoded_sentence_length = 20

