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
from joblib import Parallel, delayed
import logging
from pdb import set_trace as st
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


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
        delayed(write_result_row)(inp, out, pmid)
            for inp, out, pmid in pairs)
    out_file.close()


def build_row_result(inp, out, pmid, out_file):
    translated = decode_sequence(inp)
    #return {'Subj_Pred': inp,
    #        'Obj': translated,
    #        'Obj_true': out,
    #        'PMID': pmid}
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

# MAIN
parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument("-mp", "--predictFlag", action='store_true',
    help = "Whether to generate triples from test and validation data.")
"""
parser.add_argument("-tf", "--trainFlag", action='store_true',
    help = "Whether to train (unspecified is False, so giving"
        " evaluation data is required, --validData, as train "
        "and test data will be ignored.")

parser.add_argument("-ev", "--evaluateFlag", action='store_true',
    help = "Whether to get evaluation metrics for validation data.")

parser.add_argument("-s", "--seqLen", type=int,
    default=30, help = "Per-sample sequence length")

parser.add_argument("-b", "--batchSize", type=int,
    default=64, help = "Batch size")

parser.add_argument("-N", "--stackSize", type=int,
    default=1, help = "Stack size of encoder/decoders")

parser.add_argument("-md", "--modelDim", type=int,
    default=512, help = "Embedding Dimensionality of input and output") #base = 512

parser.add_argument("-l", "--latentDim", type=int,
    default=2048, help = "Hidden embedding dimensionality. Inner layer"
        " dimensionality. Dense dim") #dff= 2048

parser.add_argument("-H", "--nHeads", type=int,
    default=8, help = "Number of attention heads") # h = 8
"""
parser.add_argument("-f", "--nFeatures", type=int,
    default=15000, help = "Maximum vocabulary size")

parser.add_argument("-e", "--nEpochs", type=int,
    default=100, help = "Number of training epochs (training can stop earlier"
        " as improvements do not overmoce 0.005*loss within 10 epochs.)") # base = 30

parser.add_argument("-kd", "--keyDim", type=int,
    default=0, help = "Key dimensionality; default is 0,"
        " interpreted as keyDim = modelDim/nHeads") #base = 64

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
    help = "Prefix name used for output directory naming")

parser.add_argument("-gf", "--gridFile", type=str,
    default="/home/vitrion/transformerGrid.csv",
    help="Hyperparameter grid must have the following columns:"
    " (i, stack_size, batch_size, sequence_length, "
        "model_dim, embedding_dim, latent_dim, num_heads)")

parser.add_argument("-i", "--index", type=int,
    default=0, help = "Start index")

# Read arguments from command line
args = parser.parse_args()
#logging.info("Working for:\n")
#logging.info(pd.DataFrame([(arg, getattr(args, arg)) for arg in vars(args)]))

# Hyperparameters
#train_flag = args.trainFlag
train_flag = True
#eval = args.evaluateFlag
eval = True
to_predict = args.predictFlag
"""
sequence_length = args.seqLen
batch_size = args.batchSize
n_epochs = args.nEpochs
stack_size= args.stackSize
key_dim = (int(args.modelDim/args.nHeads)
            if args.keyDim <= 0 or isinstance(args.keyDim, str)
            else args.keyDim)
model_dim = args.modelDim
latent_dim = args.latentDim
num_heads = args.nHeads
"""
max_features = args.nFeatures
n_epochs = args.nEpochs
#key_dim = (int(args.modelDim/args.nHeads)
#            if args.keyDim <= 0 or isinstance(args.keyDim, str)
#            else args.keyDim)
# Input data
training_data = args.trainData
test_data = args.testData
validation_data = args.validData
# Other settings
n_demo = args.nDemo
dataset_name = args.datasetName
test_name = args.testName

# --------------------------------- MALLA -------------------------------------

#with open('/home/vitrion/transformerGrid.csv') as f:
with open(args.gridFile) as f:
    lines = f.readlines()
lines = np.array(lines)
#onlyIdxs = [318, 319, 320, 321, 322, 323]
#lines = lines[onlyIdxs]

if args.index >= len(lines):
    sys.exit("Index out of bounds!")

metric = "accuracy"
loss = "sparse_categorical_crossentropy"

for line in lines[args.index:]:
    p = line.strip().split(',')[:8]
    # Hyperparameters
    (i, stack_size, batch_size, sequence_length,
        model_dim, embedding_dim, latent_dim, num_heads) = list(map(int, p))
    # Input data
    training_data = args.trainData
    testing_data = args.testData
    # Other settings
    n_demo = args.nDemo
    key_dim = int(model_dim/num_heads)
# -----------------------------------------------------------------------------
    if to_predict and train_flag:
        logging.warning("Train (-tf) and make predictions (-mp) flags activated"
            " require a LONG TIME if --nDemo is not set to small integer. Go to"
            " your city downtown for a cofe, return and take a sit...")

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

    logging.info(checkpoint_path)
    checkpoint_dir = os.path.dirname(checkpoint_path)
    out_dir = '/'.join(checkpoint_path.split('/')[:2]) + '/'
    vectorizer_dir = out_dir.split('_e')[0] + '_vectorizer/'
    # Create a callback that saves the model's weights

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

        if (os.path.isdir(vectorizer_dir)
                and os.path.isfile(vectorizer_dir+'in_vect_model')
                and os.path.isfile(vectorizer_dir+'out_vect_model')):
            input_vectorizer =  load_vectorizer(vectorizer_dir+'in_vect_model')
            output_vectorizer = load_vectorizer(vectorizer_dir+'out_vect_model')
        else:
            input_vectorizer = layers.experimental\
                                     .preprocessing\
                                     .TextVectorization(
                output_mode="int", max_tokens=max_features,
                # ragged=False, # only for TF v2.7
                output_sequence_length=sequence_length,
                standardize=custom_standardization)

            output_vectorizer = layers.experimental\
                                      .preprocessing\
                                      .TextVectorization(
                output_mode="int", max_tokens=max_features, # ragged=False,
                output_sequence_length=sequence_length+1,
                standardize=custom_standardization)

            input_vectorizer.adapt(train_in_texts)
            output_vectorizer.adapt(train_out_texts)

            #saving the vectorizers also
            save_vectorizer(
                vectorizer=input_vectorizer,
                to_file=vectorizer_dir+'in_vect_model')
            save_vectorizer(
                vectorizer=output_vectorizer,
                to_file=vectorizer_dir+'out_vect_model')
        train_ds = make_dataset(train_pairs)
        test_ds = make_dataset(
            test_pairs, include_pmid=pmid_val_labels)
        val_ds = make_dataset(
            val_pairs, include_pmid=pmid_val_labels)

    else:
        if os.path.isdir(out_dir):
            logging.info("Loading Vectorizers")
        else:
            logging.error("NO model trained and directory with specified parameters"
                " exists in: {}".format(out_dir))
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
        input_vectorizer =  load_vectorizer(out_dir+'in_vect_model')
        output_vectorizer = load_vectorizer(out_dir+'out_vect_model')

        val_ds = make_dataset(val_pairs, include_pmid=pmid_val_labels)

    max_vocab = max([
            len(input_vectorizer.get_vocabulary()),
            len(output_vectorizer.get_vocabulary())])
    if max_features > max_vocab:
        max_features = max_vocab


    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=10,
                                                    min_delta=0.005,
                                                    mode='auto',
                                                    restore_best_weights=True,
                                                    verbose=1)
    transformer = build_transformer_encodec(
        sequence_length,
        max_features,
        model_dim,
        stack_size,
        latent_dim,
        num_heads,
        key_dim)
    transformer.summary()
    # Adam optimizer is crucial for training Transformers. Other methods
    # induce high instability
    transformer.compile(optimizer="Adam", loss=loss, metrics=[metric])

    if train_flag:
        logging.info("Training Transformer Semantic EncoDec")
        history = transformer.fit(train_ds,
            epochs=n_epochs,
            validation_data=test_ds,
                callbacks=[ #cp_callback,
                            es_callback])
        logging.info("TRAINED!!")
        rdf = pd.DataFrame(history.history)
        rdf.to_csv(out_dir + "history.csv")

        fig, axes = plt.subplots(2, 1)
        rdf[sort_cols(rdf.columns)].iloc[:, :2].plot(ax=axes[0])
        axes[0].grid(b=True,which='major',axis='both',linestyle='--')
        rdf[sort_cols(rdf.columns)].iloc[:, 2:].plot(ax=axes[1])
        axes[1].grid(b=True,which='major',axis='both',linestyle='--')
        plt.savefig(out_dir + 'history_plot.pdf')
        """ Notes about saving the model weights:
        - must be the same paramers when you load the model
        - if you specify a directory, you will save them without a prefix
        """
        logging.info("Saving learned weights to {}\n".format(
            out_dir+'transformer_model_weights/model'))
        transformer.save_weights(out_dir+'transformer_model_weights/model')

    else:
        # vectorizers have been loaded previously
        transformer.load_weights(out_dir+'transformer_model_weights/model')

    val_file = out_dir + 'evaluation_on_validation_data.txt'
    out_phr_vocab = output_vectorizer.get_vocabulary()
    out_phr_index_lookup = dict(zip(range(len(out_phr_vocab)), out_phr_vocab))
    max_decoded_sentence_length = 20

    if train_flag:
        if to_predict:
            logging.info("OBTAINING PREDICTIONS FOR TEST SET")
            """ Generate predictions for test set """
            if not (n_demo < 0 or isinstance(n_demo, str)):
                random.shuffle(test_pairs)
                test_pairs = test_pairs[:n_demo]

            save_predictions(pairs=test_pairs,
                to_file_path=out_dir + 'test_predictions.csv')
            logging.info("KBC FOR TEST SET WRITTEN TO {}".format(
                out_dir + 'test_predictions.csv'))

            logging.info("OBTAINING PREDICTIONS FOR VALIDATION SET")
            if not (n_demo < 0 or isinstance(n_demo, str)):
                random.shuffle(val_pairs)
                val_pairs = val_pairs[:n_demo]

            save_predictions(pairs=val_pairs,
                to_file_path=out_dir + 'val_predictions.csv')
            logging.info("KBC FOR VALIDATION SET WRITTEN TO {}".format(
                out_dir + 'val_predictions.csv'))

        if eval:
            logging.info(
                'Validating Transformer Semantic EncoDec to {}'.format(val_file))
            with open(val_file, 'w') as ev:
                val_loss, val_acc = transformer.evaluate(val_ds)
                line = 'Validation_loss: {}\nValidation_acc: {}\n'.format(
                    val_loss, val_acc)
                ev.write(line)
            logging.info("Model evaluated. See results in {}".format(val_file))
    else:
        if to_predict:
            logging.info("OBTAINING PREDICTIONS FOR VALIDATION SET")
            if not (n_demo < 0 or isinstance(n_demo, str)):
                random.shuffle(val_pairs)
                val_pairs = val_pairs[:n_demo]

            save_predictions(pairs=val_pairs,
                to_file_path=out_dir + 'val_predictions.csv')
            logging.info("KBC FOR VALIDATION SET WRITTEN TO {}".format(
                out_dir + 'val_predictions.csv'))
        if eval:
            logging.info(
                'Validating Transformer Semantic EncoDec to {}'.format(val_file))
            with open(val_file, 'w') as ev:
                val_loss, val_acc = transformer.evaluate(val_ds)
                line = 'Validation_loss: {}\nValidation_acc: {}\n'.format(
                    val_loss, val_acc)
                ev.write(line)
            logging.info("Model evaluated. See results in {}".format(val_file))

    logging.info("All tasks finished. See results in or load trained model "
        "from directory {}\n".format(out_dir))