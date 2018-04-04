from fr.tagc.predaction.generatedata import generate_random_seqs
from fr.tagc.predaction.visualizer import visualize_filters
from fr.tagc.predaction.features import get_compute_matrix
from fr.tagc.predaction.features import get_sequences
from fr.tagc.predaction.features import get_matrices

from multiprocessing_on_dill import Pool

from keras.utils import to_categorical
from keras.utils import plot_model

from keras.optimizers import adam
from keras.optimizers import SGD

from keras.layers import GlobalMaxPooling2D
from keras.layers import MaxPooling2D
from keras.layers import concatenate
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Input
from keras.layers import Dense

from keras.models import Model

from keras import regularizers

from numpy import expand_dims
from numpy import swapaxes
from numpy import fromiter
from numpy import matrix
from numpy import array
from numpy import stack

import pandas
import numpy

LEARNING_RATE = 0.01
FILTER_REG = 0.01
NB_FILTERS = 1
FILTER_HEIGHT = 6

SEQ_LEN = 8
PATCH_SIZE = 5
BATCH_SIZE = 500

RANDOM = False

def create_model():
    inputsA = Input(shape=(None, 21, 1,))
    inputsB = Input(shape=(None, 21, 1,))
    branchA, branchB = create_branches(inputsA, inputsB)
    merge = concatenate([branchA, branchB])
    predictions = Dense(2, activation="softmax")(merge)
    model = Model(inputs=[inputsA, inputsB], outputs=predictions)
    opt = SGD(lr=LEARNING_RATE)
    model.compile(optimizer=opt,
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


def create_branches(inputsA, inputsB):
    filter_reg = regularizers.l2(FILTER_REG)
    encode_reg = regularizers.l2(ENCODE_REG)
    layer = Conv2D(NB_FILTERS, (FILTER_HEIGHT, 21), activation="relu", kernel_regularizer=filter_reg)
    branchA, branchB = symetric_layer(inputsA, inputsB, layer)
    layer = GlobalMaxPooling2D()
    branchA, branchB = symetric_layer(branchA, branchB, layer)
    return branchA, branchB


def symetric_layer(a, b, layer):
    left = layer(a)
    right = layer(b)
    return left, right


def status_to_categories(categories):
    cats = [1 if x == "YES" else 0 for x in categories]
    cats = to_categorical(cats, num_classes=2)
    return cats


def read_data_frame(file_path):
    df = pandas.read_csv(file_path, sep = "\t")
    pairs = df.iloc[:, :-1]
    categories = status_to_categories(df.iloc[:,-1])
    proteins = set([row.protA for row in pairs.itertuples()])
    proteins.update(set([row.protB for row in pairs.itertuples()]))
    return pairs, categories, proteins


if __name__ == "__main__":
    pairs, categories, proteins = read_data_frame("data/interaction.tsv")
    sequences = get_sequences("data/sequences.fasta", proteins)
    sequencesA = (sequences[row.protA] for row in pairs.itertuples())
    sequencesB = (sequences[row.protB] for row in pairs.itertuples())
    generator = generate_sequences(sequencesA, sequencesB, categories, 1)
    model = create_model()
    model.fit_generator(generator, steps_per_epoch=20, epochs=20)
