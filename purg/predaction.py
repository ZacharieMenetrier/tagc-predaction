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


def generate_batches(arrayA, arrayB, categories, batch_size, evaluate):
    batchA, batchB, batchC = (), (), ()
    a = numpy.array
    batch = 0
    for eltA, eltB, eltC in zip(arrayA, arrayB, categories):
        eltA = evaluate(eltA); eltB = evaluate(eltB)
        if batch >= batch_size and batchA:
            batch = 0
            yield [a(batchA), a(batchB)], a(batchC)
            batchA, batchB, batchC = (), (), ()
        batchA += (eltA,); batchB += (eltB,); batchC += (eltC,)
        batch += 1

def get_arrays(pairs, sequences, extend=True, fmap=map):
    matrices = get_matrices(sequences, extend, fmap)
    array1 = [matrices[row.protA] for row in pairs.itertuples()]
    array2 = [matrices[row.protB] for row in pairs.itertuples()]
    return array1, array2


def generate_matrices(full_array1, full_array2, full_categories, batch_size):
    while True:
        length = len(full_array1)
        get_batch = lambda a, i : a[i:min(i + batch_size, length)]
        uncompress = lambda matrices : [m.toarray() for m in matrices]
        for i in range(0, length, batch_size):
            matrices1 = uncompress_matrices(get_batch(full_array1, i))
            matrices2 = uncompress_matrices(get_batch(full_array2, i))
            categories = get_batch(full_categories, i)
            yield [matrices1, matrices2], categories


def uncompress_matrices(matrices):
    mats = array([m.toarray() for m in matrices])
    mats = expand_dims(stack(mats), 4)
    return mats


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


def generate_trivial_data(batch_size):
    while True:
        n_pairs_sequences = int(batch_size / 2)
        sequences = generate_random_seqs(SEQ_LEN, PATCH_SIZE, n_pairs_sequences, RANDOM)
        sequences = [elt for sublist in sequences for elt in sublist]
        compute_matrix = get_compute_matrix()
        sequencesA = (seq.seqA for seq in sequences)
        sequencesB = (seq.seqB for seq in sequences)
        categories = status_to_categories(seq.status for seq in sequences)
        matrices1 = uncompress_matrices(map(compute_matrix, sequencesA))
        matrices2 = uncompress_matrices(map(compute_matrix, sequencesB))
        yield [matrices1, matrices2], categories


if __name__ == "__main__":
    # p = Pool(7)
    # pairs, categories, proteins = read_data_frame("data/interaction.tsv")
    # sequences = get_sequences("data/sequences.fasta", proteins)
    # array1, array2 = get_arrays(pairs, sequences, False, p.map)
    # generator = generate_matrices(array1, array2, categories, 1)
    generator = generate_trivial_data(BATCH_SIZE)
    model = create_model()
    visualize_filters(model, 2, NB_FILTERS, "results/filters/init.png", "Reds")
    model.fit_generator(generator, steps_per_epoch=20, epochs=20)
    visualize_filters(model, 2, NB_FILTERS, "results/filters/fit.png", "Greens")
