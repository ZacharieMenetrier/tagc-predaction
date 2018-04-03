from fr.tagc.predaction.generatedata import generate_random_seqs
from fr.tagc.predaction.visualizer import visualize_filters
from fr.tagc.predaction.features import get_compute_matrix
from fr.tagc.predaction.features import get_sequences
from fr.tagc.predaction.features import get_matrices
from keras.layers import GlobalMaxPooling2D
from multiprocessing_on_dill import Pool
from keras.utils import to_categorical
from keras.layers import MaxPooling2D
from keras.layers import concatenate
from keras.optimizers import adam
from keras.optimizers import SGD
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

INTERMEDIATE_DIM = 4
LEARNING_RATE = 0.01
DROPOUT_PROB = 0.3
FILTER_REG = 0.001
ENCODE_REG = 0.01
NB_FILTERS = 1
ENCODE_DIM = 2
FILTER_HEIGHT = 20

def create_model():
    inputs1 = Input(shape=(200, 21, 1,))
    inputs2 = Input(shape=(200, 21, 1,))
    branch = create_branch()
    encodedA = branch(inputs1)
    encodedB = branch(inputs2)
    merge = concatenate([encodedA, encodedB])
    intermediate = Dense(INTERMEDIATE_DIM, activation="relu", )(merge)
    drop = Dropout(DROPOUT_PROB)(intermediate)
    predictions = Dense(2, activation="softmax")(drop)
    model = Model(inputs=[inputs1, inputs2], outputs=predictions)
    opt = adam(lr=LEARNING_RATE)
    model.compile(optimizer=opt,
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


def create_branch():
    reg_filter = regularizers.l2(FILTER_REG)
    reg_encode = regularizers.l2(ENCODE_REG)
    conv = Conv2D(NB_FILTERS, (FILTER_HEIGHT, 21), activation="relu")
    pool = MaxPooling2D(pool_size = (2,1))(conv)
    pool = GlobalMaxPooling2D()(pool)
    branch = Dense(ENCODE_DIM, activation="relu")(pool)
    drop = Dropout(DROPOUT_PROB)(branch)
    return drop


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


def generate_trivial_data(batch_size, n):
    random_sequences = generate_random_seqs(nb_pairs_of_seq_pairs=n)
    random_sequences = [elt for sublist in random_sequences for elt in sublist]
    length = len(random_sequences)
    compute_matrix = get_compute_matrix()
    while True:
        get_batch = lambda a, i : a[i:min(i + batch_size, length)]
        for i in range(0, length, batch_size):
            sequences = get_batch(random_sequences, batch_size)
            sequencesA = (seq.seqA for seq in sequences)
            sequencesB = (seq.seqB for seq in sequences)
            categories = status_to_categories(seq.status for seq in sequences)
            matrices1 = uncompress_matrices(map(compute_matrix, sequencesA))
            matrices2 = uncompress_matrices(map(compute_matrix, sequencesB))
            yield [matrices1, matrices2], categories


if __name__ == "__main__":
    import sys
    # p = Pool(7)
    # pairs, categories, proteins = read_data_frame("data/trivial_data.tsv")
    # labels = numpy.array(numpy.argmax(categories, axis=1))
    # sequences = get_sequences("data/small_sequences.fasta", proteins)
    # array1, array2 = get_arrays(pairs, sequences, False, p.map)
    # generator = generate_matrices(array1, array2, categories, 1)
    generator = generate_trivial_data(100, 100)
    # max_len = len(max(sequences.values(), key=len))
    model = create_model()
    visualize_filters(model, 2, NB_FILTERS, "results/filters/init1.png", "Reds")
    visualize_filters(model, 3, NB_FILTERS, "results/filters/init2.png", "Reds")
    model.fit_generator(generator, steps_per_epoch=20, epochs=30)
    visualize_filters(model, 2, NB_FILTERS, "results/filters/fit1.png", "Greens")
    visualize_filters(model, 3, NB_FILTERS, "results/filters/fit2.png", "Greens")
