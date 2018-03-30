from fr.tagc.predaction.visualizer import visualize_filters
from fr.tagc.predaction.features import get_matrices
from fr.tagc.predaction.features import get_sequences
from multiprocessing_on_dill import Pool
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import concatenate
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


def create_model(max_len):
    inputs1, branch1 = create_branch(max_len)
    inputs2, branch2 = create_branch(max_len)
    merge = concatenate([branch1, branch2])
    intermediate = Dense(12, activation="relu")(merge)
    predictions = Dense(2, activation="softmax")(intermediate)
    model = Model(inputs=[inputs1, inputs2], outputs=predictions)
    sgd = SGD(lr=0.01, clipnorm=0.1)
    model.compile(optimizer=sgd,
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


def create_branch(max_len):
    reg = regularizers.l2(0.01)
    inputs = Input(shape=(None, 7, 1,))
    conv = Conv2D(20, (8, 7), activation="relu", kernel_regularizer=reg)(inputs)
    pool = MaxPooling2D(pool_size = (2,1))(conv)
    pool = GlobalMaxPooling2D()(pool)
    # flat = Flatten()(conv)
    branch = Dense(20, activation="relu", activity_regularizer=reg)(pool)
    return inputs, branch


def get_arrays(pairs, sequences, extend=True, fmap=map):
    matrices = get_matrices(sequences, extend, fmap)
    array1 = [matrices[row.protA] for row in pairs.itertuples()]
    array2 = [matrices[row.protB] for row in pairs.itertuples()]
    return array1, array2



def uncompress_matrices(full_array1, full_array2, full_categories, batch_size):
    while True:
        length = len(full_array1)
        get_batch = lambda a, i : a[i:min(i + batch_size, length)]
        uncompress = lambda matrices : [m.toarray() for m in matrices]
        for i in range(0, length, batch_size):
            matrices1 = expand_dims(stack(uncompress(get_batch(full_array1, i))), 4)
            matrices2 = expand_dims(stack(uncompress(get_batch(full_array2, i))), 4)
            categories = get_batch(full_categories, i)
            yield [matrices1, matrices2], categories


def read_data_frame(file_path):
    df = pandas.read_csv(file_path, sep = "\t")
    pairs = df.iloc[:, :-1]
    categories = [1 if x == "YES" else 0 for x in df.iloc[:,-1]]
    categories = to_categorical(categories, num_classes=2)
    proteins = set([row.protA for row in pairs.itertuples()])
    proteins.update(set([row.protB for row in pairs.itertuples()]))
    return pairs, categories, proteins


if __name__ == "__main__":
    import sys
    p = Pool(7)
    pairs, categories, proteins = read_data_frame("data/trivial_data.tsv")
    labels = numpy.array(numpy.argmax(categories, axis=1))
    sequences = get_sequences("data/small_sequences.fasta", proteins)
    array1, array2 = get_arrays(pairs, sequences, False, p.map)
    generator = uncompress_matrices(array1, array2, categories, 1)
    max_len = len(max(sequences.values(), key=len))
    model = create_model(max_len)
    visualize_filters(model, 2, 20, "results/filters/init1.png", "Reds")
    visualize_filters(model, 3, 20, "results/filters/init2.png", "Reds")
    model.fit_generator(generator, steps_per_epoch=100, epochs=10)
    visualize_filters(model, 2, 20, "results/filters/fit1.png", "Greens")
    visualize_filters(model, 3, 20, "results/filters/fit2.png", "Greens")
    preds = model.predict_generator(generator, steps=len(pairs))
    preds = numpy.array(numpy.argmax(preds, axis = 1))
    print(pandas.crosstab(preds, labels))
