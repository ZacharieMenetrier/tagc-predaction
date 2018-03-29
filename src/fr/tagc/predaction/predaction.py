from fr.tagc.predaction.sequence import get_matrices
from fr.tagc.predaction.sequence import get_sequences
from multiprocessing_on_dill import Pool
from keras.utils import to_categorical
from keras.layers import concatenate
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Input
from keras.layers import Dense
from keras.models import Model
from numpy import fromiter
from numpy import array
from numpy import matrix
from numpy import stack
from numpy import swapaxes
from numpy import expand_dims
import pandas


def create_model(max_len):
    inputs1, branch1 = create_branch(max_len)
    inputs2, branch2 = create_branch(max_len)
    merge = concatenate([branch1, branch2])
    predictions = Dense(2, activation="softmax")(merge)
    model = Model(inputs=[inputs1, inputs2], outputs=predictions)
    model.compile(optimizer="rmsprop",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


def create_branch(max_len):
    inputs = Input(shape=(max_len, 21, 1,))
    conv = Conv2D(5, (21,21))(inputs)
    flat = Flatten()(conv)
    branch = Dense(64, activation="relu")(flat)
    return inputs, branch


def get_arrays(pairs, sequences, fmap=map):
    matrices = get_matrices(sequences, fmap=fmap)
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
    pairs, categories, proteins = read_data_frame("data/interaction.tsv")
    sequences = get_sequences("data/sequences.fasta", proteins)
    array1, array2 = get_arrays(pairs, sequences, p.map)
    generator = uncompress_matrices(array1, array2, categories, 100)
    max_len = len(max(sequences.values(), key=len))
    for g, c in generator:
        print(g[0].shape, c.shape)
        break
    model = create_model(max_len)
    model.fit_generator(generator, steps_per_epoch=100, epochs=100)
