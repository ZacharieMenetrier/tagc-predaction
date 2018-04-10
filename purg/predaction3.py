import fr.tagc.predaction.sequence as sequence
import fr.tagc.predaction.parser as parse
from visualizer import visualize_filters
from keras.utils import plot_model
from keras.optimizers import adam
from keras.models import Model
from keras import regularizers
import keras.layers as layers
import numpy



def create_model(params):

    def create_branches(a, b):

        def sym(l, x, y):
            a = l(x)
            b = l(y)
            return a, b

        a, b = sym(layers.LSTM(params["units"]), a, b)
        for dim in params["d_encode"]:
            a, b = sym(layers.Dense(dim), a, b)
            a, b = sym(layers.Dropout(params["p_dropout"]), a, b)
        return a, b

    inputsA = layers.Input(shape=(None, 100,))
    inputsB = layers.Input(shape=(None, 100,))
    branchA, branchB = create_branches(inputsA, inputsB)
    layer = layers.concatenate([branchA, branchB])
    for dim in params["d_intermediate"]:
        layer = layers.Dense(dim)(layer)
        layer = layers.Dropout(params["p_dropout"])(layer)
    outputs = layers.Dense(2, activation="softmax")(layer)
    model = Model(inputs=[inputsA, inputsB], outputs=outputs)
    opt = adam(lr=params["learning_rate"])
    model.compile(optimizer=opt, loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


def generate_batches(arrayA, arrayB, categories):
    for eltA, eltB, eltC in zip(arrayA, arrayB, categories):
        yield [numpy.array([eltA]), numpy.array([eltB])], numpy.array([eltC])



if __name__ == "__main__":

    ############################################################################
    units = 256
    d_encode = [128, 128]
    d_intermediate = [128]
    p_dropout = 0.3
    learning_rate = 0.1
    ############################################################################
    model = create_model(locals())
    plot_model(model, "results/filters/model.png")
    pairs, categories, proteins = parse.read_data_frame("data/intersectome.tsv")
    # sequences = parse.get_sequences("data/sequences.fasta", proteins)
    sequences = parse.get_sequences_from_tsv("data/sequences.tsv", proteins)
    trans_function = sequence.get_compute_embedded_matrix("data/protvec.tsv")
    matrices = sequence.transform_sequences(sequences, trans_function)
    matricesA = (matrices[row.protA] for row in pairs.itertuples())
    matricesB = (matrices[row.protB] for row in pairs.itertuples())
    generator = generate_batches(matricesA, matricesB, categories)
    model.fit_generator(generator, steps_per_epoch=400, epochs=5)
