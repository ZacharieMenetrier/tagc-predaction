import fr.tagc.predaction.sequence as sequence
import fr.tagc.predaction.parser as parse
from visualizer import visualize_filters
from keras.utils import plot_model
from keras.optimizers import SGD
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
        a, b = sym(layers.Conv2D(params["n_filter"], (params["h_filter"], 100)), a, b)
        a, b = sym(layers.GlobalMaxPooling2D(), a, b)
        for dim in params["d_encode"]:
            a, b = sym(layers.Dense(dim), a, b)
            a, b = sym(layers.Dropout(params["p_dropout"]),a , b)
        return a, b

    inputsA = layers.Input(shape=(None, 100, 1,))
    inputsB = layers.Input(shape=(None, 100, 1,))
    branchA, branchB = create_branches(inputsA, inputsB)
    layer = layers.concatenate([branchA, branchB])
    for dim in params["d_intermediate"]:
        layer = layers.Dense(dim)(layer)
        layer = layers.Dropout(params["p_dropout"])(layer)
    outputs = layers.Dense(2, activation="softmax")(layer)
    model = Model(inputs=[inputsA, inputsB], outputs=outputs)
    opt = SGD(lr=params["learning_rate"])
    model.compile(optimizer=opt, loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model



def generate_batches(arrayA, arrayB, categories):
    while True:
        tweak = lambda m : numpy.expand_dims(numpy.array(m), 4)
        for eltA, eltB, eltC in zip(arrayA, arrayB, categories):
            yield [tweak([eltA]), tweak([eltB])], numpy.array([eltC])



if __name__ == "__main__":

    ############################################################################
    n_filter = 25
    h_filter = 50
    d_encode = [64, 64, 64]
    d_intermediate = [128, 64, 32]
    p_dropout = 0.2
    learning_rate = 0.5
    ############################################################################
    model = create_model(locals())
    pairs, categories, proteins = parse.read_data_frame("data/interactions.tsv")
    sequences = parse.get_sequences("data/sequences.fasta", proteins)
    trans_function = sequence.get_compute_embedded_matrix("data/protvec.tsv")
    matrices = sequence.transform_sequences(sequences, trans_function)
    matricesA = (matrices[row.protA] for row in pairs.itertuples())
    matricesB = (matrices[row.protB] for row in pairs.itertuples())
    generator = generate_batches(matricesA, matricesB, categories)
    plot_model(model, "results/filters/model.png")
    visualize_filters(model, 2, n_filter, "results/filters/init.png", "Reds")
    model.fit_generator(generator, steps_per_epoch=10000, epochs=20)
    visualize_filters(model, 2, n_filter, "results/filters/fit.png", "Greens")
