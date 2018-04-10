from keras.preprocessing.sequence import pad_sequences
import fr.tagc.predaction.sequence as sequence
import fr.tagc.predaction.parser as parse
from visualizer import visualize_filters
from simulate import get_simulate_sample
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
        for dim in params["dim_encode"]:
            a, b = sym(layers.Dense(dim), a, b)
            a, b = sym(layers.Dropout(params["drop_encode"]), a, b)
        return a, b

    inputsA = layers.Input(shape=(None, 100,))
    inputsB = layers.Input(shape=(None, 100,))
    branchA, branchB = create_branches(inputsA, inputsB)
    layer = layers.concatenate([branchA, branchB])
    for dim in params["dim_intermediate"]:
        layer = layers.Dense(dim)(layer)
        layer = layers.Dropout(params["drop_intermediate"])(layer)
    outputs = layers.Dense(2, activation="softmax")(layer)
    model = Model(inputs=[inputsA, inputsB], outputs=outputs)
    opt = adam(lr=params["learning_rate"])
    model.compile(optimizer=opt, loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model



def generate_batches(arrayA, arrayB, categories, shaper):
    array = numpy.array
    while True:
        for eltA, eltB, eltC in zip(arrayA, arrayB, categories):
            yield [shaper(eltA), shaper(eltB)], array([eltC])



def generate_artificial_batches(simulate, shaper):
    array = numpy.array
    while True:
        eltA, eltB, eltC = simulate()
        yield [shaper(eltA), shaper(eltB)], eltC



if __name__ == "__main__":

    ############################################################################
    learning_rate = 0.1

    units = 128
    n_filter = 81
    h_filter = 30
    reg_filter = 0.1
    dim_encode = []
    drop_encode = 0.3
    dim_intermediate = [2048, 2048]
    drop_intermediate = 0.3

    steps_per_epoch = 10
    epochs = 3

    simulate = True
    ############################################################################
    model = create_model(locals())
    # plot_model(model, "results/images/model.png")
    pairs, categories, proteins = parse.read_data_frame("data/interactions.tsv")
    sequences = parse.get_sequences("data/sequences.fasta", proteins)

    trans_function = sequence.get_compute_embedded_matrix("data/protvec.tsv")
    matrices = sequence.transform_sequences(sequences, trans_function)

    matricesA = [matrices[row.protA] for row in pairs.itertuples()]
    matricesB = [matrices[row.protB] for row in pairs.itertuples()]

    def shaper(matrix):
        return numpy.array([matrix])

    def cnn_shaper(matrix):
        x = numpy.expand_dims(shaper(matrix), 3)
        # x = numpy.swapaxes(x, 2, 3)
        return x

    if simulate:
        simulate = get_simulate_sample((100,200), (50,70), trans_function)
        gen = generate_artificial_batches(simulate, shaper)
    else:
        gen = generate_batches(matricesA, matricesB, categories, shaper)
    model.fit_generator(gen, steps_per_epoch, epochs)
    evaluation = model.evaluate_generator(gen, steps_per_epoch, epochs)
    print("loss = " + str(evaluation[0]))
    print("accuracy = " + str(evaluation[1]))
