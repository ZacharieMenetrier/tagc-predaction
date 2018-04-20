#!/usr/bin/env python3

from keras.preprocessing.sequence import pad_sequences
import fr.tagc.predaction.sequence as sequence
import fr.tagc.predaction.parser as parse
from simulate import get_simulate_sample
from keras.optimizers import adam
from keras.models import Model
from keras import regularizers
import keras.layers as layers
import numpy


def create_model(*params):

    def create_branches(a, b):

        def sym(l, x, y):
            a = l(x)
            b = l(y)
            return a, b

        a, b = sym(layers.Embedding(20**kmer, dim_embed), a, b)
        a, b = sym(layers.Dropout(drop_embed), a, b)
        for n, h, drop in zip(n_filter, h_filter, drop_filter):
            a, b = sym(layers.Conv1D(n, h), a, b)
            a, b = sym(layers.MaxPooling1D(), a, b)
            a, b = sym(layers.Dropout(drop), a, b)
        if n_last_filter:
            a, b = sym(layers.Conv1D(n_last_filter, h_last_filter), a, b)
        if units:
            a, b = sym(layers.LSTM(units), a, b)
            a, b = sym(layers.Dropout(drop_lstm), a, b)
        else:
            a, b = sym(layers.GlobalMaxPooling1D(), a, b)
        for dim in dim_encode:
            a, b = sym(layers.Dense(dim), a, b)
            a, b = sym(layers.Dropout(drop_encode), a, b)
        return a, b

    inputsA = layers.Input(shape=(None,))
    inputsB = layers.Input(shape=(None,))
    branchA, branchB = create_branches(inputsA, inputsB)
    layer = layers.concatenate([branchA, branchB])
    for dim, drop in zip(dim_intermediate, drop_intermediate):
        layer = layers.Dense(dim)(layer)
        layer = layers.Dropout(drop)(layer)
    outputs = layers.Dense(2, activation="softmax")(layer)
    model = Model(inputs=[inputsA, inputsB], outputs=outputs)
    opt = adam(lr=learning_rate)
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


    print("step 0")

    ############################################################################
    learning_rate = 0.01

    kmer = 3
    dim_embed = 150
    drop_embed = 0.1

    n_filter = [100, 50]
    h_filter = [6, 6]
    drop_filter = [0.3, 0.3]

    n_last_filter = 50
    h_last_filter = 9
    drop_last_filter = 0.3

    units = 4
    drop_lstm = 0.3

    dim_encode = []
    drop_encode = 0.2

    dim_intermediate = [1024, 512]
    drop_intermediate = [0.3, 0.3]

    steps_per_epoch = 100
    epochs = 1

    simulate = True
    ############################################################################

    model = create_model(locals())

    pairs, categories, proteins = parse.read_data_frame("data/interactions.tsv")
    sequences = parse.get_sequences("data/sequences.fasta", proteins)

    # trans_function = sequence.get_compute_embedded_matrix("data/protvec.tsv")
    trans_function = sequence.get_tokenize_sequence(3)
    matrices = sequence.transform_sequences(sequences, trans_function)

    matricesA = [matrices[row.protA] for row in pairs.itertuples()]
    matricesB = [matrices[row.protB] for row in pairs.itertuples()]

    def shaper(matrix):
        return numpy.array([matrix])

    if simulate:
        simulate = get_simulate_sample((300,500), (20,30), trans_function)
        gen = generate_artificial_batches(simulate, shaper)
    else:
        gen = generate_batches(matricesA, matricesB, categories, shaper)
    model.fit_generator(gen, steps_per_epoch, epochs)
    x = model.evaluate_generator(gen, steps_per_epoch, epochs)
    print(x)
