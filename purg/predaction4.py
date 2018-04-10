from keras.preprocessing.sequence import pad_sequences
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

        def sym(l, a, b):
            x = l(a)
            y = l(b)
            return x, y

        a, b = sym(layers.Masking(0),a , b)
        n_token = 20**params["k_mer"]
        a, b = sym(layers.Embedding(n_token, params["d_embed"]),a , b)
        a, b = sym(layers.LSTM(params["units"], return_sequences=True), a, b)
        a, b = sym(layers.Dropout(params["p_dropout"]), a, b)
        a, b = sym(layers.Flatten(), a, b)
        for dim in params["d_encode"]:
            a, b = sym(layers.Dense(dim), a, b)
            a, b = sym(layers.Dropout(params["p_dropout"]), a, b)
        return a, b

    inputsA = layers.Input(shape=(9000,))
    inputsB = layers.Input(shape=(9000,))
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



if __name__ == "__main__":

    ############################################################################
    batch_size = 4
    units = 256
    d_encode = [64, 64]
    d_intermediate = [64, 64]
    p_dropout = 0.3
    learning_rate = 0.1
    k_mer = 3
    d_embed = 128
    ############################################################################
    model = create_model(locals())
    plot_model(model, "results/filters/model.png")
    pairs, categories, proteins = parse.read_data_frame("data/interactions.tsv")
    sequences = parse.get_sequences("data/sequences.fasta", proteins)
    trans_function = sequence.get_tokenize_sequence(k_mer)
    vectors = sequence.transform_sequences(sequences, trans_function)
    vectorsA = [vectors[row.protA] for row in pairs.itertuples()]
    vectorsB = [vectors[row.protB] for row in pairs.itertuples()]
    vectorsA = pad_sequences(vectorsA, value=0, maxlen=9000)
    vectorsB = pad_sequences(vectorsB, value=0, maxlen=9000)
    model.fit([vectorsA, vectorsB], categories, batch_size)
