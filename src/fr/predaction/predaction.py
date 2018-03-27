#!/usr/bin/env python3
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Merge, concatenate, Input
from keras.constraints import maxnorm
from keras.models import Model
from keras.optimizers import SGD
from numpy.random import seed
from keras.layers import merge
from itertools import chain
import pandas as pd
import numpy as np
import click


@click.command()
@click.argument("input_file_path")
@click.option("--encoding_dim", default=5)
@click.option("--learning_rate", default=0.1)
def predaction(input_file_path, encoding_dim, learning_rate):
    x, y = read_data_frame(input_file_path)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
    model = create_supervised_encoder(x_train, y_train,
                                      learning_rate, encoding_dim)
    labels_unlist = np.argmax(y_test, 1)
    preds = np.argmax(model.predict(split_x(x_test)), 1)
    print(pd.crosstab(preds, labels_unlist))

def split_x(x):
    nb_col = len(x.columns)
    mid = int((nb_col)/2)
    x1 = x.iloc[:,0:mid]
    x2 = x.iloc[:,mid:nb_col]
    return [x1, x2]


def create_supervised_encoder(x, y, learning_rate=0.1, encoding_dim=5):
    input_size = int(x.shape[1] / 2)
    first_input = Input(shape=(input_size, ))
    branch1 = Dense(2, kernel_initializer="normal", activation="relu")(first_input)
    second_input = Input(shape=(input_size, ))
    branch2 = Dense(2, kernel_initializer="normal", activation="relu")(second_input)
    concat = concatenate([branch1, branch2])
    encode = Dense(encoding_dim, init = 'normal', activation = 'sigmoid')(concat)
    out = Dense(2, activation="softmax")(encode)
    model = Model(inputs = [first_input, second_input], outputs = out)
    model.compile(loss="binary_crossentropy", optimizer=SGD(lr=learning_rate),
                  metrics=["accuracy"])
    model.fit(split_x(x), y, batch_size=2000, epochs=10000)
    return model


def read_data_frame(filepath):
    df = pd.read_csv(filepath, sep="\t", header=None)
    nb_col = len(df.columns)
    y = df.iloc[:,-1]
    t = to_categorical([1 if yy == "YES" else 0 for yy in y], num_classes=2)
    x = df.iloc[:,0:nb_col - 1]
    return (x, t)


if __name__ == "__main__":
    predaction()
