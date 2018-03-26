import keras
from keras.models import Sequential
from keras.layers import Dense, Activation


model = Sequential()
model.add(Dense(32, input_dim=100, activation="relu"))
model.add(Dense(2, activation="softmax"))
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

# Generate dummy data
import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000, 1))
# Convert labels to categorical one-hot encoding
one_hot_labels = keras.utils.to_categorical(labels, num_classes=2)
# Train the model, iterating on the data in batches of 32 samples
model.fit(data, one_hot_labels, epochs=10, batch_size=32)


import pandas as pd
from itertools import chain

preds = np.argmax(model.predict(data), 1)
labels_unlist = np.array(list(chain.from_iterable(labels)))

var = pd.crosstab(preds, labels_unlist)

print(var)
