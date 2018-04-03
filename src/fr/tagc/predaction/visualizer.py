from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, Flatten
from keras.layers import Conv2D, MaxPooling2D, LSTM
from keras import regularizers
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math


def visualize_filters(model, selected_layer, nb_filters, output, cmap="Greens"):
    """
    Returns, for a given layer, a list containing a representation of all
    the learned filters.
    """
    w = model.layers[selected_layer].get_weights()[0][:,:,0,:]
    for i in np.arange(0, nb_filters):
        n = math.ceil(math.sqrt(nb_filters))
        plt.subplot(n, n, i + 1)
        sns.heatmap(w[:,:,i], cmap=cmap, cbar=False, yticklabels=False, xticklabels=False)
    plt.savefig(output)
