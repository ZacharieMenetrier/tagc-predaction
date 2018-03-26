import numpy as np
import pandas as pd
from keras import *
from keras.models import *
from keras.layers import *
import time

# Source :  http://ankivil.com/visualizing-deep-neural-networks-classes-and-features/

def dnnLayerGradientAscent(model,
                            selected_layer,
                            features_list,
                            learning_rate = 2,
                            random_state = 42,
                            nb_steps_gradient_ascent = 20,
                            ):
    """
    model : a DNN model
    selected_layer : the layer on which to perform gradient ascent
    learning_rate & nb_steps_gradient_ascent : parameters
    features_list = a list of the features ID for display in the output

    returns a data frame of activations for this layer per neuron per feature
    """
    kept_images=[]
    losses=[]


    # Specify input and output of the network
    layer_input = model.layers[0].input
    layer_output = model.layers[selected_layer].output
    layer_output_size_of_output = int(layer_output.shape[1]) # Get the size of the output of the selected layer
    np.random.seed(random_state)  # for reproducibility

    for neuron_index in range(layer_output_size_of_output): # For each neuron
        start_time = time.time()
        loss = layer_output[0, neuron_index] # The loss is the activation of the neuron for the chosen neuron

        # we compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, layer_input)[0]

        # this function returns the loss and grads given the input picture
        # also add a flag to disable the learning phase (in our case dropout)
        iterate = K.function([layer_input, K.learning_phase()], [loss, grads])

        # we start from a gray image with some random noise
        input_data = np.random.uniform(0, 1, (1,) + model.input_shape[1:]) # (1,) for batch axis

        # we run gradient ascent for n steps
        for i in range(nb_steps_gradient_ascent):
            loss_value, grads_value = iterate([input_data, 0]) # Disable the learning phase
            input_data += grads_value * learning_rate # Apply gradient to image

            #print('Current loss value:', loss_value)

        # decode the resulting input image and add it to the list
        kept_images.append(input_data[0])
        losses.append(loss_value)
        end_time = time.time()
        print('Neuron %d processed in %ds' % (neuron_index, end_time - start_time))



    # Present it cleanly
    activation_dataframe = pd.DataFrame(kept_images, columns = features_list)
        #, index=LABELS)

    return activation_dataframe
