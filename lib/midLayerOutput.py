from keras.models import Model

def getRepresentation(layer_name = 'my_layer', # Use the layer name to select the appropriate layer in the network
                      model,
                      data):
    """
    Remember to fit the original model !
    """
    intermediate_layer_model = Model(inputs=model.input,
                                    outputs=model.get_layer(layer_name).output)

    # Do NOT re-fit this model, or you will erase the fitting done on the original model
    intermediate_output = intermediate_layer_model.predict(data)

    return intermediate_output
