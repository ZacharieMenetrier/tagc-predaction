from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, Flatten
from keras.layers import Conv2D, MaxPooling2D, LSTM
from keras import regularizers



def createConvoLayer(window_size,nb_lines,nb_classes,
                      nb_filters_firstlayer = 250, kernel_size = 20,
                      hidden_dims = 120,reg_coef_filter=0,reg_coef_dense=0):
    """
    This will create a CNN branch than can then be embeded in you project.

    This layer is designed to be used close to the data, to perform a processing
    than can then be fed into the more general supervised encoder.

    window_size = The number of nucleotides/columns. Can be equal to None ?
    nb_lines = The number of lines, in your case the number of amino acid classes



    """
    branch = Sequential()

    # TODO : since no column can have more than one line with a value for any
    # given column, why don't we simply concatenate the matrix as a line ?

    # The first layer will learn filters at the base TF level
    branch.add(Conv2D(filters = nb_filters_firstlayer,   # Number of filters
                     kernel_size=(kernel_size,nb_lines), # Filter shape in (width,height) order
                        # IMPORTANT : The number of lines MUST be equal to the height of your matrix so as to perform UNIDIMENSIONAL CONVOLUTION
                     input_shape=(nb_lines,window_size,1), # Shape of our data (rows,columns,c); c is the number of channels, here equal to 1
                     activation='relu',
                     padding='same',
                     kernel_regularizer=regularizers.l2(reg_coef_filter),
                        # Regularization is important both to prevent overfitting and to have human-readable elements later
                     data_format = 'channels_last'))

    # Add a pooling layer here
    branch.add(MaxPooling2D(pool_size = (1,2))) # Do NOT pool on the y (vertical) axis (number of lines)

    # We need to flatten this to supply it to a Dense layer
    branch.add(Flatten())

    # Final layer : dense one to treat our filters
    branch.add(Dense(hidden_dims,activity_regularizer=regularizers.l2(reg_coef_dense)))
    branch.add(Activation('relu'))


    return branch
