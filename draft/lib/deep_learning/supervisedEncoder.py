"""
                                                    InputLayer (None, 6)
        InputLayer (None, 4)                        Dense (None, 6)
             Dense (None, 4)                        Dense (None, 6)
                   \____________________________________/
                                     |
                                Merge (None, 10)
                                Dense (None, 1)
"""
from numpy.random import seed
from keras.models import Sequential
from keras.layers import Dense, Merge
from keras.constraints import maxnorm
from keras.optimizers import SGD


def createSupervisedEncoder(X1,X2,Y,
							learning_rate=0.1,
                            ENCODING_DIM = 5
							other_parameters = []):

    # Remark : input_shape can be (x,None) to allow input to have a variable dimension in the None part
    branch1 = Sequential()
    branch1.add(Dense(X1.shape[1], input_shape = (X1.shape[1],), init = 'normal', activation = 'relu'))

    branch2 = Sequential()
    branch2.add(Dense(X2.shape[1], input_shape =  (X2.shape[1],), init = 'normal', activation = 'relu'))
    branch2.add(Dense(X2.shape[1], init = 'normal', activation = 'relu', W_constraint = maxnorm(5)))

    model = Sequential()
    model.add(Merge([branch1, branch2], mode = 'concat'))
    model.add(Dense(ENCODING_DIM, init = 'normal', activation = 'sigmoid'))

    # Use a concatenate layer instead ?

    # TODO Remark : you can also write layers as : LayerType(layer_parms)(previous_layer_object)

    model.compile(loss = 'binary_crossentropy',
                  optimizer = SGD(lr = learning_rate),
                  metrics = ['accuracy'])
    seed(42)
    model.fit([X1, X2], Y.values,
			  batch_size = 2000, nb_epoch = 100, verbose = 1)
