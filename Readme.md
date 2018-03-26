Draft code for the analyis.

**Goal** : encode each protein in a supervised manner so mutation gain more weight

# Data
Work with data as conjoined triad - each protein a vector.
I recommend storing it in a TSV file, that can be read as a pandas DataFrame.

# Documentation
Should go in the 'doc' directory.

## Article
This paper uses a 'stacked autoencoder', whatever that means. It also uses conjoined triad.
https://doi.org/10.1186/s12859-017-1700-2

## Links
Keras Documentation : https://keras.io/

# Contents
## Library
Functions are stored in the *'./lib'* directory ; the main file is at the root, *'./main.py'*

## Supervised encoder
Use a supervised encoder to create a new representation of the proteins.
The supervised aspect allow us to emphasize the mutations and their impact on interactivity.

This encoder by itself may be able to perform better than what you've tried before.

You can then extract outputs from the intermediary layer to, well, extract this representation

### Attention map
Trying to maximize output for the neurons in the representation layers allows us to see where each neuron pays attention.

## Decision tree
Such a representation can be used in an Adaboost model.

It is also possible to create a simple decision tree. While less precise, you can visualize it graphically

## Draft
Draft is proof of concept folder meant to be reviewed by Quentin Ferré.

# Git
Make it a git bitbucket or a git renater maybe.

# Authors
Zacharie Menetrier <zacharie.menetrier@gmail.com>
Quentin Ferré <quentin.ferre@inserm.fr>
