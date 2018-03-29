This repository contains draft code for the *tagc-predaction* tool.

The main principle behind this tool is to predict whether a mutation will have an impact on a protein's interaction with another.

To do so, we encode each protein in a supervised manner to increase the relative weight of the mutations.


# Draft

The *'./draft'* directory contains proof-of-concept code supplied by Quentin Ferré and meant to be mainly edited and reviewed by him.

# Data

We may work with conjoined triad (representaition of amino acid composition) or with FASTA sequences.

# Approach

## Supervised encoder

Use a supervised encoder to create a new representation of the proteins ,alone or in relation with their mutants.

The supervised aspect allow us to emphasize the mutations and their impact on interactivity.

It is then possible to extract outputs from the intermediary layer to, well, extract this representation.

### Attention map

Trying to maximize output for the neurons in the representation layers allows us to see where each neuron pays attention.

## Decision tree

Such a representation can be used in an Adaboost model.

It is also possible to create a simple decision tree. While less precise, it is graphically visualizable.


# Links

Keras Documentation : https://keras.io/

# Authors

Zacharie Menetrier <zacharie.menetrier@gmail.com>

Quentin Ferré <quentin.ferre@inserm.fr>