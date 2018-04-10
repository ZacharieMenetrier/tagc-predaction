from scipy.sparse import coo_matrix
from parser import get_vectors
import numpy
import utils



def transform_sequences(sequences, transformation, fmap=map):
    """
    Return a dictionary of sequence features for each entry of the sequences
    dictionary given in the input.
    The features are computed by the function passed in parameter.
    Set extend to None if you want to get variable lengths results.
    """
    proteins, sequences = utils.list_keys_values(sequences)
    features = fmap(transformation, sequences)
    features = dict(zip(proteins, features))
    return features



def get_compute_embedded_matrix(vectors_file_path):
    """
    Return a function that compute the protein vector for a given sequence.
    """
    vectors = get_vectors(vectors_file_path)
    def compute_embedded_matrix(sequence):
        trimers = utils.get_kmers(sequence, 3)
        return [vectors[trimer] for trimer in trimers]
    return compute_embedded_matrix



def get_compute_numeric_sequence():
    """
    Return a function that compute the numerical array for a given sequence.
    """
    amino_acid_values = {
                           "A" :  1, "C" :  2, "D" :  3, "E" :  4, "F" :  5,
                           "G" :  6, "H" :  7, "I" :  8, "K" :  9, "L" : 10,
                           "M" : 11, "N" : 12, "P" : 13, "Q" : 14, "R" : 15,
                           "S" : 16, "T" : 17, "V" : 18, "W" : 19, "Y" : 20
                        }
    def compute_numeric_sequence(sequence):
        return [amino_acid_values[amino_acid] for amino_acid in sequence]
    return compute_numeric_sequence



def get_tokenize_sequence(k):
    """
    Return a function that compute the numerical token for a given sequence.
    """
    amino_acid_values = {
                           "A" :  0, "C" :  1, "D" :  2, "E" :  3, "F" :  4,
                           "G" :  5, "H" :  6, "I" :  7, "K" :  8, "L" :  9,
                           "M" : 10, "N" : 11, "P" : 12, "Q" : 13, "R" : 14,
                           "S" : 15, "T" : 16, "V" : 17, "W" : 18, "Y" : 19
                        }
    def tokenize_kmer(kmer):
        token = 0
        for i in range(len(kmer)):
            n = amino_acid_values[kmer[i]]
            token += n * (i + 1)
        return token + 1
    def tokenize_sequence(sequence):
        kmers = utils.get_kmers(sequence, k)
        return [tokenize_kmer(kmer) for kmer in kmers]
    return tokenize_sequence



def get_compute_matrix():
    """
    Return a function that compute the compressed matrix for a given sequence.
    The matrix contains 21 columns for each possible amino acids.
    And the length of the rows is the length of the sequence.
    Or if specified the length of extend with 0s only.
    """
    amino_acid_values = {
                       "A" : [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                       "C" : [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                       "D" : [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                       "E" : [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                       "F" : [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                       "G" : [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                       "H" : [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                       "I" : [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                       "K" : [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                       "L" : [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                       "M" : [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
                       "N" : [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
                       "P" : [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                       "Q" : [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
                       "R" : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
                       "S" : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
                       "T" : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
                       "U" : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
                       "V" : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
                       "W" : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
                       "Y" : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                       "B" : [0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
                       "Z" : [0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
                       "X" : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                      }
    def compute_matrix(sequence):
        return [amino_acid_values[amino_acid] for amino_acid in sequence]
    return compute_matrix
