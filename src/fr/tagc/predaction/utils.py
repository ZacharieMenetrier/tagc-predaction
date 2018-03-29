from scipy.sparse import coo_matrix
from functools import partial
from numpy import pad, int8


def get_sequences(file_path, includes=None):
    """
    Return a dictionary of sequences with protein names as keys.
    The input file must be in the FASTA format.
    The include may be a list of protein to keep.
    If None all the proteins found are returned.
    """
    sequences = {}
    protein = False
    sequence = ""
    with open(file_path, "r") as file:
        lines = file.readlines()
        while not protein:
            line = lines.pop(0)
            if line.startswith(">"):
                protein = line.replace(">", "").strip()
        for line in lines:
            if line.startswith(">"):
                sequences[protein] = sequence
                sequence = ""
                protein = line.replace(">", "").strip()
            else:
                sequence += line.strip()
    if includes:
        sequences = {include: sequences[include] for include in includes}
    return sequences


def get_matrices(sequences, extend=True, fmap=map):
    """
    Return a dictionary of sequence matrices for each entry of the sequences
    dictionary given in the input.
    Extend the matrices by the max length of the sequences if extend is true.
    Set extend to None if you want to get variable lengths matrices.
    """
    matrices = {}
    extend = len(max(sequences.values(), key=len)) if extend else None
    compute_matrix = get_compute_matrix()
    items = sequences.items()
    sequences = [i[1] for i in items]
    proteins = [i[0] for i in items]
    matrices = fmap(partial(compute_matrix, extend=extend), sequences)
    matrices = dict(zip(proteins, matrices))
    return matrices


def get_compute_matrix():
    """
    Return a function that compute the numpy matrix for a given sequence.
    The matrix contains 21 columns for each possible amino acids.
    And the length of the rows is the length of the sequence.
    Or if specified the length of extend with 0s only.
    """
    extending_row = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    amino_acid_rows = {
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
    def get_row(amino_acid):
        return amino_acid_rows[amino_acid]
    def compute_matrix(sequence, extend=None):
        rows = list(map(get_row, sequence))
        if extend:
            padding = extend - len(sequence)
            rows = pad(rows, ((0, padding),(0, 0)), "constant")
        return coo_matrix(rows)
        # return coo_matrix(rows, dtype=int8)
    return compute_matrix
