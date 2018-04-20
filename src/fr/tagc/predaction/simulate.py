from keras.utils import to_categorical
from numpy.random import choice
from random import randint
from random import sample
from sequence import get_compute_embedded_matrix


def get_simulate_sample(sequence_size, patch_size, trans_function):
    aa_list =  ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", "L", "K", "M",
                "F", "P", "S", "T", "W", "Y", "V"]

    def simulate_sample():

        category = choice([1, 0], 1)
        seqA = random_sequence(aa_list, sequence_size)
        seqB = random_sequence(aa_list, sequence_size)
        if category == 1:
            # amino_acids = ["A", "A"]
            amino_acids = sample(aa_list, 1) * 2
        else:
            amino_acids = sample(aa_list, 2)
        seqA = patch_of(seqA, patch_size, amino_acids[0])
        seqB = patch_of(seqB, patch_size, amino_acids[1])
        eltA = trans_function(seqA)
        eltB = trans_function(seqB)
        category = to_categorical(category, num_classes=2)
        return eltA, eltB, category

    return simulate_sample



def random_sequence(aa_list, size) :
    """
    Returns a random list of the given size, with elements drawn from aa_list.
    """
    return "".join(choice(aa_list, randint(*size), replace=True))



def patch_of(sequence, size, amino_acid):
    """
    Given a sequence, will randomly select a contiguous window of the given size
    and overwrite it with the given amino acid.
    """
    size = randint(*size)
    pos = randint(0, len(sequence) - size + 1)
    patch = amino_acid * size
    new_seq =  sequence[:pos] + patch + sequence[pos + size:]
    return new_seq
