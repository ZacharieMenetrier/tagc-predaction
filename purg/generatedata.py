from collections import namedtuple
from random import sample
import numpy as np


def random_sequence(aa_list, size) :
    """
    Returns a random list of the given size, with elements drawn from aa_list.
    """
    return "".join(np.random.choice(aa_list, size, replace=True))


def patch_of(sequence, size, amino_acid):
    """
    Given a sequence, will randomly select a contiguous window of the given size
    and overwrite it with the given amino acid
    """
    pos = np.random.randint(0, len(sequence) - size + 1)
    patch = amino_acid * size
    new_seq =  sequence[:pos] + patch + sequence[pos + size:]
    return new_seq


def generate(status, seq_len, patch_size, nb_seq, random_validate=False) :
    """
    If status is True : will generate two strings with a random patch of the
    SAME amino acid applied to them.
    If status is False : will generate two strings with a random patch of
    TWO DIFFERENT amino acids applied to them.
    This is a generator, and will generate up to 'nb_seq' strings.
    """
    aa_list =  ["A", "R", "N", "D", "B", "C", "E", "Q", "Z", "G", "H", "I", "L",
                "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
    Couple = namedtuple("Couple", ["seqA", "seqB", "status"])
    def get_couple(status):
        if not random_validate and status:
            # amino_acids = sample(aa_list, 1) * 2
            amino_acids = ["C", "W"]
        else:
            amino_acids = sample(aa_list, 2)
        seqA = random_sequence(aa_list, seq_len)
        seqA = patch_of(seqA, patch_size, amino_acids[0])
        seqB = random_sequence(aa_list, seq_len)
        seqB = patch_of(seqB, patch_size, amino_acids[1])
        couple = Couple(seqA, seqB, "YES" if status else "NO")
        return couple
    return (get_couple(status) for x in range(nb_seq))


def generate_random_seqs(seq_len, patch_size, n_pairs_sequences, random_validate=False):
    """
    Generates random sequences for the given parameters.
    Will generate one pair of sequences of YES status, and one pair of NO, and
    will repeat this opeation for nb_pairs_of_seq_pairs times.
    Returns them as a dict
    with an unique couple of sequences as key and their status as value.
    """
    iter_yes = generate(True, seq_len, patch_size, n_pairs_sequences, random_validate)
    iter_no = generate(False, seq_len, patch_size, n_pairs_sequences, random_validate)
    return zip(iter_yes, iter_no)

if __name__ == "__main__":
    print(list(generate_random_seqs(100, 50, 10, True)))
