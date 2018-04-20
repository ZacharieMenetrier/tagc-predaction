from numpy.random import choice
import numpy as np
import random
import bisect
import itertools
from collections import Counter

# Probability distribution for each amino acid
aa_list = list("ACDEFGHIKLMNPQRSTVWY")
aa_proba_raw = 20 * [1/20] # THIS IS THE KEY ARGUMENT, you should replace this with your actual probabilities
aa_proba = dict(zip(aa_list, aa_proba_raw))

# Weighted Random Generator using King-of-the-Hill method
# Source : source : https://eli.thegreenplace.net/2010/01/22/weighted-random-generation-in-python#id3
class WeightedRandomGenerator:
    def __init__(self, weights):
        self.totals = []
        running_total = 0

        for w in weights:
            running_total += w
            self.totals.append(running_total)

    def next(self):
        rnd = random.random() * self.totals[-1]
        return bisect.bisect_right(self.totals, rnd)

    def __call__(self): return self.next()

# Generate a random protein sequence
def generator_random_prot(aa_proba,seqlen):
    aa_list = list(aa_proba.keys())
    aa_proba_raw = list(aa_proba.values())
    gen = WeightedRandomGenerator(aa_proba_raw)
    weighted_randoms = [gen() for x in np.arange(seqlen)]
    yield ''.join([aa_list[i] for i in weighted_randoms])

# Efficient generation of a list of sequences
NUMBER_OF_SEQUENCES = 1000
SEQLEN = 30
random_sequences = [generator_random_prot(aa_proba,SEQLEN).__next__() for j in np.arange(NUMBER_OF_SEQUENCES)]
# Since generator_random_prot is a generator, you can also write everything to
# a file or generate the sequences one by one

# Generate all possible k-mers
KMER_SIZE = 3
kmers = [''.join(i) for i in itertools.product(*itertools.repeat(aa_list, KMER_SIZE))]

# Counting all k-mers
# There is probably a much more efficient way to to this.
c = Counter()
for kmer in kmers: c.update({kmer:0})

for seq in random_sequences :
    l = len(seq)-KMER_SIZE
    for i in np.arange(l):
        word = seq[i:i+KMER_SIZE]
        c[word] += 1

# The baseline probability of each k-mer, considering conditional dependencies,
# has thus been estimated with a Monte Carlo approach.
monte_carlo_probas=Counter()
for key in c: monte_carlo_probas[key] = c[key]/sum(c.values())
