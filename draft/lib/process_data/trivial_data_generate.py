import numpy as np

aa_list =  ['A','R','N','D','B','C','E','Q','Z',
            'G','H','I','L','K','M','F','P','S',
            'T','W','Y','V']

######## HELPER FUNCTIONS FOR SEQUENCE GENERATION ########

def background_random(size) :
    """
    Returns a random list of the given size, with elements drawn from aa_list.
    """
    return np.random.choice(aa_list, size, replace=True)

def patch_of(list,size,aa):
    """
    Given a list, will randomly select a contiguous window of the given size
    and overwrite it with the given amino acid
    """
    pos = np.random.randint(0,len(list)-size)
    list[pos:pos+size] = np.repeat(aa,size)
    return list

def generate(status,seq_len,patch_size,nb_seq) :
    """
    If status is True : will generate two strings with a random patch of the
    SAME amino acid applied to them.
    If status is False : will generate two strings with a random patch of
    TWO DIFFERENT amino acids applied to them.
    This is a generator, and will generate up to 'nb_seq' strings.
    """
    while nb_seq > 0:

        # Pick amino acids
        if status : aas = [np.random.choice(aa_list,1)] * 2
        else : aas = np.random.choice(aa_list,2,replace = False) # Draw without replacement

        strA = ''.join(patch_of(background_random(seq_len),patch_size,aas[0]))
        strB = ''.join(patch_of(background_random(seq_len),patch_size,aas[1]))
        nb_seq -= 1
        yield (strA,strB)


######## SEQUENCE GENERATORS ########

def generate_random_seqs(seq_len=200,patch_size=20,nb_pairs_of_seq_pairs=1000) :
    """
    Generates random sequences for the given parameters.

    Will generate one pair of sequences of YES status, and one pair of NO, and
    will repeat this opeation for nb_pairs_of_seq_pairs times.

    Returns them as a dict
    with an unique couple of sequences as key and their status as value.
    """
    iter_yes = generate(True,seq_len,patch_size,nb_pairs_of_seq_pairs)
    iter_no = generate(False,seq_len,patch_size,nb_pairs_of_seq_pairs)
    return {**{x:'YES' for x in iter_yes},**{y:'NO' for y in iter_no}}
