def list_keys_values(dictionary):
    """
    Deploy a dictionary and return two lists of keys and values.
    """
    items = dictionary.items()
    keys = (i[0] for i in items)
    values = (i[1] for i in items)
    return keys, values

def get_kmers(sequence, k):
    """
    Return a generator iterating k-mers of the given sequence.
    """
    length = len(sequence)
    for i in range(length - k + 1):
        yield sequence[i : i + k]
