from keras.utils import to_categorical
from pandas import read_csv



def get_sequences(file_path, includes=None):
    """
    Return a dictionary of sequences with protein names as keys.
    The input file must be in the FASTA format.
    The includes parameter may be a list of proteins to keep.
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
        sequences = {protein : sequences[protein] for protein in includes}
    return sequences



def get_sequences_from_tsv(file_path, includes=None):
    """
    Return a dictionary of sequences with protein names as keys.
    The input file must be in the FASTA format.
    The includes parameter may be a list of proteins to keep.
    If None all the proteins found are returned.
    """
    sequences = {}
    protein = False
    sequence = ""
    with open(file_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip().split("\t")
            sequences[line[0]] = line[1]
    if includes:
        sequences = {protein : sequences[protein] for protein in includes}
    return sequences



def get_vectors(file_path):
    """
    Return a dictionary of vectors with 3-mers names as keys.
    The file is furnished by the BioVec project.
    """
    vectors = {}
    with open(file_path, "r") as file:
        lines = file.readlines()
        header = lines.pop(0)
        for line in lines:
            line = line.strip().split("\t")
            word = line.pop(0)
            vectors[word] = [float(cell) for cell in line]
    return vectors



def read_data_frame(file_path):
    """
    Read a data frame of interacting proteins and return the pairs,
    the categories and the list of all proteins without duplication.
    """
    def status_to_categories(categories):
        cats = [1 if x == "YES" else 0 for x in categories]
        cats = to_categorical(cats, num_classes=2)
        return cats

    df = read_csv(file_path, sep = "\t")
    pairs = df.iloc[:, :-1]
    categories = status_to_categories(df.iloc[:,-1])
    proteins = set([row.protA for row in pairs.itertuples()])
    proteins.update(set([row.protB for row in pairs.itertuples()]))
    return pairs, categories, proteins
