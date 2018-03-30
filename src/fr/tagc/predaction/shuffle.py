import pandas
import copy

def parse_interactions(file_path):
    df = pandas.read_csv(file_path, sep="\t")
    interactions = set()
    for row in df.itertuples():
        interactions.add(frozenset([row.protA, row.protB]))
    return interactions

def shuffle(interactions):
    interactions = copy.copy(interactions)
    shuffles = set()
    while len(interactions) > 1:
        a = set(interactions.pop())
        b = set(interactions.pop())
        aa = frozenset((a.pop(), b.pop()))
        bb = frozenset((a.pop(), b.pop()))
        if aa not in interactions:
            shuffles.add(aa)
        if bb not in interactions:
            shuffles.add(bb)
    return shuffles


def get_line(interaction, category):
    interaction = list(interaction)
    print(interaction)
    return interaction[0] + "\t" + interaction[1] + "\t" + category


def write_interactome(positome, negatome, output_file_path):
    with open(output_file_path, "w") as output_file:
        output_file.write("protA\tprotB\tinteraction\n")
        for neg, pos in zip(positome, negatome):
            line = get_line(neg, "YES") + "\n"
            output_file.write(line)
            line = get_line(pos, "NO") + "\n"
            output_file.write(line)

if __name__ == "__main__":
    i = parse_interactions("data/trivial_data.tsv")
    s = shuffle(i)
    write_interactome(i, s, "data/interactome.tsv")
