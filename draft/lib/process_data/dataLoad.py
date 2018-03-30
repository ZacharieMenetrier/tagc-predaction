import numpy as np
import panda as pd

def readDataFrame(filepath):
    df = pd.read_csv(filepath)
    # The last column should be the labels, which leaves an even number of columns.
    nb_col = len(df.columns)
    mid = (nb_col-1)/2
    Y = df[-1] # Interacts or not ? Labels
    X1 = df[1:mid]# First protein representations
    X2 = df[mid:-1] # Second protein representations
    # The respective order of the labels MUST be conserved

    # Reshape with numpy !

    # Data must be converted into a vector processable by sklearn
    # Based on what I've read on conjoined triads, it should be the case already.

    return (X1,X2,Y)
