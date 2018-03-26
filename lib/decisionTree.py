# Standard decision tree
from sklearn import tree
import graphviz

# Here put my functino to export tree into pdf
def exportTree(treeToPrint, output_path, features_names):
    """
    Quick wrapper to export a tree to a given output path with default parameters
    """
    dot_data = tree.export_graphviz(treeToPrint, out_file=None,filled=True,
                                    feature_names=features_names,
                                    rounded=True, special_characters=True)
    graph_decision = graphviz.Source(dot_data)
    graph_decision.render(output_path)


# Also put my code to turn decision tree into dataframe ?

# Random forest

# Copy my Adaboost code here






import collections
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import matplotlib as plt
import seaborn as sns

# Turning the decision tree into a figure where each path has a worth
# Source : http://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html

def get_lineage(tree, feature_names):
    left      = tree.tree_.children_left
    right     = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features  = [feature_names[i] for i in tree.tree_.feature]

    combinations = list()
    combinations_dict = dict()

    # Start from every node except the first one
    idx=np.arange(1,tree.tree_.node_count)

    def recurse(left, right, child, lineage=None):
        if lineage is None: lineage = [child]
        if child in left:
            parent = np.where(left == child)[0].item()
            split = 'l'
        else:
            parent = np.where(right == child)[0].item()
            split = 'r'

        lineage.append((parent, split, threshold[parent], features[parent]))


        if parent == 0:
            lineage.reverse()
            return lineage
        else:
            return recurse(left, right, parent, lineage)

    for child in idx:
        for node in recurse(left, right, child):
            combinations.append(node)

    def lineageElementToHumanReadable(lineage_elem):
        parent, split, threshold, features = (lineage_elem)
        if split == 'l' : split_ap = '<'
        else : split_ap = '>'
        return (str(features) + ' ' + split_ap + ' ' + str(threshold.round(4)))


    # Processing the list and assigning nodes to a dictionary
    currentLineage = list()
    for elem in combinations :
        if type(elem) == tuple :    # If we are processing a lineage element
            hr_elem = lineageElementToHumanReadable(elem)
            currentLineage.append(hr_elem)
        else :  # If we have reached an end node
            combinations_dict[elem] = currentLineage
            currentLineage = list()

    return combinations_dict











def getIDofSamplePerNode(clf, data):
    """
    clf = the decision tree to be processed
    data = the data
    Will return a dictionary of each node along with the ID of samples assigned to it
    """
    samples = collections.defaultdict(list)
    dec_paths = clf.decision_path(data)

    for d, dec in enumerate(dec_paths):
        for i in range(clf.tree_.node_count):
            if dec.toarray()[0][i]  == 1:
                samples[i].append(d)
    return samples













# Concatenate all this information
def decisionTreeIntoDataFrame(clf, data, target, feature_names):
    """
    Takes as input a fitted decision tree.
    THE CLASSIFIER MUST HAVE BEEN TRAINED BEFORE, IT WILL NOT BE TRAINED ON THE
    DATA YOU PROVIDE HERE - so you can run this with testing data safely

    clf = the decision tree
    data = the matrix of features
    target = a list of labels/targets for each example in the data
    feature_names = ordered list of feature_names associated with the tree ; is data is a pandas dataframe you can use list(data)

    Returns a dataframe with, for each leaf node, its decision path and the
    ID of the samples that were assigned to it in training.
    """

    # Turn the decision tree into a list of lineages
    # Data does not intervene at this stage
    genealogy = get_lineage(clf,feature_names)

    # For the provided data, assign each element to a node
    # The tree will not be re-trained, so this can be used with testing data
    assignations = getIDofSamplePerNode(clf, data)

    # Now for each node, compute the count of samples of each class
    class_counts_per_leaf = {k:collections.Counter(target[assignations[k]]) for k in list(genealogy.keys())}

    # Finally : instead of node number, use the node's lineage
    rules = {tuple(genealogy[k]):class_counts_per_leaf[k] for k in list(genealogy.keys())}



    #### Processing for legibility ####

    ## Turning the rules into a dataframe with one step per column
    longest_lineage_length = len(max(rules.keys(), key=len))
    rules_tree_df = pd.DataFrame(list(rules.keys()), columns = np.arange(longest_lineage_length))

    ## Turning the counts into a dataframe
    counts_tree_df = pd.DataFrame()
    i=0
    for v in rules.values():
        v_dict = dict(v) # Turn the counter object in a regular dictionary
        counts_tree_df = counts_tree_df.append(pd.DataFrame(v_dict, index=[i]))
        i=i+1

    # Dirty hack : if a class is absent, its value is NaN; so we must replace all NaNs with zeroes
    counts_tree_df = counts_tree_df.fillna(0)

    ## Adding the ID of the assignations
    samples_id_per_leaf = {k:','.join(map(str,assignations[k])) for k in list(genealogy.keys())}

    id_per_node_df =  pd.DataFrame()
    i=0
    for n in samples_id_per_leaf.keys():
        id_per_node_df = id_per_node_df.append(
                                                pd.DataFrame(
                                                            {'samples_id':samples_id_per_leaf[n]}

                                                ,index=[i])
                                                )
        i=i+1

    # Join the three dataframes
    tree_df = pd.concat([rules_tree_df,counts_tree_df,id_per_node_df], axis=1)

    return tree_df














def computeAverageProfiles(tree_df,data,figsize=[60,30]):
    """
    tree_df : a decision tree turned into a dataframe by our custom decisionTreeIntoDataFrame.decisionTreeIntoDataFrame function
    data = the pandas dataframe of the data
    """

    # Compute average profile per node
    average_profiles = list()
    for samples_id in tree_df['samples_id']:
        samples_id_as_list = [int(s) for s in samples_id.split(',')]
        average_profiles.append(data.iloc[samples_id_as_list,].mean(axis=0))

    # Turn it into a heatmap
    plt.rcParams["figure.figsize"] = figsize
    myCPal = sns.cubehelix_palette(32, start=3, rot=0, dark=0, light=1, reverse=False)
    fig = sns.heatmap(pd.DataFrame(average_profiles, index = tree_df.index),cmap=myCPal).get_figure()

    return (fig, average_profiles)
