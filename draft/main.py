#!/usr/bin/python
import sys
sys.path.append("./lib")

import pandas as pd
import numpy as np

# Load data
import dataLoad
X1, X2, Y = dataLoad.readDataFrame("./data/data.tsv")

X1_train, X1_test, X2_train, X2_test, y1_train, y1_test = train_test_split(
...     X1,X2 y, test_size=0.33, random_state=42)


# Phase 1 : Transcoder with merge (supervised)
# Supervised because they must learn what features in combination with which are important.
# Otherwise we are back to your problem : small differences are not taken into account because a similar (non-mutant) example also works
import supervisedEncoder
ENCODING_DIM = 5
model = supervisedEncoder.createSupervisedEncoder(X1,X2,Y,ENCODING_DIM,other_parameters)

# For comparison, what happens on random data ?
random_data = pd.DataFrame(np.random.uniform(0, 1, size = (5000,len(features_combined.columns))), columns=list(features_combined.columns))
autoencoder.fit(random_data, random_data,epochs=50)
reconstructed_features_rand = autoencoder.predict(random_data)
normalized_mse_rand = (((reconstructed_features_rand-random_data))**2).sum().sum() / (random_data**2).sum().sum()


# This encoder by itself may be able to perform better than what you've tried before.
# Now predict using the testing data and see.
pd.crosstab(model.predict(X_test),Y_test)







# Phase 2 : extract output intermediate layer
# This allows to extract a representation of our proteins, a representation that was created with SUPERVISION and so emphasizes important features
# TODO in supervisedEncoder.createSupervisedEncoder, you must name the layers so you can select them here
import midLayerOutput
encoded_data = [midLayerOutput.getRepresentation(layer_name,model,[x1,x2]) for x1,x2 in zip(X1,X2)]

# Put those encoded features into a dataframe
features_names = ['A'+str(i) for i in np.arange(ENCODING_DIM)] + ['B'+str(i) for i in np.arange(ENCODING_DIM)]
encoded_df = pd.DataFrame(encoded_data, columns=features_names )









# Phase 4 : gradient ascent to understand our encoded features
import dnnLayerVisualization
# TODO in dnnLayerVisualization, replace layer_id with layer_name for the reasons mentioned above
dnnLayerVisualization.dnnLayerGradientAscent(model,
                            -4,
                            features_combined.columns,
                            learning_rate = 2,
                            random_state = 666,
                            nb_steps_gradient_ascent = 20,
                            )








# Phase 5 : AdaBoost
import numpy as np
from sklearn.ensemble import AdaBoostClassifier

# Create and fit an AdaBoosted decision tree
underlying_decision_tree = DecisionTreeClassifier(max_depth=1) # AdaBoost uses stumps, ie. trees of depth of 1 or 2 at most
bdt = AdaBoostClassifier(algorithm="SAMME",
                         n_estimators=200)
bdt.fit(X, y)
result = bdt.predict(data)


adaClassifierTree = ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=1,criterion='entropy',    # Ada uses STUMPS
                                                                            class_weight='balanced'))

params_ada =  {"n_estimators": np.arange(10,1000,200), # Previous : 10-1000-50
                "learning_rate": [0.0005,0.001,0.005,0.01,0.05]}
adaClassifierTree = evaluation.performGridSearch(estimator=adaClassifierTree, parameters_grid= params_ada,
                            X_train=features_combined, y_train=labels_discrete_for_tree)

plt.show(evaluation.plotLearningCurve(adaClassifierTree, 'AdaBoost-ed Classifiers',
                                        features_combined[RANDOM_FOREST_FEATURES],
                                        labels_discrete_for_forest))











# Phase 6 : decision tree visualisation (combined with understnading of encoded features)
# Less effective than Adaboost, but at least it is visualizable
import decisionTree
import evaluation
deicisionTree.exportTree(tree,output_path,feature_names)
tree_df = decisionTree.decisionTreeIntoDataFrame(clf_OS, encoded_df, Y, list(encoded_df))

myClfTree = tree.DecisionTreeClassifier(class_weight='balanced')

# Grid Search for the optimal number of nodes on all the dataset
params_classifier =  {"max_leaf_nodes": np.arange(20,60,30),     # Previous 20-60-5
                      "min_samples_leaf": np.arange(100,300,30), # Previous 100-300-15
                      "criterion": ['entropy','gini']}
clf_OS = evaluation.performGridSearch(estimator=myClfTree, parameters_grid= params_classifier,
                            X_train=encoded_df, y_train=Y)

decisionTree.exportTree(treeToPrint=clf_OS, features_names=list(encoded_df),
           output_path=current_path+"/results/clf.dot")

# Cross-table of the decision tree - training set included
pd.crosstab(clf_OS.predict(encoded_df),Y)



# Decision tree into DataFrame


## Compute class enrichment



# Another framing of the question : will this mutation prevent my protein from interacting ?
    # Features :
