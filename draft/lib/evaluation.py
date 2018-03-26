from __future__ import print_function
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from scipy import interp
import dataProcessing
import itertools
import ast


def plotLearningCurve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.
    This curves depends on the number of training examples.
    Adapted from scikit-learn.org.

    ARGUMENTS
    estimator : object type that implements the "fit" and "predict" methods
    title : string for title
    X : training vector
    y : target relative to X
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt




def performGridSearch(estimator, parameters_grid, X_train, y_train, scorer=None, verbose = 0):
    """
    Performs Grid search and returns the best estimations for the parameters

    ARGUMENTS
    estimator = an estimator, to be used
    parameters_grid = a hash giving, for each parameter, an array of values to be tested
    X_train & y_train = data to train the estimator
    scorer = call to a custom score function, such as one returned by make_scorer. Optional.
    verbose = how verbose should we be ?

    RETURNS : a list of respectively the curve, the TPR and the FPR arrays
    """
    grid = GridSearchCV(estimator=estimator, param_grid=parameters_grid, scoring=scorer, verbose=verbose)
    grid.fit(X_train, y_train)

    # Printing
    print("Best score :")
    print(grid.best_score_)

    return grid.best_estimator_







def visualizeContingencyTableCertitude(X,y,model,cert_mini, labels=None):
    """Given X, y and a DNN model, will predict X with this model
    and draw a contingency table of the predictions against the given y
    with respect to the provided certitude (0 to 1)"""
    a = pd.Series(dataProcessing.multiClassIntoNumeric(model.predict(X), certitude = cert_mini)).fillna('UNDEFINED')
    foo = pd.Categorical(a, set(a))
    bar = pd.Categorical(y, labels)
    return pd.crosstab(foo,bar)







def whoWasMisclassified(y_true, y_pred,
                        true_status, selected_mistake):
    """
    Given y_true (must be a pandas Series) and y_pred, will return the IDs of the examples.
    of true_status tha were mistakenly identified as belonging to the class selected_mistake
    """
    y_pred = pd.Series(y_pred)  # Conversion of y_pred into a Series
    pairwise_comparison = [t == true_status and p == selected_mistake for t, p in zip(y_true,y_pred)]

    # Output the IDs from y_true, not y_pred !
    selected_mistakes = y_true[pairwise_comparison]
    return list(selected_mistakes.index.values)
