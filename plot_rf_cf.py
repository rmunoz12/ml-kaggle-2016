from __future__ import division

import logging
from argparse import ArgumentParser

from config import config
from ml2016.adaboost import Adaboost, PfAdaBoost
from ml2016.pca_knn import PrinCompKNN
from ml2016.logistic_reg import LogisticReg
from ml2016.nneighbors import NNeighbors
from ml2016.preprocess import load_data, extract_xy
from ml2016.svm import SVM
from ml2016.randomforest import RandomForest
from ml2016.kmeans import Kmeans
from ml2016.gmm import Gmm
from ml2016.dpgmm import Dpgmm
from ml2016.submit import save_submission

from sklearn.learning_curve import validation_curve
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import sys, operator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():

    S, col_names_S = load_data(config.paths.training_data,
                               config.paths.cache_folder)
    Xs, Ys, col_names_S = extract_xy(S, col_names_S)

    a = RandomForestClassifier(n_estimators=1)
    a.fit(Xs.toarray(), Ys.toarray().ravel())
    best_features = a.feature_importances_
    max_ind, max_val = max(enumerate(best_features), key=operator.itemgetter(1))
    print best_features
    print max_ind, max_val

    print Xs.shape
    print Ys.shape
    param_range = [1, 3, 5, 7, 10, 15, 20, 30, 60, 80]
    train_scores, test_scores = validation_curve(RandomForestClassifier(criterion='entropy'), Xs, Ys.toarray().ravel(),
                                                 'n_estimators', param_range)

    print train_scores
    print test_scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.title("Validation Curve for Random Forest")
    plt.xlabel("Number of Trees")
    plt.ylabel("Score")
    plt.plot(param_range, train_mean, label="Training Score", color='r')
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.2, color='r')
    plt.plot(param_range, test_mean, label="Test Score", color='b')
    plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.2, color='b')
    plt.legend(loc="best")
    plt.show()

if __name__ == '__main__':
    main()
