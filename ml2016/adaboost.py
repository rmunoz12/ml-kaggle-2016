import logging
import time

import numpy as np
import scipy.io as sio
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_adaboost(X, Y, n_est=100):
    logger.info("Training adaboost with max iterations: %d" % n_est)

    Y = Y.toarray()
    Y = Y.ravel()

    # clf = DecisionTreeClassifier(max_depth=1)
    clf = AdaBoostClassifier(n_estimators=n_est)
    clf.fit(X, Y)
    pred_train = clf.predict(X)
    loss_train = sum(pred_train != Y)
    score_train = clf.score(X, Y)
    logger.info("r: {:3}\t loss: {:4d}\t error: {:.5f}"
                .format(n_est, loss_train, 1 - score_train))
    return clf



