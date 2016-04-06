import logging
from math import ceil, log

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import zero_one_loss


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_adaboost(X, Y, n_est=100):
    logger.info("Training adaboost with n_estimators: %d" % n_est)

    Y = Y.toarray().ravel()

    # clf = DecisionTreeClassifier(max_depth=1)
    clf = AdaBoostClassifier(n_estimators=n_est)
    clf.fit(X, Y)
    pred_train = clf.predict(X)
    loss_train = sum(pred_train != Y)
    score_train = clf.score(X, Y)
    logger.info("r: {:3}\t loss: {:4d}\t error: {:.5f}"
                .format(n_est, loss_train, 1 - score_train))
    return clf


def train_cv(X, Y, max_n_est=100, n_jobs=1):
    logger.info("Training cross-validated adaboost")

    n_to_try = list(range(int(ceil(log(max_n_est, 2)))))
    n_to_try = [2 ** n for n in n_to_try]
    if max_n_est != n_to_try[-1]:
        n_to_try.append(max_n_est)

    Y = Y.toarray().ravel()
    cv = KFold(X.shape[0], n_folds=10, shuffle=True, random_state=92309)
    err_cv = np.zeros((max_n_est,))

    for n_est in n_to_try:
        clf = AdaBoostClassifier(n_estimators=n_est)
        scores = cross_val_score(clf, X, Y, cv=cv, n_jobs=n_jobs)
        logger.info("n_est: %d \t cv_err: %f" % (n_est, 1 - scores.mean()))
        err_cv[n_est - 1] = 1 - scores.mean()

    return err_cv


