import logging
from math import ceil, log

import numpy as np
import scipy.stats
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import RandomizedSearchCV


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_cv(X, Y,
             n_est=[100, 200, 400, 800, 1600],
             max_depth=[3, 6, 12],
             learning_rate=scipy.stats.expon(0.1),
             subsample=[0.5, 0.75, 1],
             n_jobs=1):
    """
    Report 10-fold cross-validation scores for training.

    Parameters
    ----------
    X : csr_matrix
        (n x n_feats) feature matrix

    Y : csr_matrix
        n x 1 array of labels

    max_n_est : int
        Max number of estimators to use for the classifier. Additionally,
        all powers of 2 that are less than `max_n_est` will be tried.

    n_jobs : int
        Number of cores to use during cross-validation scoring. A value of -1
        will use all available cores.

    Returns
    -------
    train_scores : dict[int, float]
        Average training score during each CV training.

    test_scores : ????
        Average CV score
    """
    logger.info("Training GBM")

    Y = Y.toarray().ravel()
    cv = KFold(X.shape[0], n_folds=10, shuffle=True, random_state=92309)



    err_cv = {}



    # Xs = preprocessing.scale(Xs)
    # tuned_parameters = \
    #     {'kernel': ['rbf'], 'C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10],
    #      'gamma': ['auto', 0.0001, 0.001, 0.01, 0.1]}
    # mdl = SVR(epsilon=0.0001, tol=1e-10, verbose=False)
    # mdl = GridSearchCV(mdl, tuned_parameters, cv=10, verbose=VERBOSE)
    # mdl.fit(Xs, Ys)


    tuned_parameters = \
        {'n_estimators': n_est,
         'learning_rate': learning_rate,
         'max_depth': max_depth,
         'subsample': subsample}

    X = X.toarray()

    clf = GradientBoostingClassifier()
    clf = RandomizedSearchCV(clf, tuned_parameters, cv=cv, verbose=100)
    clf.fit(X, Y)
    logger.info("err: %f" % (1 - clf.best_score_))
    return clf

    # clf = GradientBoostingClassifier(n_estimators=max_n_est,
    #                                  learning_rate=learning_rate,
    #                                  max_depth=max_depth,
    #                                  subsample=subsample)
    # clf.fit(X, Y)
    # X = X.toarray()
    # score = clf.score(X, Y)
    # logger.info("n_est %d \t train_err: %f" % (max_n_est, 1 - score))



    # scores = cross_val_score(clf, X, Y, cv=cv, n_jobs=n_jobs)
    # logger.info("n_est: %d \t cv_err: %f" % (n_est, 1 - scores.mean()))
    # err_cv[n_est] = 1 - scores.mean()


    # train_scores, test_scores = validation_curve(
    #     SVC(), X, y, param_name="gamma", param_range=param_range,
    #     cv=10, scoring="accuracy", n_jobs=1)

    # return err_cv
