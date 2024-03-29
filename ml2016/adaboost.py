from __future__ import division
import logging
from time import time

from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import KFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from .util import BaseClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Adaboost(BaseClassifier):
    """
    Adaboost wrapper class.

    Attributes
    ----------
    clf : AdaBoostClassifier | None
        Classifier set only after calling `fit` or `tune`.
    """
    def __init__(self):
        super(Adaboost, self).__init__()

    def fit(self, X, Y, max_depth=1, n_estimators=50, learning_rate=1.0):
        """
        Train an Adaboost classifier using decision trees.

        Sets `self.clf` equal to the trained classifier.

        Parameters
        ----------
        X : csr_matrix
            (n x n_feats) feature matrix

        Y : csr_matrix
            (n x 1) array of labels

        n_estimators : int
            Number of estimators to use for the classifier.

        learning_rate : float
            Rate of decreasing estimator contribution.

        Returns
        -------
        score_train : float
            Score of the classifier on the training set.
        """
        logger.info("Training Adaboost "
                    "<max_depth=%d, n_estimators=%d, learning_rate=%f>"
                    % (max_depth, n_estimators, learning_rate))
        base_estimator = DecisionTreeClassifier(max_depth=max_depth)
        self.clf = AdaBoostClassifier(base_estimator=base_estimator,
                                      n_estimators=n_estimators,
                                      learning_rate=learning_rate)
        Y = Y.toarray().ravel()
        self.clf.fit(X, Y)
        score_train = self.clf.score(X, Y)
        logger.info("Training score: %0.5f" % score_train)
        return score_train

    def tune(self, X, Y, max_depth=1, n_estimators=50, learning_rate=1.0,
             n_jobs=1, verbose=0):
        """
        Report 10-fold cross-validation scores for tuning `X` on `Y` using
        a grid search over the hyper-parameters.

        Sets `self.clf` equal to the classifier training on the best
        parameters using the full set `X`.

        Parameters
        ----------
        X : csr_matrix
            (n x n_feats) feature matrix

        Y : csr_matrix
            (n x 1) array of labels

        max_depth : int | list[int]
            The individual tree depth or list of depths to try.

        n_estimators : int | list[int]
            If int, then number of estimators to use for the classifier. If
            list, then all values in the list will be tried.

        learning_rate : float | list[float]
            The learning_rate or list of learning rates to try.

        n_jobs : int
            Number of cores to use during cross-validation scoring. A value
            of -1 will use all available cores.

        verbose : int
            Verbosity of GridSearchCV, higher values output more messages.

        Returns
        -------
        self.clf.best_params_ : dict[str, T]
            Contains the best parameter values found.
        """
        if not isinstance(max_depth, list):
            max_depth = [max_depth]
        if not isinstance(n_estimators, list):
            n_estimators = [n_estimators]
        if not isinstance(learning_rate, list):
            learning_rate = [learning_rate]

        logger.info("Grid searching (10-fold cv)")
        start_time = time()

        base_estimators = \
            [DecisionTreeClassifier(max_depth=d) for d in max_depth]

        param_grid = [{'base_estimator': base_estimators,
                       'n_estimators': n_estimators,
                       'learning_rate': learning_rate}]

        Y = Y.toarray().ravel()
        cv = KFold(X.shape[0], n_folds=10, shuffle=True, random_state=92309)
        mdl = AdaBoostClassifier()
        self.clf = GridSearchCV(mdl, param_grid=param_grid, n_jobs=n_jobs,
                                cv=cv, verbose=verbose)

        self.clf.fit(X, Y)
        self.report_cv_scores()

        logger.info("--- %0.3f minutes ---" % ((time() - start_time) / 60))
        return self.clf.best_params_


class PfAdaBoost(Adaboost):
    """
    Plynomial-Features input to Adaboost wrapper class.

    Attributes
    ----------
    clf : AdaBoostClassifier | None
        Classifier set only after calling `fit` or `tune`.
    """
    def __init__(self):
        super(PfAdaBoost, self).__init__()

    def fit(self, X, Y, max_depth=1, n_estimators=50, learning_rate=1.0,
            degree=2):
        """
        Train an Adaboost classifier using decision trees.

        Sets `self.clf` equal to the trained classifier.

        Parameters
        ----------
        X : csr_matrix
            (n x n_feats) feature matrix

        Y : csr_matrix
            (n x 1) array of labels

        n_estimators : int
            Number of estimators to use for the classifier.

        learning_rate : float
            Rate of decreasing estimator contribution.

        degree : int | list[int]
            Degree of polynomial features.

        Returns
        -------
        score_train : float
            Score of the classifier on the training set.
        """
        logger.info("Training Pf-Adaboost "
                    "<degree = %d, max_depth=%d, n_estimators=%d, learning_rate=%f>"
                    % (degree, max_depth, n_estimators, learning_rate))
        base_estimator = DecisionTreeClassifier(max_depth=max_depth)
        self.clf = Pipeline(steps=[('pf', PolynomialFeatures()),
                                   ('ab', AdaBoostClassifier())])
        self.clf.set_params(pf__degree=degree,
                            ab__base_estimator=base_estimator,
                            ab__n_estimators=n_estimators,
                            ab__learning_rate=learning_rate)
        X = X.toarray()
        Y = Y.toarray().ravel()
        self.clf.fit(X, Y)
        score_train = self.clf.score(X, Y)
        logger.info("Training score: %0.5f" % score_train)
        return score_train

    def tune(self, X, Y, max_depth=1, n_estimators=50, learning_rate=1.0,
             degree=2, n_jobs=1, verbose=0):
        """
        Report 10-fold cross-validation scores for tuning `X` on `Y` using
        a grid search over the hyper-parameters.

        Sets `self.clf` equal to the classifier training on the best
        parameters using the full set `X`.

        Parameters
        ----------
        X : csr_matrix
            (n x n_feats) feature matrix

        Y : csr_matrix
            (n x 1) array of labels

        max_depth : int | list[int]
            The individual tree depth or list of depths to try.

        n_estimators : int | list[int]
            If int, then number of estimators to use for the classifier. If
            list, then all values in the list will be tried.

        learning_rate : float | list[float]
            The learning_rate or list of learning rates to try.

        degree : int | list[int]
            Degree of polynomial features.

        n_jobs : int
            Number of cores to use during cross-validation scoring. A value
            of -1 will use all available cores.

        verbose : int
            Verbosity of GridSearchCV, higher values output more messages.

        Returns
        -------
        self.clf.best_params_ : dict[str, T]
            Contains the best parameter values found.
        """
        if not isinstance(max_depth, list):
            max_depth = [max_depth]
        if not isinstance(n_estimators, list):
            n_estimators = [n_estimators]
        if not isinstance(learning_rate, list):
            learning_rate = [learning_rate]
        if not isinstance(degree, list):
            degree = [degree]

        logger.info("Grid searching (10-fold cv)")
        start_time = time()

        base_estimators = \
            [DecisionTreeClassifier(max_depth=d) for d in max_depth]

        mdl = Pipeline(steps=[('pf', PolynomialFeatures()),
                              ('ab', AdaBoostClassifier())])

        param_grid = [{'pf__degree': degree,
                       'ab__base_estimator': base_estimators,
                       'ab__n_estimators': n_estimators,
                       'ab__learning_rate': learning_rate}]

        X = X.toarray()
        Y = Y.toarray().ravel()
        cv = KFold(X.shape[0], n_folds=10, shuffle=True, random_state=92309)

        self.clf = GridSearchCV(mdl, param_grid=param_grid, n_jobs=n_jobs,
                                cv=cv, verbose=verbose)

        self.clf.fit(X, Y)
        self.report_cv_scores()

        logger.info("--- %0.3f minutes ---" % ((time() - start_time) / 60))
        return self.clf.best_params_
