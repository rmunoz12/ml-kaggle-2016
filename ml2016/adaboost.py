from __future__ import division
import logging
from time import time

from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import KFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.grid_search import GridSearchCV

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

    def fit(self, X, Y, max_depth=1, n_estimators=50, learning_rate=1):
        """
        Train an Adaboost classifier decision trees.

        Sets `self.clf` equal to the trained classifier.

        Parameters
        ----------
        X : csr_matrix
            (n x n_feats) feature matrix

        Y : csr_matrix
            n x 1 array of labels

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

    def tune(self, X, Y, max_depth=1, n_estimators=50, learning_rate=1,
             n_jobs=1, verbose=0):
        """
        Report 10-fold cross-validation scores for tuning `X` on `Y` using
        a grid search over the hyper-parameters.

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

        logger.info("--- CV Scores ---")
        for params, mean_cv_score, cv_scores in self.clf.grid_scores_:
            logger.info("cv score: %0.5f (+/-%0.05f) for %r"
                        % (mean_cv_score, cv_scores.std() * 2, params))

        logger.info("--- Summary ---")
        logger.info("Best parameters: %s" % self.clf.best_params_)
        logger.info("Best score: %0.5f" % self.clf.best_score_)
        logger.info("--- %0.3f minutes ---" % ((time() - start_time) / 60))
        return self.clf.best_params_
