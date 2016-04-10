"""
Utilities and base classes.
"""

import logging

LOGGER = logging.getLogger(__name__)


class BaseClassifier(object):
    """
    Base classification wrapper.

    This class should not be used directly, but instead should be derived.

    Attributes
    -----
    clf : object
        A classifier object following the scikit-learn interface.
    """

    def __init__(self):
        self.clf = None

    def fit(self):
        """Used to fit a classifier to training data."""
        raise NotImplementedError

    def tune(self):
        """Used to tune hyper-parameters."""
        raise NotImplementedError

    def predict(self, X):
        """Use `self.clf` to predict on the set `X`."""
        return self.clf.predict(X)

    def score(self, X, Y):
        """Use `self.clf` to score predictions on labels `Y`."""
        return self.clf.score(X, Y.toarray().ravel())

    def report_cv_scores(self, logger=LOGGER):
        """Logs results of `self.clf` tuned with GridSearchCV"""
        logger.info("--- CV Scores ---")
        for params, mean_cv_score, cv_scores in self.clf.grid_scores_:
            logger.info("cv score: %0.5f (+/-%0.05f) for %r"
                        % (mean_cv_score, cv_scores.std() * 2, params))

        logger.info("--- Summary ---")
        logger.info("Best parameters: %s" % self.clf.best_params_)
        logger.info("Best cv score: %0.5f" % self.clf.best_score_)

    def __str__(self):
        return "<%s %s>" % (self.__class__.__name__, self.__dict__)
