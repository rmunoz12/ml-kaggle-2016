import logging
from time import time

from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.cluster import KMeans

from .util import BaseClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Kmeans(BaseClassifier):
    """
    K-means wrapper class.

    Attributes
    ----------
    clf : KMeans | None
        Classifier set only after calling `fit` or `tune`.
    """
    def __init__(self):
        super(Kmeans, self).__init__()

    def fit(self, X, Y, n_clusters=8):
        """
        Train a K-means classifier.

        Sets `self.clf` equal to the trained classifier.

        Parameters
        ----------
        X : csr_matrix
            (n x n_feats) feature matrix

        Y : csr_matrix
            n x 1 array of labels

        n_clusters : int
            Number of clusters.

        Returns
        -------
        score_train : float
            Score of the classifier on the training set.
        """
        logger.info("Training K-Means classifier <n_clusters=%s>"
                    % (str(n_clusters)))

        Y = Y.toarray().ravel()

        self.clf = KMeans(n_clusters=n_clusters)
        self.clf.fit(X, Y)
        score_train = self.clf.score(X, Y)
        logger.info("Training score: %0.5f" % score_train)
        return score_train

    def tune(self, X, Y, n_clusters=8, n_jobs=1,
             verbose=0):
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
            n x 1 array of labels

        n_clusters : int
            Number of clusters.

        n_jobs : int
            Number of cores to use during cross-validation scoring. A value of -1
            will use all available cores.

        verbose : int
            Verbosity of GridSearchCV, higher values output more messages.

        Returns
        -------
        self.clf.best_params_ : dict[str, T]
            Contains the best parameter values found.
        """
        if not isinstance(n_clusters, list):
            n_esimators = [n_clusters]

        logger.info("Grid searching (10-fold cv)")
        start_time = time()

        param_grid = [{'n_clusters': n_clusters}]

        Y = Y.toarray().ravel()
        cv = KFold(X.shape[0], n_folds=10, shuffle=True, random_state=92309)

        mdl = KMeans()
        self.clf = GridSearchCV(mdl, param_grid=param_grid, n_jobs=n_jobs,
                                cv=cv, verbose=verbose)
        self.clf.fit(X, Y)
        self.report_cv_scores()

        logger.info("--- %0.3f minutes ---" % ((time() - start_time) / 60))
        return self.clf.best_params_

    def score(self, X, Y):
        """
        Use `self.clf` to score predictions on labels `Y`.
        """
        return self.clf.score(X.toarray(), Y.toarray().ravel())

    def predict(self, X):
        """
        Use `self.clf` to predict on the set `X`.
        """
        return self.clf.predict(X.toarray())
