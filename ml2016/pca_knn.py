import logging
from time import time

from sklearn.cross_validation import KFold
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from .nneighbors import NNeighbors

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PrinCompKNN(NNeighbors):
    """
    Wrapper class that does PCA followed by KNN.

    Attributes
    ----------
    clf : object | None
        Classifier set only after calling `fit` or `tune`.
    """
    def __init__(self):
        super(PrinCompKNN, self).__init__()

    def fit(self, X, Y, n_components=5, n_neighbors=5):
        """
        Train a nearest neighbor classifier.

        Sets `self.clf` equal to the trained classifier.

        Parameters
        ----------
        X : csr_matrix
            (n x n_feats) feature matrix

        Y : csr_matrix
            n x 1 array of labels

        n_components : int | None
            Number of principal components to use. If none, uses all components.

        n_neighbors : int
            Number of neighbors to use for the classifier.

        Returns
        -------
        score_train : float
            Score of the classifier on the training set.
        """
        logger.info("Training PCA=KNN <n_components=%d, n_neighbors=%d>"
                    % (n_components, n_neighbors))
        X = X.toarray()
        Y = Y.toarray().ravel()
        self.clf = Pipeline(steps=[('pca', PCA()),
                                   ('knn', KNeighborsClassifier)])
        self.clf.set_params(pca__n_components= n_components,
                            knn__n_neighbors=n_neighbors)
        self.clf.fit(X, Y)
        score_train = self.clf.score(X, Y)
        logger.info("Training score: %0.5f" % score_train)
        return score_train

    def tune(self, X, Y, n_components=5, n_neighbors=1, n_jobs=1,
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

        n_neighbors : int | list[int]
            Number of neighbors, k, or list of k values to try.

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
        if not isinstance(n_neighbors, list):
            n_neighbors = [n_neighbors]

        logger.info("Grid searching (10-fold cv)")
        start_time = time()

        mdl = Pipeline(steps=[('pca', PCA()),
                              ('knn', KNeighborsClassifier())])

        param_grid = [{'pca__n_components': n_components,
                       'knn__n_neighbors': n_neighbors}]
        X = X.toarray()
        Y = Y.toarray().ravel()
        cv = KFold(X.shape[0], n_folds=10, shuffle=True, random_state=92309)

        self.clf = GridSearchCV(mdl, param_grid=param_grid, n_jobs=n_jobs,
                                cv=cv, verbose=verbose)
        self.clf.fit(X, Y)
        self.report_cv_scores()

        logger.info("--- %0.3f minutes ---" % ((time() - start_time) / 60))
        return self.clf.best_params_


