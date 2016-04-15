import logging
from time import time

from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.mixture import DPGMM

from .util import BaseClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Dpgmm(BaseClassifier):
    """
    Dirichlet Process Mixture Model wrapper class.

    Attributes
    ----------
    clf : DPGMM | None
        Classifier set only after calling `fit` or `tune`.
    """
    def __init__(self):
        super(Dpgmm, self).__init__()

    def fit(self, X, Y, n_components=10, alpha=1.0, covariance_type='full', tol = 1e-3):
        """
        Train a Dirichlet Process Mixture Model classifier.

        Sets `self.clf` equal to the trained classifier.

        Parameters
        ----------
        X : csr_matrix
            (n x n_feats) feature matrix

        Y : csr_matrix
            n x 1 array of labels

        n_components : int
            Number of clusters.

        alpha : float
            Concentration parameter.

        covariance_type : str
            How sparse the covariance should be.

        tol : float
            Convergence threshold for log likelihood.

        Returns
        -------
        score_train : float
            Score of the classifier on the training set.
        """
        logger.info("Training Dirichlet Process Mixture Model classifier "
                    "   <n_components=%s, alpha=%s, covariance_type=%s, tol=%0.5f>"
                    % (str(n_components), str(alpha), covariance_type, tol))

        Y = Y.toarray().ravel()

        self.clf = DPGMM(n_components=n_components, alpha=alpha, covariance_type=covariance_type, tol=tol)
        self.clf.fit(X, Y)
        score_train = self.clf.score(X, Y)
        logger.info("Training score: %0.5f" % score_train)
        return score_train

    def tune(self, X, Y, n_components=10, alpha=1.0, covariance_type='full', tol=1e-3, n_jobs=1,
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

        n_estimators : int
            Number of voting trees.

        max_depth : int
            Max depth of tree.

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
        if not isinstance(n_components, list):
            n_esimators = [n_components]
        if not isinstance(alpha, list):
            alpha = [alpha]
        if not isinstance(covariance_type, list):
            covariance_type = [covariance_type]
        if not isinstance(tol, list):
            tol = [tol]

        logger.info("Grid searching (10-fold cv)")
        start_time = time()

        param_grid = [{'n_components': n_components},
                      {'alpha': alpha},
                      {'covariance_type': covariance_type},
                      {'tol': tol}]

        Y = Y.toarray().ravel()
        cv = KFold(X.shape[0], n_folds=10, shuffle=True, random_state=92309)

        mdl = DPGMM()
        self.clf = GridSearchCV(mdl, param_grid=param_grid, n_jobs=n_jobs,
                                cv=cv, verbose=verbose)
        self.clf.fit(X.toarray(), Y)
        self.report_cv_scores()

        logger.info("--- %0.3f minutes ---" % ((time() - start_time) / 60))
        return self.clf.best_params_

    def score(self, X, Y):
        """
        Use `self.clf` to score predictions on labels `Y`.

        Need to convert to dense 'X' array for KNN.
        """
        return self.clf.score(X.toarray(), Y.toarray().ravel())

    def predict(self, X):
        """
        Use `self.clf` to predict on the set `X`.

        Need to convert to dense 'X' array for KNN.
        """
        return self.clf.predict(X.toarray())
