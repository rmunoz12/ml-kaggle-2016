import logging
from math import ceil, log

import numpy as np
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import zero_one_loss


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NNeighbors:
    @staticmethod
    def train(X, Y, nnbrs=5):
        """
        Train a nearest neighbor classifier.

        Parameters
        ----------
        X : csr_matrix
            (n x n_feats) feature matrix

        Y : csr_matrix
            n x 1 array of labels

        nnbrs : int
            Number of neighbors to use for the classifier.

        Returns
        -------
        clf : KNeighborsClassifier
            Classifier fit to X and Y
        """
        logger.info("Training nearest neighbors with nnbrs: %d" % nnbrs)

        Y = Y.toarray().ravel()


        clf = KNeighborsClassifier(n_neighbors=nnbrs)
        clf.fit(X, Y)
        pred_train = clf.predict(X)
        loss_train = sum(pred_train != Y)
        score_train = clf.score(X, Y)
        logger.info("r: {:3}\t loss: {:4d}\t error: {:.5f}"
                    .format(nnbrs, loss_train, 1 - score_train))
        return clf

    @staticmethod
    def train_cv(X, Y, max_n_est=10, n_jobs=1):
        """
        Report 10-fold cross-validation scores for training a nearest neighbor classifier.

        Parameters
        ----------
        X : csr_matrix
            (n x n_feats) feature matrix

        Y : csr_matrix
            n x 1 array of labels

        max_n_est : int
            Max number of estimators to use for the classifier.

        n_jobs : int
            Number of cores to use during cross-validation scoring. A value of -1
            will use all available cores.

        Returns
        -------
        err_cv : dict[int, float]
            Cross-validation errors for each value tested.
        """
        logger.info("Training cross-validated nearest neighbors")

        n_to_try = list(np.arange(1, 10, 3))
        if max_n_est != n_to_try[-1]:
            n_to_try.append(max_n_est)

        Y = Y.toarray().ravel()
        cv = KFold(X.shape[0], n_folds=10, shuffle=True, random_state=92309)

        err_cv = {}

        for n_est in n_to_try:
            clf = KNeighborsClassifier(n_neighbors=n_est)
            scores = cross_val_score(clf, X, Y, cv=cv, n_jobs=n_jobs)
            logger.info("n_est: %d \t cv_err: %f" % (n_est, 1 - scores.mean()))
            err_cv[n_est] = 1 - scores.mean()

        return err_cv


