"""
Predict a positive label with the chance equal to the fraction of positive
labels in the training set.
"""

from __future__ import division

import logging
from argparse import ArgumentParser

from matplotlib import pyplot as plt
import numpy as np

from config import config
from ml2016.adaboost import train_adaboost, train_cv
from ml2016.preprocess import load_data, extract_xy
from ml2016.submit import save_submission

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot(err_cv):
    """
    Plots number of estimators vs cross-validation error.

    Parameters
    ----------
    err_cv : dict[int, float]
        Cross-validation errors
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    n_est, errs = [], []
    for k in sorted(err_cv):
        v = err_cv[k]
        n_est.append(k)
        errs.append(v)

    ax.plot(n_est, errs,
            label='Real AdaBoost CV Error')
    ax.set_ylim((0.0, 0.5))
    ax.set_xlabel('n_estimators')
    ax.set_ylabel('error rate')
    leg = ax.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.7)
    plt.show()


def get_args():
    p = ArgumentParser(description="Predict using adaboost")
    p.add_argument("-f", "--filename",
                   help="output filename (default: %(default)s)",
                   type=str, default='predict-adaboost.csv')
    p.add_argument('-n', '--n-est',
                   help="max adaboost estimators (default: %(default)s)",
                   type=int, default=100)
    p.add_argument('-j', '--jobs',
                   help='number of cv processes (default: %(default)d)',
                   type=int, default=1)
    args = p.parse_args()
    return args


def main():
    args = get_args()
    S, col_names_S = load_data(config.paths.training_data,
                               config.paths.feat_types,
                               config.paths.cache_folder)
    Xs, Ys, col_names_S = extract_xy(S, col_names_S)


    err_cv = train_cv(Xs, Ys, args.n_est, args.jobs)

    plot(err_cv)

    logger.info("training cross-validated classifier")

    # n = list(err_cv).index(min(err_cv))
    min_err = min(err_cv.values())
    n = min([k for k, v in err_cv.items() if v == min_err])
    logger.info("minimum cv error: %f" % min_err)

    clf = train_adaboost(Xs, Ys, n)

    T, col_names_T = load_data(config.paths.test_data,
                               config.paths.feat_types,
                               config.paths.cache_folder)
    # Xt, Yt, col_names_t = extract_xy(T, col_names_T)
    Xt = T

    logger.info('predicting test set labels')
    test_pred = clf.predict(Xt)
    pred = {}
    for i, lbl in enumerate(test_pred):
        pred[i + 1] = int(lbl)
    save_submission(pred, config.paths.out_folder + args.filename)


if __name__ == '__main__':
    main()
