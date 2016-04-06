"""
Predict a positive label with the chance equal to the fraction of positive
labels in the training set.
"""

from __future__ import division

import logging
from argparse import ArgumentParser
from copy import copy

from matplotlib import pyplot as plt
import numpy as np

from config import config
from ml2016.adaboost import train_adaboost, train_cv
from ml2016.preprocess import drop_feature, load_data, extract_xy
from ml2016.submit import save_submission

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args():
    p = ArgumentParser(description="Predict using adaboost")
    p.add_argument("-f", "--filename",
                   help="output filename (default: %(default)s)",
                   type=str, default='predict-adaboost.csv')
    p.add_argument('-n', '--n-est',
                   help="max adaboost estimators (default: %(default)s)",
                   type=int, default=100)
    p.add_argument('--ignore-high-d-feats',
                   help='ignores features 23 and 58',
                   action='store_true')
    args = p.parse_args()
    return args


def plot(err_cv):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(len(err_cv)) + 1, err_cv,
            label='Real AdaBoost CV Error')
    ax.set_ylim((0.0, 0.5))
    ax.set_xlabel('n_estimators')
    ax.set_ylabel('error rate')
    leg = ax.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.7)
    plt.show()


def main():
    args = get_args()
    S, col_names_S = load_data(config.paths.training_data,
                               config.paths.feat_types,
                               config.paths.cache_folder)
    Xs, Ys, col_names_S = extract_xy(S, col_names_S)

    if args.ignore_high_d_feats:
        names = copy(col_names_S)
        for name in names:
            if name[:2] == '23' or name[:2] == '58':
                Xs, col_names_S = drop_feature(Xs, col_names_S, name)


    err_cv = train_cv(Xs, Ys, args.n_est)
    logger.info("minimum cv error: %f" % min(err_cv))

    plot(err_cv)

    logger.info("training cross-validated classifier")
    clf = train_adaboost(Xs, Ys, list(err_cv).index(min(err_cv)))

    T, col_names_T = load_data(config.paths.test_data,
                               config.paths.feat_types,
                               config.paths.cache_folder)
    # Xt, Yt, col_names_t = extract_xy(T, col_names_T)
    Xt = T

    if args.ignore_high_d_feats:
        names = copy(col_names_T)
        for name in names:
            if name[:2] == '23' or name[:2] == '58':
                Xt, col_names_T = drop_feature(Xt, col_names_T, name)

    logger.info('predicting test set labels')
    test_pred = clf.predict(Xt)
    pred = {}
    for i, lbl in enumerate(test_pred):
        pred[i + 1] = lbl
    save_submission(pred, config.paths.out_folder + args.filename)


if __name__ == '__main__':
    main()
