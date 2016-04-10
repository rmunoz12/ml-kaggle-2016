"""
Predict a positive label with the chance equal to the fraction of positive
labels in the training set.
"""

from __future__ import division

import logging
from argparse import ArgumentParser
from copy import copy

from matplotlib import pyplot as plt

from config import config
from ml2016.adaboost import Adaboost
from ml2016.preprocess import drop_feature, load_data, extract_xy
from ml2016.submit import save_submission

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args():
    p = ArgumentParser(description="Predict using adaboost")
    p.add_argument("-f", "--filename",
                   help="output filename (default: %(default)s)",
                   type=str, default='predict-adaboost.csv')
    p.add_argument('-j', '--jobs',
                   help='number of cv processes (default: %(default)d)',
                   type=int, default=1)
    p.add_argument('--ignore-high-d-feats',
                   help='ignores features 23 and 58',
                   action='store_true')
    p.add_argument('--verbose',
                   help='grid search verbosity level',
                   action='store_true')
    args = p.parse_args()
    return args


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


def remove_cols(X, col_names):
    names = copy(col_names)
    for name in names:
        if name[:2] == '23' or name[:2] == '58':
            X, col_names = drop_feature(X, col_names, name)
    return X, col_names


def main():
    args = get_args()
    verbose = 0
    if args.verbose:
        verbose = 100

    S, col_names_S = load_data(config.paths.training_data,
                               config.paths.feat_types,
                               config.paths.cache_folder)
    Xs, Ys, col_names_S = extract_xy(S, col_names_S)

    if args.ignore_high_d_feats:
        Xs, col_names_S = remove_cols(Xs, col_names_S)


    mdl = Adaboost()
    best_params = mdl.tune(Xs, Ys, max_depth=[1, 3], n_estimators=[1, 2],
                           learning_rate=[1], n_jobs=args.jobs, verbose=verbose)
    best_params['max_depth'] = best_params['base_estimator'].max_depth
    best_params.pop('base_estimator')

    logger.info("Training model with best params")

    final_mdl = Adaboost()
    final_mdl.fit(Xs, Ys, **best_params)

    T, col_names_T = load_data(config.paths.test_data,
                               config.paths.feat_types,
                               config.paths.cache_folder)
    # Xt, Yt, col_names_t = extract_xy(T, col_names_T)
    Xt = T

    if args.ignore_high_d_feats:
        Xt, col_names_T = remove_cols(Xt, col_names_T)

    logger.info('predicting test set labels')
    test_pred = final_mdl.predict(Xt)
    pred = {}
    for i, lbl in enumerate(test_pred):
        pred[i + 1] = int(lbl)
    save_submission(pred, config.paths.out_folder + args.filename)


if __name__ == '__main__':
    main()
