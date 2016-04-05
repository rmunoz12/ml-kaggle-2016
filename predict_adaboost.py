"""
Predict a positive label with the chance equal to the fraction of positive
labels in the training set.
"""

from __future__ import division

import logging
from argparse import ArgumentParser

from config import config
from ml2016.adaboost import train_adaboost
from ml2016.preprocess import load_data, extract_xy
from ml2016.submit import save_submission

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args():
    p = ArgumentParser(description="Predict using adaboost")
    p.add_argument("-f", "--filename",
                   help="output filename (default: %(default)s)",
                   type=str, default='predict-avg.csv')
    p.add_argument('-n', '--n-est',
                   help="max adaboost estimators (default: %(default)s)",
                   type=int, default=10000)
    args = p.parse_args()
    return args


def main():
    args = get_args()
    S, col_names_S = load_data(config.paths.training_data,
                               config.paths.feat_types,
                               config.paths.cache_folder)
    Xs, Ys, col_names_S = extract_xy(S, col_names_S)

    clf = train_adaboost(Xs, Ys, args.n_est)

    # T, col_names_T = load_data(config.paths.test_data,
    #                            config.paths.feat_types,
    #                            config.paths.cache_folder)
    logger.info('predicting with probability equal to avg')
    # pred = predict_on_avg(T, frac_positive)
    # save_submission(pred, config.paths.out_folder + args.filename)


if __name__ == '__main__':
    main()
