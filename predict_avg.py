"""
Predict a positive label with the chance equal to the fraction of positive
labels in the training set.
"""

from __future__ import division

import logging
from argparse import ArgumentParser

from config import config
from ml2016.naive import predict_on_avg, fraction_positive
from ml2016.preprocess import load_data, extract_xy
from ml2016.submit import save_submission

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args():
    p = ArgumentParser(description="Predict positive with probability equal to "
                                   "fraction of positive labels in training "
                                   "set.")
    p.add_argument("-f", "--filename",
                   help="output filename (default: %(default)s)",
                   type=str, default='predict-avg.csv')
    args = p.parse_args()
    return args


def main():
    args = get_args()
    S, col_names_S = load_data(config.paths.training_data,
                               config.paths.cache_folder)
    Xs, Ys, col_names_S = extract_xy(S, col_names_S)

    frac_positive = fraction_positive(Ys)

    T, col_names_T = load_data(config.paths.test_data,
                               config.paths.cache_folder)
    logger.info('predicting with probability equal to avg')
    pred = predict_on_avg(T, frac_positive)
    save_submission(pred, config.paths.out_folder + args.filename)


if __name__ == '__main__':
    main()
