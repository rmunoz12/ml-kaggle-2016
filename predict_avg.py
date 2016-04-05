"""
Predict a positive label with the chance equal to the fraction of positive
labels in the training set.
"""

from __future__ import division
from argparse import ArgumentParser
import logging

from config import config
from ml2016.preprocess import load_data, extract_xy
from ml2016.predict import predict_on_avg, save_submission


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
                               config.paths.feat_types,
                               config.paths.cache_folder)
    Xs, Ys, col_names_S = extract_xy(S, col_names_S)

    sum_positive = sum([1 if v == 1 else 0 for v in Ys.toarray()])
    frac_positive = sum_positive / Ys.shape[0]

    T, col_names_T = load_data(config.paths.test_data,
                               config.paths.feat_types,
                               config.paths.cache_folder)
    logger.info('predicting with probability equal to avg')
    pred = predict_on_avg(T, frac_positive)
    save_submission(pred, config.paths.out_folder + args.filename)


if __name__ == '__main__':
    main()
