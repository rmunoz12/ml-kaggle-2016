from __future__ import division

import logging
from argparse import ArgumentParser

from config import config
from ml2016.logistic_reg import LogisticReg
from ml2016.preprocess import load_data, extract_xy, remove_cols
from ml2016.submit import save_submission

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args():
    p = ArgumentParser(description="Predict using adaboost")
    p.add_argument("-f", "--filename",
                   help="output filename (default: %(default)s)",
                   type=str, default='predict-logit.csv')
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


    mdl = LogisticReg()
    mdl.tune(Xs, Ys, C=1.0,n_jobs=args.jobs, verbose=verbose)

    logger.info("Training score: %0.5f" % mdl.score(Xs, Ys))

    T, col_names_T = load_data(config.paths.test_data,
                               config.paths.feat_types,
                               config.paths.cache_folder)
    Xt = T

    if args.ignore_high_d_feats:
        Xt, col_names_T = remove_cols(Xt, col_names_T)

    logger.info('predicting test set labels')
    test_pred = mdl.predict(Xt)
    pred = {}
    for i, lbl in enumerate(test_pred):
        pred[i + 1] = int(lbl)
    save_submission(pred, config.paths.out_folder + args.filename)


if __name__ == '__main__':
    main()
