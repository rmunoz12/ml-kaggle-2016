from __future__ import division

import logging
from argparse import ArgumentParser

from matplotlib import pyplot as plt

from config import config
from ml2016.adaboost import Adaboost
from ml2016.logistic_reg import LogisticReg
from ml2016.nneighbors import NNeighbors
from ml2016.preprocess import load_data, extract_xy, remove_cols
from ml2016.submit import save_submission

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


ALGORITHM_CHOICES = ['adaboost', 'knn', 'logit']


def get_args():
    p = ArgumentParser(description="Predictions")

    p.add_argument('algorithm',
                   help='machine learning algorithm to run '
                        '(choices: %s)'
                        % ", ".join(ALGORITHM_CHOICES),
                   type=str)

    p.add_argument("-f", "--filename",
                   help="output filename, if not given a default name is used",
                   type=str, default=None)
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


def choose_model(args):
    """
    Switch function used to return a model object and load
    its corresponding parameters from `config` and `args`.

    Parameters
    ----------
    args :
        args.algorithm : str
            Name of algorithm, must be in `ALGORITHM_CHOICES`.

        args.jobs : int
            Number of processes. Value of -1 indicates use all cores.

        args.verbose : None | bool
            If true, then set verbose to a high level.

    Returns
    -------
    mdl : object
        scikit-learn style classifier object.

    params : dict[str, T]
        Model parameters to pass to `mdl.tune`

    Raises
    ------
    ValueError
        If args.algorithm is not in `ALGORITHM_CHOICES`

    NotImplementedError
        If args.algorithm recognized, but has not yet be handled
        herein.
    """
    params = {'n_jobs': args.jobs,
              'verbose': 100 if args.verbose else 0}
    if args.algorithm not in ALGORITHM_CHOICES:
        logger.error("unrecognized algorithm: %s" % args.algorithm)
        raise ValueError
    elif args.algorithm == 'adaboost':
        mdl = Adaboost()
        params.update(config.params.adaboost)
    elif args.algorithm == 'knn':
        mdl = NNeighbors()
        params.update(config.params.knn)
    elif args.algorithm == 'logit':
        mdl = LogisticReg()
        params.update(config.params.logit)
    else:
        raise NotImplementedError
    return mdl, params


def get_output_filename(args):
    if args.filename:
        return args.filename
    return 'predict-' + args.algorithm + '.csv'


def main():
    args = get_args()
    mdl, params = choose_model(args)

    S, col_names_S = load_data(config.paths.training_data,
                               config.paths.feat_types,
                               config.paths.cache_folder)
    Xs, Ys, col_names_S = extract_xy(S, col_names_S)

    if args.ignore_high_d_feats:
        Xs, col_names_S = remove_cols(Xs, col_names_S)

    mdl.tune(Xs, Ys, **params)

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

    out_filename = get_output_filename(args)
    save_submission(pred, config.paths.out_folder + out_filename)


if __name__ == '__main__':
    main()
