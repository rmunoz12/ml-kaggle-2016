from argparse import ArgumentParser
import logging

from config import config
from ml2016.preprocess import Preprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args():
    p = ArgumentParser(description="Force preprocessing of training and test data.")
    p.add_argument('--ignore-high-d-feats',
                   help='ignores features 23 and 58',
                   action='store_true')
    p.add_argument('--scale',
                   help='applies scaling to numerical features',
                   action='store_true')
    p.add_argument('--pf',
                   help='add degree of polynomial features (default: none)',
                   type=int, default=None)
    args = p.parse_args()
    return args


def main():
    args = get_args()
    ignore_cols = list()
    if args.ignore_high_d_feats:
        ignore_cols = ['23', '58']
        logger.info('ignoring columns: %s' % str(ignore_cols))
    if not args.scale:
        args.scale = False
    preproc = Preprocessor(config.paths.training_data, config.paths.test_data,
                           config.paths.cache_folder, config.paths.feat_types,
                           ignore_cols, args.scale, args.pf)
    (Xs, _), (Xt, _) = preproc.fresh_load_data()

    logger.info('training data shape: %s' % str(Xs.shape))
    logger.info('test data shape: %s' % str(Xt.shape))


if __name__ == '__main__':
    main()
