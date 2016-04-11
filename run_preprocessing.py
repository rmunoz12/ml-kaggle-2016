from argparse import ArgumentParser
import logging

from config import config
from ml2016.preprocess import load_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args():
    p = ArgumentParser(description="Force preprocessing of training and test data.")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--training", help="preprocess training data only",
                       action='store_true')
    group.add_argument("--test", help="preprocess test data only",
                       action='store_true')
    p.add_argument('--ignore-high-d-feats',
                   help='ignores features 23 and 58',
                   action='store_true')
    args = p.parse_args()
    return args


def main():
    args = get_args()
    run_all = True
    if args.training or args.test:
        run_all = False
    ignore_cols = None
    if args.ignore_high_d_feats:
        ignore_cols = ['23', '58']
        logger.info('ignoring columns: %s' % str(ignore_cols))
    if run_all or args.training:
        logger.info('preprocessing training data: data.csv')
        data_s, _ = load_data(config.paths.training_data, config.paths.feat_types,
                           config.paths.cache_folder, use_cache=False,
                           ignore_cols=ignore_cols)
        logger.info('training data shape: %s' % str(data_s.shape))
    if run_all or args.test:
        logger.info('preprocessing test data: quiz.csv')
        data_t, _ = load_data(config.paths.test_data, config.paths.feat_types,
                           config.paths.cache_folder, use_cache=False,
                           ignore_cols=ignore_cols)
        logger.info('test data shape: %s' % str(data_t.shape))


if __name__ == '__main__':
    main()
