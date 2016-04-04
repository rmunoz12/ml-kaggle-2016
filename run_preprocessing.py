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
    args = p.parse_args()
    return args


def main():
    args = get_args()
    run_all = True
    if args.training or args.test:
        run_all = False
    if run_all or args.training:
        logger.info('preprocessing training data: data.csv')
        data_s = load_data(config.paths.data_folder + 'data.csv',
                           use_cache=False)
    if run_all or args.test:
        logger.info('preprocessing test data: quiz.csv')
        data_t = load_data(config.paths.data_folder + 'quiz.csv',
                           use_cache=False)


if __name__ == '__main__':
    main()
