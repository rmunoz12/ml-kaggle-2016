"""
Predict a constant value for all test cases.
"""

from argparse import ArgumentParser
import logging

from config import config
from ml2016.preprocess import load_data
from ml2016.submit import save_submission
from ml2016.naive import predict_const

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args():
    p = ArgumentParser(description="Predict a constant value for all test cases.")
    p.add_argument("-v", "--value",
                   help="prediction value (default: %(default)d)",
                   type=int, default=-1)
    p.add_argument("-f", "--filename",
                   help="output filename (default: %(default)s)",
                   type=str, default='predict-minus-one.csv')
    args = p.parse_args()
    return args


def main():
    args = get_args()
    T, col_names = load_data(config.paths.test_data, config.paths.cache_folder)
    logger.info('predicting a constant value: %d' % args.value)
    pred = predict_const(T, args.value)
    save_submission(pred, config.paths.out_folder + args.filename)


if __name__ == '__main__':
    main()
