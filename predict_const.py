"""
Predict a constant value for all test cases.
"""

from argparse import ArgumentParser
import logging

from config import config
from ml2016.predict import predict_const, save_submission


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


if __name__ == '__main__':
    args = get_args()
    logger.info('predicting a constant value: %d' % args.value)
    pred = predict_const(config.paths.data_folder + 'quiz.csv', args.value)
    save_submission(pred, config.paths.out_folder + args.filename)