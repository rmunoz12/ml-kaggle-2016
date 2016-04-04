"""
Final prediction script, with interface matching the instructions:

python final_predictions.py DATAFILE QUIZFILE OUTPUTFILE

Note that cached versions of (encoded) DATAFILE and QUIZFILE are used by
default.
"""

from argparse import ArgumentParser
import logging
import os

from config import Config, Paths, cache_folder, feat_types
from ml2016.preprocess import load_data
from ml2016.predict import save_submission


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args():
    p = ArgumentParser(description="Reproduce final prediction")
    p.add_argument("DATAFILE", help="path to data.csv", type=str)
    p.add_argument("QUIZFILE", help="path to quiz.csv", type=str)
    p.add_argument("OUTPUTFILE", help="path to store predictions", type=str)
    args = p.parse_args()
    return args


def main():
    args = get_args()
    path_dict = {'training_data': args.DATAFILE, 'test_data': args.QUIZFILE,
                 'out_folder': os.path.split(args.OUTPUTFILE)[0],
                 'cache_folder': cache_folder, 'feat_types': feat_types}
    paths = Paths(**path_dict)
    config = Config(paths)

    raise NotImplementedError


if __name__ == '__main__':
    main()
