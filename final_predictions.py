"""
Final prediction script, with interface matching the instructions:

python final_predictions.py DATAFILE QUIZFILE OUTPUTFILE
"""

from argparse import ArgumentParser
import logging
import os
import sys

from config import Config, Paths, cache_folder, feat_types
from ml2016.preprocess import Preprocessor
from ml2016.preprocess import load_data, extract_xy
from ml2016.randomforest import RandomForest
from ml2016.submit import save_submission

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


logger.info(sys.path[0])

mdl = RandomForest()

params = {'n_estimators': 50,
          'max_depth': None,
          'criterion': 'gini',
          'max_features': 0.1,
          'n_jobs': -1,
          'verbose': 100}

scale = True


def get_args():
    p = ArgumentParser(description="Reproduce final prediction")
    p.add_argument("DATAFILE", help="path to data.csv", type=str)
    p.add_argument("QUIZFILE", help="path to quiz.csv", type=str)
    p.add_argument("OUTPUTFILE", help="path to store predictions", type=str)
    args = p.parse_args()
    return args


def main():
    args = get_args()

    logger.info('--- Preprocessing ---')

    path_dict = {'training_data': args.DATAFILE, 'test_data': args.QUIZFILE,
                 'out_folder': os.path.split(args.OUTPUTFILE)[0],
                 'cache_folder': cache_folder, 'feat_types': feat_types}
    paths = Paths(**path_dict)

    PARAMS = {'adaboost': None, 'knn': None, 'logit': None,
              'pca_knn': None, 'pf_adaboost': None, 'svm': None,
              'randomforest': None, 'kmeans': None, 'gmm': None,
              'dpgmm': None}

    config = Config(paths, PARAMS)

    ignore_cols = ['23', '58']
    logger.info('ignoring columns: %s' % str(ignore_cols))

    preproc = Preprocessor(config.paths.training_data, config.paths.test_data,
                           config.paths.cache_folder, config.paths.feat_types,
                           ignore_cols, scale, None)
    (Xs, _), (Xt, _) = preproc.fresh_load_data()

    logger.info('training data shape: %s' % str(Xs.shape))
    logger.info('test data shape: %s' % str(Xt.shape))

    logger.info('--- Tuning & Predicting ---')

    S, col_names_S = load_data(config.paths.training_data,
                               config.paths.cache_folder)
    Xs, Ys, col_names_S = extract_xy(S, col_names_S)

    mdl.tune(Xs, Ys, **params)

    logger.info("Training score: %0.5f" % mdl.score(Xs, Ys))

    T, col_names_T = load_data(config.paths.test_data,
                               config.paths.cache_folder)
    Xt = T

    logger.info('predicting test set labels')
    test_pred = mdl.predict(Xt)
    pred = {}
    for i, lbl in enumerate(test_pred):
        pred[i + 1] = int(lbl)

    out_filename = os.path.split(args.OUTPUTFILE)[1]
    save_submission(pred, os.path.join(config.paths.out_folder, out_filename))



if __name__ == '__main__':
    main()
