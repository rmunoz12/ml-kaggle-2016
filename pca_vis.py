from __future__ import division

import logging
from argparse import ArgumentParser
from copy import copy

from matplotlib import pyplot as plt
import numpy as np

from config import config
from ml2016.preprocess import drop_feature, load_data, extract_xy

from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args():
    p = ArgumentParser(description="Plot 3D PCA")
    p.add_argument('--ignore-high-d-feats',
                   help='ignores features 23 and 58',
                   action='store_true')
    args = p.parse_args()
    return args

def main():
    args = get_args()
    S, col_names_S = load_data(config.paths.training_data,
                               config.paths.cache_folder)
    Xs, Ys, col_names_S = extract_xy(S, col_names_S)

    if args.ignore_high_d_feats:
        names = copy(col_names_S)
        for name in names:
            if name[:2] == '23' or name[:2] == '58':
                Xs, col_names_S = drop_feature(Xs, col_names_S, name)

    X = Xs.toarray()
    y = Ys.toarray()
    pca = decomposition.PCA(n_components=3)
    pca.fit(X)
    Xpc = pca.transform(X)

    fig = plt.figure(1, figsize=(8, 8))
    plt.clf()
    ax = Axes3D(fig)
    ax.scatter(Xpc[:, 0], Xpc[:, 1], Xpc[:, 2], c=y, cmap=plt.cm.spectral)
    plt.show()


if __name__ == '__main__':
    main()
