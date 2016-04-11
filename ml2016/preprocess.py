"""
Loading data and preprocessing.


Features described in field_types.txt.
"""

from __future__ import print_function
import json
import logging
import os

import numpy as np
import pandas as pd
from scipy.io import mmread, mmwrite
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class Preprocessor(object):
    """
    Preprocessing class that ensures application of the same transformations
    to both training and test data sets.

    Parameters
    ----------
    data_path : str
            Filepath to training/test case data.

    cache_folder : str
        Path to store encoded data sets.

    feat_types_path : str
        Path to field_types.txt.

    ignore_cols : None | list[str]
            Column names to ignore.

    Attributes
    ----------
    feat_types : dict[str, list[str]]
            Maps feature names to their raw types from field_types.txt.
    """
    def __init__(self, data_path, cache_folder, feat_types_path,
                 ignore_cols=None):
        self.data_path = data_path
        self.cache_folder = cache_folder
        self.feat_types = self.get_feat_types(feat_types_path)
        self.ignore_cols = ignore_cols

    @staticmethod
    def get_feat_types(path):
        """
        Load feature type information.

        Parameters
        ----------
        path : str
            Path to field_types.txt

        Returns
        -------
        feature_types : dict[str, list[str]]
            Maps feature names to their raw types from field_types.txt.
        """
        with open(path, 'rb') as fi:
            feat_types = {}
            for line in fi:
                k, v = line.split(' ', 1)
                v = v.split()
                if len(v) > 1:
                    v = [t[1:] for t in v]
                feat_types[k] = v
        return feat_types

    def encode_dataset(self, data):
        """
        Parameters
        ----------
        data : pd.DataFrame
            raw data read from input file with pd.read_csv

        Returns
        -------
        new_data : csr_matrix
            Data with categorical variables encoded using one-hot-encoding.

        col_names : dict[str, int]
            Maps names of columns in new_data to column indexes. This dictionary
            is invertible.
        """
        logger.info('splitting numeric and factor features')

        numeric_col_names, factor_col_names = [], []
        for column in data:
            if column == 'label' or self.feat_types[column][0] == 'numeric':
                numeric_col_names.append(column)
            else:
                factor_col_names.append(column)

        numeric_data = data[numeric_col_names]
        factor_data = data[factor_col_names]

        logger.info('encoding factor features')

        typ_dicts = []
        for f, typ in self.feat_types.items():
            if len(typ) > 1 and f not in self.ignore_cols:
                typ_dicts.extend([{f: t} for t in typ])
        dv = DictVectorizer()
        dv.fit(typ_dicts)
        x = dv.transform(factor_data.to_dict('records'))

        logger.info('combining numeric and encoded factor features')

        new_data = csr_matrix(numeric_data.as_matrix())
        new_data = hstack((new_data, x), format='csr')

        col_names = {numeric_col_names[i]: i for i in range(len(numeric_col_names))}
        factor_col_idx = \
            {k: val + len(col_names) for k, val in dv.vocabulary_.items()}
        col_names.update(factor_col_idx)

        return new_data, col_names

    def fresh_load_data(self):
        """
        Loads the training or test data, assumed to contain a header, encodes
        all categorical features (using one-hot-encoding), and caches the
        result.

        Returns
        -------
        data : csr_matrix
            Sparse data matrix, with id increasing with row number.

        col_names : dict[str, int]
            Maps names of columns in new_data to column indexes. This dictionary
            is invertible.
        """
        with open(self.data_path, 'rb') as fi:
            data = pd.read_csv(fi)
        if self.ignore_cols:
            for name in self.ignore_cols:
                del data[name]
        data.index = range(1, data.index[-1] + 2)  # ID name index from 1 not 0

        data, col_names = self.encode_dataset(data)

        orig_filename = os.path.split(self.data_path)[1]
        orig_filename = orig_filename.rsplit('.', 1)[0]
        matrix_cache_path = self.cache_folder + 'encoded_' + \
                            orig_filename + '.mtx'
        names_cache_path = self.cache_folder + 'encoded_col_names_' + \
                           orig_filename + '.json'
        logger.info('caching encoded data matrix to: %s' % matrix_cache_path)
        mmwrite(matrix_cache_path, data)
        logger.info('caching encoded data matrix column names to: %s' %
                    names_cache_path)
        with open(names_cache_path, 'wb') as fo:
            json.dump(col_names, fo)

        return data, col_names


def load_data(data_path, feat_types_path, cache_folder, use_cache=True,
              ignore_cols=None):
    """
    By default, loads the (encoded) training or test data from cache, otherwise,
    loads and encodes the data, assumed to contain a header, and caches the
    result.

    Parameters
    ----------
    data_path : str
        Filepath to training/test case data. Expects this value to be the
        original file path even when use_cache is True.

    feat_types_path : str
        Path to field_types.txt.

    cache_folder : str
        Path to store/load encoded data sets.

    use_cache : bool
        When true, attempts to load the cached data at a location determined
        based on the original `path`.

    ignore_cols : None | list[str]
        Column names to ignore. This only has an effect when the data is
        loaded from the original files and not from cache.

    Returns
    -------
    data : csr_matrix
        Sparse data matrix, with id increasing with row number.

    col_names : dict[str, int]
        Maps names of columns in new_data to column indexes. This dictionary
        is invertible.
    """
    orig_filename = os.path.split(data_path)[1]
    orig_filename = orig_filename.rsplit('.', 1)[0]
    data_cache_path = cache_folder + \
                      'encoded_' + orig_filename + '.mtx'
    names_cache_path = cache_folder + \
                        'encoded_col_names_' + orig_filename + '.json'
    if not use_cache or not os.path.exists(data_cache_path):
        preproc = Preprocessor(data_path, cache_folder, feat_types_path,
                               ignore_cols)
        logger.info('performing a fresh load of %s'
                    % os.path.split(data_path)[1])
        return preproc.fresh_load_data()
    else:
        logger.info('using cached version of %s' % os.path.split(data_path)[1])
        data = mmread(data_cache_path)
        data = data.tocsr()
        with open(names_cache_path, 'rb') as fi:
            col_names = json.load(fi)
        return data, col_names


def drop_feature(data, col_names, feature_name):
    """
    Parameters
    ----------
    data : csr_matrix
        Sparse data matrix with feature to be dropped.

    col_names : dict[str, int]
        Map of column names to column indices in `data`.

    feature_name : string
        Column to be removed.

    Returns
    -------
    X : csr_matrix
        `data` without the dropped feature.

    col_names_X : dict[str, int]
        Map of column names to column indices in `X`.
    """
    f_idx = col_names[feature_name]
    X = hstack((data[:, :f_idx], data[:, (f_idx + 1):]), format='csr')

    col_names_X = {}
    for k, v in col_names.items():
        if v > f_idx:
            col_names_X[k] = v - 1
        else:
            col_names_X[k] = v
    return X, col_names_X


def extract_xy(data, col_names, label_key="label"):
    """
    Given a labeled dataset `data`, where the labels are in value mapped to the
    name `label_key` in `col_names`, split the data matrix into a feature
    matrix `X` and label vector `Y`.

    Parameters
    ----------
    data : csr_matrix
        Sparse data matrix, with features and labels.

    col_names : dict[str, int]
        Map of column names to column indices in `data`.

    label_key : str
        The column name in `col_names` that indentifies the labels.

    Returns
    -------
    X : csr_matrix
        n x (num_feats) matrix of features

    Y : csr_matrix
        n x 1 vector of labels

    col_names_X : dict[str, int]
        Map of column names to column indices in `X`.
    """
    lbl_idx = col_names[label_key]
    Y = data[:, lbl_idx]

    X, col_names_X = drop_feature(data, col_names, label_key)
    return X, Y, col_names_X
