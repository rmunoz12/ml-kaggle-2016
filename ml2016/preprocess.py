"""
Loading data and preprocessing.


Features described in field_types.txt.
"""

from __future__ import print_function
import csv
import json
import logging
import os

import numpy as np
from scipy.io import mmread, mmwrite
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction import DictVectorizer

logger = logging.getLogger(__name__)


def load_feat_types(path):
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


def encode_dataset(data, feat_types):
    """
    Parameters
    ----------
    data : dict[str, dict[str, str]]
        raw data read from input file with csv.DictReader

    feat_types : dict[str, list[str]]
        The result of load_feat_types().

    Returns
    -------
    new_data : csr_matrix
        Data with categorical variables encoded using one-hot-encoding.

    col_names : dict[str, int]
        Maps names of columns in new_data to column indexes. This dictionary
        is invertible.
    """
    logger.info('splitting numeric and factor features')
    numeric_data, factor_data = {}, {}
    for id, row in data.items():
        numerics, factors = {}, {}
        for f, v in row.items():
            if f == 'label' or feat_types[f][0] == 'numeric':
                numerics[f] = np.float64(v)
            else:
                factors[f] = v
        numeric_data[id], factor_data[id] = numerics, factors

    logger.info('encoding factor features')
    factor_data = [factor_data[k] for k in sorted(factor_data)]

    typ_dicts = []
    for f, typ in feat_types.items():
        if len(typ) > 1:
            typ_dicts.extend([{f: t} for t in typ])
    dv = DictVectorizer()
    dv.fit(typ_dicts)
    x = dv.transform(factor_data)

    logger.info('combining numeric and encoded factor features')
    new_data = []
    num_col_names = sorted(numeric_data[1].keys())
    num_col_idx = {}
    for i in range(len(num_col_names)):
        num_col_idx[num_col_names[i]] = i

    for k in sorted(numeric_data):
        row = numeric_data[k]
        new_row = []
        for f in sorted(row):
            v = row[f]
            new_row.append(v)
        new_data.append(new_row)
    new_data = csr_matrix(new_data)
    new_data = hstack((new_data, x), format='csr')

    factor_col_idx = \
        {k: val + len(num_col_idx) for k, val in dv.vocabulary_.items()}

    col_names = num_col_idx
    col_names.update(factor_col_idx)
    return new_data, col_names


def _fresh_load_data(data_path, cache_folder, feat_types):
    """
    Loads the training or test data, assumed to contain a header, encodes all
    categorical features (using one-hot-encoding), and caches the result.

    Parameters
    ----------
    data_path : str
        Filepath to training/test case data.

    cache_folder : str
        Path to store encoded data sets.

    feat_types : dict[str, list[str]]
        The result of load_feat_types().

    Returns
    -------
    data : csr_matrix
        Sparse data matrix, with id increasing with row number.

    col_names : dict[str, int]
        Maps names of columns in new_data to column indexes. This dictionary
        is invertible.
    """
    data = {}
    id = 1
    fi = open(data_path, 'rb')
    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)
    reader = csv.DictReader(fi)
    for row in reader:
        data[id] = row
        id += 1
    fi.close()

    data, col_names = encode_dataset(data, feat_types)

    orig_filename = os.path.split(data_path)[1]
    orig_filename = orig_filename.rsplit('.', 1)[0]
    matrix_cache_path = cache_folder + 'encoded_' + \
                        orig_filename + '.mtx'
    names_cache_path = cache_folder + 'encoded_labels_' + \
                       orig_filename + '.json'
    logger.info('caching encoded data matrix to: %s' % matrix_cache_path)
    mmwrite(matrix_cache_path, data)
    logger.info('caching encoded data matrix column names to: %s' %
                names_cache_path)
    with open(names_cache_path, 'wb') as fo:
        json.dump(col_names, fo)

    return data, col_names


def load_data(data_path, feat_types_path, cache_folder, use_cache=True):
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
                        'encoded_labels_' + orig_filename + '.json'
    if not use_cache or not os.path.exists(data_cache_path):
        feat_types = load_feat_types(feat_types_path)
        logger.info('performing a fresh load of %s'
                    % os.path.split(data_path)[1])
        return _fresh_load_data(data_path, cache_folder, feat_types)
    else:
        logger.info('using cached version of %s' % os.path.split(data_path)[1])
        data = mmread(data_cache_path)
        data = data.tocsr()
        with open(names_cache_path, 'rb') as fi:
            col_names = json.load(fi)
        return data, col_names
