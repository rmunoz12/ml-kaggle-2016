"""
Loading data and preprocessing.


Features described in field_types.txt.
"""

from __future__ import print_function
import csv
import logging
import os

import numpy as np
from sklearn.feature_extraction import DictVectorizer

from config import config

logger = logging.getLogger(__name__)


def load_feat_types():
    """
    Returns
    -------
    feature_types : dict[str, list[str]]
        Maps feature names to their raw types from field_types.txt.
    """
    with open(config.paths.data_folder + 'field_types.txt', 'rb') as fi:
        feat_types = {}
        for line in fi:
            k, v = line.split(' ', 1)
            v = v.split()
            if len(v) > 1:
                v = [t[1:] for t in v]
            feat_types[k] = v
    return feat_types


def encode_feat_instance(row, feature_name, feat_types):
    """
    Parameters
    ----------
    row : dict[str, str]


    feature_name : str
        feature identifier, e.g. '58'

    Returns
    -------

    """
    typ = feat_types[feature_name]
    typ_dicts = [{feature_name: t} for t in typ]
    # logger.info('FN: {} typ: {}'.format(feature_name, typ))
    v1 = DictVectorizer()
    x1 = v1.fit_transform(typ_dicts)
    names = v1.get_feature_names()

    val = row[feature_name]
    assert(feature_name + '=' + val in names)

    d = {feature_name: val}
    v2 = DictVectorizer()
    x2 = v2.fit_transform(d)
    new_feat = v2.inverse_transform(x2)[0]
    for n in names:
        if n not in new_feat.keys():
            new_feat[n] = np.float64(0.0)
    return new_feat


def encode_row(row, feat_types):
    new_row = {}
    for f, v in row.items():
        if f == 'label' or feat_types[f][0] == 'numeric':
            new_row[f] = np.float64(v)
        else:
            new_row.update(encode_feat_instance(row, f, feat_types))
    return new_row


def _fresh_load_data(path, feat_types):
    """
    Loads the training or test data, assumed to contain a header, encodes all
    categorical features (using one-hot-encoding), and caches the result.

    Parameters
    ----------
    path : str
        Filepath to training/test case data.

    feat_types : dict[str, list[str]]
        The result of load_feat_types().

    Returns
    -------
    data : dict[int, dict[str, float64]]
        Maps the id of each input line to a dictionary mapping fields to values.
    """
    data = {}
    id = 1
    fi = open(path, 'rb')
    if not os.path.exists(config.paths.cache_folder):
        os.makedirs(config.paths.cache_folder)
    fo = open(config.paths.cache_folder + 'encoded_' + os.path.split(path)[1],
              'wb')
    reader = csv.DictReader(fi)
    first_line = True
    for row in reader:
        if id % 100 == 0:
            logger.info('reading and encoding line: %d' % id)
        encoded_row = encode_row(row, feat_types)
        data[id] = encoded_row
        if first_line:
            header = [k for k in sorted(encoded_row)]
            header = ','.join(header) + '\n'
            fo.write(header)
            first_line = False
        line = [str(encoded_row[k]) for k in sorted(encoded_row)]
        line = ','.join(line) + '\n'
        fo.write(line)
        id += 1
    fi.close()
    fo.close()
    return data


def load_data(path, use_cache=True):
    """
    By default, loads the (encoded) training or test data from cache, otherwise,
    loads and encodes the data, assumed to contain a header, and caches the
    result.

    Parameters
    ----------
    path : str
        Filepath to training/test case data. Expects this value to be the
        original file path even when use_cache is True.

    use_cache : bool
        When true, attempts to load the cached data at a location determined
        based on the original `path`.

    Returns
    -------
    data : dict[int, dict[str, float64]]
        Maps the id of each input line to a dictionary mapping fields to values.
    """
    cache_path = config.paths.cache_folder + \
                 'encoded_' + os.path.split(path)[1]
    if not use_cache or not os.path.exists(cache_path):
        feat_types = load_feat_types()
        logger.info('performing a fresh load of %s' % os.path.split(path)[1])
        return _fresh_load_data(path, feat_types)
    else:
        logger.info('using cached version of %s' % os.path.split(path)[1])
        data = {}
        id = 1
        fi = open(cache_path, 'rb')
        reader = csv.DictReader(fi)
        for row in reader:
            data[id] = row
            id += 1
        fi.close()
        return data
