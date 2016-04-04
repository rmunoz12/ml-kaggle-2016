"""
Loading data and preprocessing.


Features described in field_types.txt.
"""

from __future__ import print_function
import csv
import logging

import numpy as np
from sklearn.feature_extraction import DictVectorizer

from config import config

logger = logging.getLogger(__name__)


"""
feature_types : dict[str, list[str]]
    Maps feature names to their raw types
"""
with open(config.paths.data_folder + 'field_types.txt', 'rb') as fi:
    feature_types = {}
    for line in fi:
        k, v = line.split(' ', 1)
        v = v.split()
        if len(v) > 1:
            v = [t[1:] for t in v]
        feature_types[k] = v


def encode_vocabulary(row, feature_name):
    """
    Parameters
    ----------
    row : dict[str, str]


    feature_name : str
        feature identifier, e.g. '58'

    Returns
    -------

    """
    typ = feature_types[feature_name]
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


def encode_dataset(data):
    new_data = {}
    for i, row in data.items():
        new_row = {}
        for f, v in row.items():
            if feature_types[f][0] == 'numeric':
                new_row[f] = np.float64(v)
            else:
                new_row.update(encode_vocabulary(row, f))
        new_data[i] = new_row
    return new_data


def load_data(path):
    """
    Loads the training or test data, assumed to contain a header.

    Parameters
    ----------
    path : str
        Filepath to training/test case data.

    Returns
    -------
    data : dict[int, dict[str, str]]
        Maps the id of each input line to a dictionary mapping fields to values.
    """
    data = {}
    id = 1
    with open(path, 'rb') as fi:
        reader = csv.DictReader(fi)
        for row in reader:
            data[id] = row
            id += 1
    data = encode_dataset(data)
    return data
