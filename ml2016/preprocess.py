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
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

logger = logging.getLogger(__name__)


class Preprocessor(object):
    """
    Preprocessing class that ensures application of the same transformations
    to both training and test data sets.

    Parameters
    ----------
    train_data_path : str
            Filepath to training data.

    test_data_path : str
            Filepath to test data.

    cache_folder : str
        Path to store encoded data sets.

    feat_types_path : str
        Path to field_types.txt.

    ignore_cols : list[str]
            Column names to ignore.

    scale : bool
        Whether to apply numerical centering & scaling.

    pf : int | None
        Polynomial feature degree.

    Attributes
    ----------
    feat_types : dict[str, list[str]]
            Maps feature names to their raw types from field_types.txt.

    num_mdl : StandardScaler
        Numeric model used to apply same scaling to training and test data. Set
        only after loading of training data.

    factor_mdl : DictVectorizer
        Factor model used to apply same encoding to training and test data.
    """
    def __init__(self, train_data_path, test_data_path, cache_folder,
                 feat_types_path, ignore_cols=list(), scale=False, pf=None):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.cache_folder = cache_folder
        self.ignore_cols = ignore_cols
        self.num_mdl = None
        self.factor_mdl = None
        self.scale = scale
        self.pf = pf

        self.set_feat_types(feat_types_path)
        self.set_factor_mdl()

    def set_feat_types(self, path):
        """
        Load feature type information.

        Parameters
        ----------
        path : str
            Path to field_types.txt
        """
        logger.info('loading feature type information')
        with open(path, 'rb') as fi:
            self.feat_types = {}
            for line in fi:
                k, v = line.split(' ', 1)
                v = v.split()
                if len(v) > 1:
                    v = [t[1:] for t in v]
                self.feat_types[k] = v

    def set_factor_mdl(self):
        """Fit a factor model to all possible factor values"""
        logger.info('setting factor model')
        typ_dicts = []
        for f, typ in self.feat_types.items():
            if len(typ) > 1 and f not in self.ignore_cols:
                typ_dicts.extend([{f: t} for t in typ])
        self.factor_mdl = DictVectorizer()
        self.factor_mdl.fit(typ_dicts)

    def set_num_mdl(self, numeric_data):
        """Scale and center numerical data columns"""
        logger.info('setting numerical model')
        self.num_mdl = StandardScaler(copy=False)
        self.num_mdl.fit(numeric_data)

    def scale_dataset(self, numeric_data):
        """
        Scale and center numerical features

        Also sets 'self.num_mdl`.

        Parameters
        ----------
        numeric_data : pd.DataFrame
            raw numeric data read from input file with pd.read_csv

        Returns
        -------
        new_data : csr_matrix
            Data with processed numeric variables.

        col_names : pd.Index
            Column labels
        """
        logger.info('scaling numerical features')
        col_names = numeric_data.columns
        if self.scale:
            # only X59 and X60 are actually numeric features
            # the remainder are actually factors {0, 1} or {0, 1, 2}
            logger.info('numeric_data.shape: (%d, %d)' % numeric_data.shape)
            Z = numeric_data.ix[:, ('59', '60')]
            logger.info('Z.shape: (%d, %d)' % Z.shape)
            Z = self.num_mdl.transform(Z)
            logger.info('Z.shape: (%d, %d)' % Z.shape)
            Z = pd.DataFrame(Z)
            Z.index = np.arange(1, len(Z) + 1)
            logger.info('Z.shape: (%d, %d)' % Z.shape)
            numeric_data = numeric_data.drop(['59', '60'], 1)
            logger.info('numeric_data.shape: (%d, %d)' % numeric_data.shape)
            new_data = pd.concat([numeric_data, Z], axis=1, ignore_index=True)
            logger.info('new_data.shape: (%d, %d)' % new_data.shape)
            col_names = new_data.columns
        else:
            new_data = numeric_data
        return new_data, col_names

    def encode_dataset(self, factor_data):
        """
        Parameters
        ----------
        factor_data : pd.DataFrame
            raw factor data read from input file with pd.read_csv

        Returns
        -------
        new_data : csr_matrix
            Data with categorical variables encoded using one-hot-encoding.

        col_names : dict[str, int]
            Maps names of columns in new_data to column indexes. This dictionary
            is invertible.
        """
        logger.info('encoding factor features')
        new_data = self.factor_mdl.transform(factor_data.to_dict('records'))
        col_names = {k: val for k, val in self.factor_mdl.vocabulary_.items()}
        return new_data, col_names

    def _load_data(self, path):
        """
        Load and split data set

        Parameters
        ----------
        path : str
            Dataset filepath.

        Returns
        -------
        numeric_data, factor_data, label_data : pd.DataFrame | None
            Dataset split out into column type. `label_data` is returned as
            None if there is no column named 'label'.
        """
        with open(path, 'rb') as fi:
            data = pd.read_csv(fi)
        if self.ignore_cols:
            for name in self.ignore_cols:
                del data[name]
        data.index = range(1, data.index[-1] + 2)  # ID name index from 1 not 0

        logger.info('splitting numeric and factor features')

        labelled = False
        numeric_col_names, factor_col_names = [], []
        for column in data:
            if column == 'label':
                labelled = True
            elif self.feat_types[column][0] == 'numeric':
                numeric_col_names.append(column)
            else:
                factor_col_names.append(column)

        numeric_data = data[numeric_col_names]
        factor_data = data[factor_col_names]
        label_data = None
        if labelled:
            label_data = data['label'].to_frame()

        return numeric_data, factor_data, label_data

    def _process_data(self, numeric_data, factor_data, label_data=None):
        """
        Apply preprocessing.

        Parameters
        ----------
        numeric_data, factor_data, label_data : pd.DataFrame | None
            Dataframes for each feature type. `labe_data` can be None if there
            was no corresponding column.

        Returns
        -------
        data : csr_matrix
            Data matrix where numerical, factor, and (if present) label columns
            have been horizontally stacked.

        col_names : dict[str, int]
            Maps names of columns in `data` to column indexes.
        """
        numeric_data, numeric_col_names = self.scale_dataset(numeric_data)
        factor_data, factor_col_names = self.encode_dataset(factor_data)

        logger.info('combining numeric and encoded factor features')
        data = hstack((numeric_data, factor_data), format='csr')
        col_names = \
            {numeric_col_names[i]: i for i in range(len(numeric_col_names))}
        factor_col_idx = \
            {k: val + len(col_names) for k, val in factor_col_names.items()}
        col_names.update(factor_col_idx)

        if self.pf:
            logger.info('Converting data to array...')
            data = data.toarray()
            logger.info('Adding polynomial features of degree: %d' % self.pf)
            mdl = PolynomialFeatures(self.pf)
            data = mdl.fit_transform(data)

        if label_data is not None:
            data = hstack((data, label_data), format='csr')
            col_names.update({'label': len(col_names)})

        return data, col_names

    def _cache_data(self, orig_path, data, col_names):
        """
        Cache data to disk.

        Parameters
        ----------
        orig_path : str
            Filepath to input data.

        data : csr_matrix
            Processed data.

        col_names : dict[str, int]
            Corresponding column names for `data`.
        """
        orig_filename = os.path.split(orig_path)[1]
        orig_filename = orig_filename.rsplit('.', 1)[0]
        matrix_cache_path = self.cache_folder + 'encoded_' + \
                            orig_filename + '.mtx'
        names_cache_path = self.cache_folder + 'encoded_col_names_' + \
                           orig_filename + '.json'
        logger.info('caching encoded data matrix to: %s' % matrix_cache_path)
        if not os.path.exists(self.cache_folder):
            os.makedirs(self.cache_folder)
        with open(matrix_cache_path, 'wb') as fo:
            mmwrite(fo, data)
        logger.info('caching encoded data matrix column names to: %s' %
                    names_cache_path)
        with open(names_cache_path, 'wb') as fo:
            json.dump(col_names, fo)

    def fresh_load_data(self):
        """
        Loads the training and test data, assumed to contain headers, applies
        preprocessing and caches the result.

        Returns
        -------
        tuple[csr_matrix, dict[str, int]]
            A two-value tuple containing training and then test data, along with
            associated column names. Thus, each value in this tuple is a tuple:

                Xs, Xt : csr_matrix
                    Sparse data matrix, with id increasing with row number.

                col_names_s, col_names_t : dict[str, int]
                    Maps names of columns in Xs / Xt to column indexes.
                    These dictionaries are invertible.
        """
        nd, fd, ld = self._load_data(self.train_data_path)

        # only X59 and X60 are actually numeric features
        # the remainder are actually factors {0, 1} or {0, 1, 2}
        Z = nd.ix[:, ('59', '60')]
        self.set_num_mdl(Z)

        Xs, col_names_s = self._process_data(nd, fd, ld)
        self._cache_data(self.train_data_path, Xs, col_names_s)

        nd, fd, _ = self._load_data(self.test_data_path)
        Xt, col_names_t = self._process_data(nd, fd)
        self._cache_data(self.test_data_path, Xt, col_names_t)
        return (Xs, col_names_s), (Xt, col_names_t)


def load_data(data_path, cache_folder):
    """
    Loads the (encoded) training or test data from cache.

    Parameters
    ----------
    data_path : str
        Filepath to training/test case data. Expects this value to be the
        original file path.

    cache_folder : str
        Path to store/load encoded data sets.

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
    if not os.path.exists(data_cache_path):
        logger.error("Cache file does not exist, "
                     "use `run_preprocessing.py` first")
        raise ValueError(data_cache_path)
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
    if f_idx != data.shape[1] - 1:
        X = hstack((data[:, :f_idx], data[:, (f_idx + 1):]), format='csr')
    else:
        X = data[:, :f_idx]

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
