"""
Builds `config` to be imported by training & prediction modules.

Modify default elements in this section as needed. Each sub-section header
identifies the attribute of config where the variables will be stored. Or,
alternatively, import classes at the end to roll your own.
"""

from collections import namedtuple

import numpy as np


# config.params.*
adaboost = {'max_depth': 1,
            'n_estimators': [32, 2048],
            'learning_rate': 0.2}

knn = {'n_neighbors': [1, 3, 5]}

logit = {'C': [0.7, 1, 1.3]}

pca_knn = {'n_components': [50, 100, 200],
           'n_neighbors': [1, 3, 5]}

pf_adaboost = {'degree': 2,
            'max_depth': [1, 3],
            'n_estimators': [1, 2, 4, 8, 16, 32],
            'learning_rate': 1.0}

svm = {'C': list(np.logspace(3, 5, num=3)),
       'kernel': 'rbf',
       'gamma': 'auto'}

randomforest = {'n_estimators': [100, 200, 500],
                'max_depth': [None],
                'criterion': ['gini'],
                'max_features': ['auto', 0.25, 0.5, 0.75]}

kmeans = {'n_clusters': [100]}

gmm = {'n_components': [10],
       'covariance_type': ['full'],
       'tol': [1e-3]}

dpgmm = {'n_components': [10],
         'alpha': [1.0],
         'covariance_type': ['full'],
         'tol': [1e-3]}

# config.paths.*
training_data = 'data/data.csv'
test_data = 'data/quiz.csv'
out_folder = 'out/'
cache_folder = 'cache/'
feat_types = 'data/field_types.txt'

# ----------------------------------------------------------------------------

_PATHS = {'training_data': training_data, 'test_data': test_data,
          'out_folder': out_folder, 'cache_folder': cache_folder,
          'feat_types': feat_types}

Paths = namedtuple('Paths', sorted(_PATHS))
_p = Paths(**_PATHS)

_PARAMS = {'adaboost': adaboost, 'knn': knn, 'logit': logit,
           'pca_knn': pca_knn, 'pf_adaboost': pf_adaboost, 'svm': svm,
           'randomforest': randomforest, 'kmeans': kmeans, 'gmm': gmm, 'dpgmm':dpgmm}

Params = namedtuple('Params', sorted(_PARAMS))
_params = Params(**_PARAMS)

Config = namedtuple('Config', ['paths', 'params'])
config = Config(_p, _params)
