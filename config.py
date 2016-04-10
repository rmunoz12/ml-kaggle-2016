"""
Builds `config` to be imported by training & prediction modules.

Modify default elements in this section as needed. Each sub-section header
identifies the attribute of config where the variables will be stored. Or,
alternatively, import classes at the end to roll your own.
"""


# config.params.*
adaboost = {'max_depth': [1, 3],
            'n_estimators': [1, 2, 4, 8, 16, 32],
            'learning_rate': 1.0}

knn = {'n_neighbors': [1, 3, 5]}

logit = {'C': [0.7, 1, 1.3]}

# config.paths.*
training_data = 'data/data.csv'
test_data = 'data/quiz.csv'
out_folder = 'out/'
cache_folder = 'cache/'
feat_types = 'data/field_types.txt'

# ----------------------------------------------------------------------------

from collections import namedtuple

_PATHS = {'training_data': training_data, 'test_data': test_data,
          'out_folder': out_folder, 'cache_folder': cache_folder,
          'feat_types': feat_types}

Paths = namedtuple('Paths', sorted(_PATHS))
_p = Paths(**_PATHS)

_PARAMS = {'adaboost': adaboost, 'knn': knn, 'logit': logit}

Params = namedtuple('Params', sorted(_PARAMS))
_params = Params(**_PARAMS)

Config = namedtuple('Config', ['paths', 'params'])
config = Config(_p, _params)
