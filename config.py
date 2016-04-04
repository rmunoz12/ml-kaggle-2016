"""
Builds `config` to be imported by training & prediction modules.

Modify default elements in this section as needed. Each sub-section header
identifies the attribute of config where the variables will be stored. Or,
alternatively, import the `Paths` and `Config` classes to roll your own.
"""


# Config.paths.*
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

_path_names = ['training_data', 'test_data', 'out_folder', 'cache_folder',
               'feat_types']
Paths = namedtuple('Paths', _path_names)
_p = Paths(**_PATHS)

Config = namedtuple('Config', 'paths')
config = Config(_p)
