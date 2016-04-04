"""
Builds `config` to be imported by training & prediction modules.

Modify elements in this section as needed. Each sub-section header identifies
the attribute of config where the variables will be stored.
"""


# Config.paths.*
data_folder = 'data/'
out_folder = 'out/'
cache_folder = 'cache/'


# ----------------------------------------------------------------------------

from collections import namedtuple

_PATHS = {'data_folder': data_folder, 'out_folder': out_folder,
          'cache_folder': cache_folder}

_path_names = ['data_folder', 'out_folder', 'cache_folder']
_paths = namedtuple('Paths', _path_names)
_p = _paths(**_PATHS)

_config = namedtuple('Config', 'paths')
config = _config(_p)
