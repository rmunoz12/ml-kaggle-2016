"""
Loading data and preprocessing.


Features described in field_types.txt.
"""

from __future__ import print_function
import csv
import logging

logger = logging.getLogger(__name__)


def load_data(path):
    """
    Loads the training or test data, assummed to contain a header.

    Parameters
    ----------
    path : str
        Filepath to training/test case data.

    header : bool
        Flag indicating whether the input contains a header line, which will be
        skipped.

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
            # TODO apply preprocessing
            data[id] = row
            id += 1
    return data
