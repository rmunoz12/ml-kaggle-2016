"""
Create a submission file for `quiz.csv`.

Submission files should be CSV files containing two columns: Id and Prediction.

Id: an integer i between 1 and 31709, corresponding to the data point from the
i-th row of the quiz set file.

Prediction: a {-1,+1}-value prediction for the corresponding data point.
The file should contain a header and have the following format:

Id,Prediction
1,1
2,-1
3,1
4,-1
etc.
"""

from __future__ import print_function

import logging
import os

logger = logging.getLogger(__name__)


def _verify_pred_labels(pred):
    for _, lbl in pred.items():
        if lbl not in {-1, 1}:
            raise ValueError("predicted labels must be in {-1, 1}")
    return True


def _verify_pred_ids(pred):
    j = 1
    for i in sorted(pred):
        if i != j:
            raise ValueError("Unexpected prediction id: {}; Expected: {}"
                             .format(i, j))
        j += 1
    return True


def save_submission(pred, path):
    """
    Save a submission CSV file in proper format, sorted by id.

    Parameters
    ----------
    pred : dict[int, int]
        Maps test case ids to predicted labels.

    path : str
        Filepath for saving the submission file.

    Raises
    ------
    ValueError
        If any predicted label is not in {-1, 1}.
    """
    if not _verify_pred_labels(pred):
        raise ValueError
    if not _verify_pred_ids(pred):
        raise ValueError

    folder = os.path.split(path)[0]
    if not os.path.exists(folder):
        os.makedirs(folder)

    header = 'Id,Prediction\n'
    lines = [str(i) + ',' + str(pred[i]) + '\n' for i in sorted(pred)]
    with open(path, 'wb') as fo:
        fo.write(header)
        for l in lines:
            fo.write(l)
    logger.info('saved submission file: %s' % path)


