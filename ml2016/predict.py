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


def load_test_data(path, header=True):
    """
    Loads the test data from quiz.csv.

    Parameters
    ----------
    path : str
        Filepath to test case data.

    header : bool
        Flag indicating whether the input contains a header line, which will be
        skipped.

    Returns
    -------
    data : dict[int, str]
        Maps the id of each input line to the line.
    """
    data = {}
    id = 1
    with open(path) as fi:
        # TODO use header names (see python csv module)
        if header:
            fi.next()
        for line in fi:
            # TODO apply preprocessing
            data[id] = line
            id += 1
    return data


def _verify_pred_labels(pred):
    for _, lbl in pred.items():
        if lbl not in {-1, 1}:
            raise ValueError("predicted labels must be in {-1, 1}")
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

    folder = os.path.split(path)[0]
    if not os.path.exists(folder):
        os.makedirs(folder)

    header = 'Id,Prediction\n'
    lines = [str(i) + ',' + str(pred[i]) + '\n' for i in sorted(pred)]
    with open(path, 'w') as fo:
        fo.write(header)
        for l in lines:
            fo.write(l)
    logger.info('saved submission file: %s' % path)


def predict_const(path, val):
    """
    Load data and predict a constant value for all cases.

    Parameters
    ----------
    path : str
        Path to training/test cases.

    val : int
        Value to predict for each case in {-1, 1}.

    Returns
    -------
    pred : dict[int, int]
        Map of example ids to prediction labels.

    Raises
    ------
    ValueError
        If val is not in the set {-1, 1}.
    """
    if val not in {-1, 1}:
        raise ValueError("val must be in {-1, 1}")
    data = load_test_data(path)
    pred = {}
    for i, line in data.items():
        pred[i] = val
    return pred
