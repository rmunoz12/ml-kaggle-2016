"""
Naive prediction functions, largely for testing purposes.
"""

from random import random


def predict_on_avg(data, avg_prob):
    """
    Load data and predict a constant value for all cases.

    Parameters
    ----------
    data : csr_matrix
        Sparse data matrix, with id increasing with row number.

    avg_prob : float
        Probability [0, 1] to predict a positive label.

    Returns
    -------
    pred : dict[int, int]
        Map of example ids to prediction labels.

    Raises
    ------
    ValueError
        If `avg_prob` is not in [0, 1]
    """
    if avg_prob < 0 or avg_prob > 1:
        raise ValueError("avg_prob must be in [0, 1]")
    pred = {}
    for i in range(data.shape[0]):
        pred[i + 1] = 1 if random() < avg_prob else -1
    return pred


def predict_const(data, val):
    """
    Predict a constant value for all cases.

    Parameters
    ----------
    data : csr_matrix
        Sparse data matrix, with id increasing with row number.

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
    pred = {}
    for i in range(data.shape[0]):
        pred[i + 1] = val
    return pred
