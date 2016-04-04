"""
Loading data and preprocessing.
"""


def load_data(path, header=True):
    """
    Loads the training or test data from quiz.csv.

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
