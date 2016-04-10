"""
Utilities and base classes.
"""


class BaseClassifier(object):
    """
    Base classification wrapper.

    This class should not be used directly, but instead should be derived.

    Attributes
    -----
    clf : object
        A classifier object following the scikit-learn interface.
    """

    def __init__(self):
        self.clf = None

    def fit(self):
        """Used to fit a classifier to training data."""
        raise NotImplementedError

    def tune(self):
        """Used to tune hyper-parameters."""
        raise NotImplementedError

    def predict(self, X):
        """Use `self.clf` to predict on the set `X`."""
        return self.clf.predict(X)

    def score(self, X, Y):
        """Use `self.clf` to score predictions on labels `Y`."""
        return self.clf.score(X, Y)

    def __str__(self):
        return "<%s %s>" % (self.__class__.__name__, self.__dict__)
