"""
Bayes based classification models
"""

import numpy as np
from sklearn.neighbors.kde import KernelDensity

def priorDistribution(y):
    labels = np.unique(y)
    n = len(labels)
    d = float(len(y))
    priors = np.zeros(n, dtype=float)
    for idx in xrange(0, n):
        priors[idx] = np.sum(y == labels[idx]) / d
    return priors

class BayesClassifierKDE():
    """
    Performs Classification using the Bayes rule:
    
    p(y | X) = p( X | y) p (y) / p(X)

    Assumptions:
    """
    
    def __init__(self, kde = None):
        if kde:
            self.kde = kde
        else:
            self.kde = KernelDensity()
        return

    def fit(self, X, y):

        labels = np.unique(y)

        # Conditional probability distributions
        self.cpdf = []
        for label in labels:
            self.cpdf.append(self.kde.fit( (X[y == label, :])) )

        # Prior distributions
        self.prior = np.log(priorDistribution(y))

    def predict(X):
        return

    def predict_proba(X):
        return

    def predictk(X):
        return

