"""
Bayes based classification models
"""

import sklearn.base
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
        self.classes_ = labels

        # Conditional probability distributions
        self.cpdf = []
        for label in labels:
            kde = sklearn.base.clone(self.kde)
            kde.fit( X[y == label, :] )
            self.cpdf.append( kde )

        # Prior distributions
        self.prior = np.log(priorDistribution(y))

        
    def predict(self, X):
        return self.predictk(X, 1).reshape(-1, 1)

    
    def predict_proba(self, X):
        out = np.zeros(X.shape[0], len(self.classes_))
        for row in out:
            row = self.posteriors(row)
        return out

    
    def predictk(self, X, k):
        out = np.zeros((X.shape[0], k))
        for row in xrange(0, X.shape[0]):
            bestIndices = common.largestK(self.posteriors(X[row, :]))
            out[row, :] = self.classes_[bestIndices]
        return out

    
    def posteriors(self, x):
        values = np.zeros(len(x))
        for k in xrange(0, len(self.classes_)):
            values[k] = self.cpdf[k].score(x.reshape(1, -1)) + self.prior[k]
        return values
