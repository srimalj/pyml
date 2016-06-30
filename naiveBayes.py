"""
 Naive Bayes for independent features

 Uses histograms to estimate the probability density function since we 
 only work with 1D features
"""

import numpy as np

class NaiveBayesClassifier():
    """
    Performs Naive Bayes Classification assuming features are in-dependant
    """

    def __init__(self, bins):
        self.bins = bins
        return
    
    def fit(self, X, y):

        classes = np.unique(y)
        
        # Probability distributions
        pdf = {}
        for feature in range(0, X.shape[1]):
            pdf["x%d" % feature] = PDF(X[:, feature])

        # Conditional probability distributions
        cpdf = {}
        for feature in range(0, X.shape[1]):
            for class in range(0. len(classes)):
                cpdf["x%d|c%d" % (feature, class)] = PDF(X[y == classes(class), feature])            
        return

    def predict(X):
        return

    def predict_proba(X):
        return
    
# numpy.histogram retuerns histogram
#http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.histogram.html
