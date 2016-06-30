# -*- coding: utf8

"""
The :mod:`ensemble` module implements ensemble of classifiers
"""


import numpy as np
from scipy.linalg import pinv2

from elm import GenELMClassifier
from random_layer import RBFRandomLayer, MLPRandomLayer
from sklearn.cross_validation import KFold
import copy as copy

class Ensemble:
    'Ensemble of learners'

    def __init__(self, learners, nfolds=1):
        self.learners = learners
        self.nfolds = nfolds
        self.nclasses = None
        
    def fit(self, X, y):
        self.nclasses = np.max(y) + 1
        
        if self.nfolds == 1:
            for learner in self.learners:
                learner.fit(X, y)
        else:
            foldlearners = []
            for learner in self.learners:
                for train_indices, test_indices in KFold(X.shape[0], \
                                                     self.nfolds, shuffle=True):
                    # print "train_indices:"
                    # print train_indices
                    # print "ytrain:"
                    # print y[train_indices]
                    
                    foldlearner = copy.deepcopy(learner)
                    foldlearner.fit(X[train_indices], y[train_indices])
                    foldlearners.append(foldlearner)
                    self.learners = foldlearners
                    
        return self
    
    def _compute_predictions(self, X):
        predictions = []
        for learner in self.learners:
            predictions.append(learner.predict(X))
        return predictions
    
    def predict_proba_labels(self, X):
        'Predicts class probabilities as frequency of class labels predicted from learners'        
        predictions = self._compute_predictions(X)
        n,p = X.shape
        freqs = self._compute_freq(predictions, n)
        probabilities = self._freq_to_prob(freqs)
        return probabilities

    def _compute_freq(self, predictions, nsamples):
        print "nsamples = %d, nclasses = %d" % (nsamples,  self.nclasses)
        histogram = np.zeros([nsamples, self.nclasses])
        for row in range(0, nsamples):
            for prediction in predictions:
                predicted_class = prediction[row]
                histogram[row, predicted_class] = histogram[row, predicted_class] + 1
        return histogram

    def _freq_to_prob(self, freqs):
        probs = np.zeros(freqs.shape)
        for row in xrange(freqs.shape[0]):
            probs[row] = freqs[row] / np.sum(freqs[row])
        return probs

    def predict_proba(self, X):
        'Predict probability as average probabilities of leaners?'
        probability_list = [learner.predict_proba(X) for learner in self.learners]
        #print probability_list
        probabilties = np.sum(probability_list, axis=0) / len(self.learners)
        return probabilties
        
            

class ELMEnsemble(Ensemble):

    def __init__(self, n_learners, n_hidden_layer_nodes, activation_func, nfolds=1):

        self.nfolds = nfolds
        
        self.learners = []
        for idx in range(0, n_learners):
            hidden_layer = MLPRandomLayer(n_hidden = n_hidden_layer_nodes,\
                                          activation_func = activation_func)
            self.learners.append(GenELMClassifier(hidden_layer = hidden_layer))
        
