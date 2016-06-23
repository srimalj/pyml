"""
Trains individual estimators for each grid cell by partitioning
features in the input data/feature matrix

"""
# Partitions features to a 2D grid and trains a model on each grid cell

import numpy as np
from sklearn.externals import joblib
import sklearn.base

################################################################################
def getCellData(X, y, min0, max0, min1, max1):
    """
    Returns cell data for features on columns 0 and 1 based on given bounds
    """
    Xcell = []
    ycell = []

    for x,label in zip(X, y):
        if (x[0] >= min0) and (x[0] < max0) and (x[1] >= min1) and (x[1] < max1):
            Xcell.append(x)
            ycell.append(label)

    return np.array(Xcell), np.array(ycell)

################################################################################
def partition_index(limits, value):

    # Fail safe for values in last boundary
    if value == limits[-1]:
        return len(limits)-2
            
    idx = 0
    while not((value >= limits[idx]) and (value < limits[idx + 1])):
        idx = idx + 1
    return idx

################################################################################
def bestNPredictions(probabilities, classes, n):
    """
    Return the best n predicted classes 
    for the given probabilities
    """
    assert(len(probabilities) == len(classes))
    reverse_sorted_probability_indices = sorted(range(0, len(probabilities)),
                                                key = lambda idx: probabilities[idx],
                                                reverse = True)
    return classes[reverse_sorted_probability_indices[0:n]]

################################################################################
class Grid2d():
    """
    Trains individual estimators for each grid cell by partitioning the 
    first two columns (features) in the feature matrix X using the given
    limits limits0 and limits1.
    
    """
    def __init__(self, limits0, limits1, estimator):
        self.limits0 = limits0
        self.limits1 = limits1
        self.estimator = estimator
                
    def fit(self, X, y):

        xs = self.limits0
        ys = self.limits1
        grid = {}
        
        for ix in range(0, len(xs)-1):
            for iy in range(0, len(ys)-1):
                
                Xcell, ycell  = getCellData(X, y, xs[ix], xs[ix + 1], ys[iy], ys[iy + 1])
                print("Cell (%d,%d): (%f-%f, %f-%f) samples = %d classes = %d" %
                      (ix, iy, xs[ix], xs[ix + 1], ys[iy], ys[iy + 1], Xcell.shape[0], len(np.unique(ycell))))
                cell = sklearn.base.clone(self.estimator)
                cell.fit(Xcell, ycell)
                #joblib.dump(cell, "cell_%d_%d.pkl" % (ix, iy), compress=True)
                print("%d-%d" % (ix, iy))
                grid["%d-%d" % (ix, iy)] = cell
        self.grid = grid


    def get_estimator(self, x0, x1):
        ix = partition_index(self.limits0, x0)
        iy = partition_index(self.limits1, x1)
        return self.grid["%d-%d" % (ix, iy)]
    
    def predict_row(self, x):
        estimator = self.get_estimator(x[0], x[1])
        return estimator.predict(x.reshape(1,-1))

    def predict(self, X):
        return np.array([self.predict_row(x) for x in X]).ravel()
            
    
    def predict_best_k_row(self, x,  k):
        estimator = self.get_estimator(x[0], x[1])
        probabilities = estimator.predict_proba(x.reshape(1, -1))
        bestK = bestNPredictions(probabilities.reshape(-1, 1), estimator.classes_, k)
        return bestK

    def predict_best_k(self, X,  k):
        return np.array([self.predict_best_k_row(x, k) for x in X])
    
