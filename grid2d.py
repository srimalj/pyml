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
class Grid2d():
    
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
      
        #def predict_best_k(self, X,  k):
    # def predict(self, X):
    # def predict_proba(self, X):
    
