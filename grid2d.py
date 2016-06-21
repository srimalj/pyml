# Partitions features to a 2D grid and trains a model on each grid cell
import numpy as np

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

# class grid2d():
#     def __grid2d__(X, xlims, ylims, estimator):
        
        
#     def fit()

    
