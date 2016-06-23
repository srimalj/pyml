import unittest
import numpy as np
import grid2d
from sklearn.neighbors import KNeighborsClassifier

def data2x2(n):
    X = np.random.rand(n,2)
    y = []
    for x in X:
        if (0 <= x[0] < 0.5) and (0.0 <= x[1] < 0.5):
            y.append(1)
        if (0 <= x[0] < 0.5) and (0.5 <= x[1]):
            y.append(2)
        if (0.5 <= x[0]) and (0.0 <= x[1] < 0.5):
            y.append(3)
        if (0.5 <= x[0]) and (0.5 <= x[1]):
            y.append(4)
    return X, y

class grid2dHelperFunctionsTest (unittest.TestCase):

    def testGetCellData(self):
        X = [[1,   2, 3,],
             [1.2, 0.5, 4],
             [1.3, 0.6, 4],
             [1.3, 1.0, 40],
             [2.0, 1.0, 6]]
        y = [1, 2, 3, 4, 5]

        Xcell, ycell = grid2d.getCellData(X, y, 1.0, 1.5, 0.5, 1)
        self.assertEqual(np.linalg.norm(ycell - [2,3]), 0)

    def test_partition_index(self):
        limits = [0, 0.2, 0.6, 0.8]
        self.assertEqual(grid2d.partition_index(limits, 0), 0)
        self.assertEqual(grid2d.partition_index(limits, 0.1), 0)
        self.assertEqual(grid2d.partition_index(limits, 0.2), 1)
        self.assertEqual(grid2d.partition_index(limits, 0.65), 2)
        self.assertEqual(grid2d.partition_index(limits, 0.8), 2)
        self.assertEqual(grid2d.partition_index(limits, 0.6), 2)
        self.assertEqual(grid2d.partition_index(limits, 0.3), 1)

    def testBestNPredictions(self):
        probabilities = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        classes = np.array(range(0, len(probabilities)))
        bestK = grid2d.bestNPredictions(probabilities, classes, 3)
        self.assertTrue(np.array_equal(bestK, np.array([4, 3 ,2])))

        probabilities = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        classes = np.array([4, 3, 2, 1, 0])
        bestK = grid2d.bestNPredictions(probabilities, classes, 3)
        self.assertTrue(np.array_equal(bestK, np.array([0, 1 ,2])))

        probabilities = np.array([0.1, 0.8, 0.3, 1.4, 0.5])
        classes = np.array(range(0, len(probabilities)))
        bestK = grid2d.bestNPredictions(probabilities, classes, 3)
        self.assertTrue(np.array_equal(bestK, np.array([3, 1, 4])))

        probabilities = np.array([0.1, 0.8, 0.3, 1.4, 0.5])
        classes = np.array([4, 3, 2, 1, 0])
        bestK = grid2d.bestNPredictions(probabilities, classes, 3)
        self.assertTrue(np.array_equal(bestK, np.array([1, 3, 0])))



class Grid2dTest (unittest.TestCase):

    def test_data2x2(self):
        X, y = data2x2(100)
        estimator =  KNeighborsClassifier(n_neighbors=1)
        grid = grid2d.Grid2d([0, 0.5, 1.0], [0, 0.5, 1.0], estimator)
        grid.fit(X, y)
        pred = grid.predict(X)
        #print pred, y           
        self.assertTrue(np.array_equal(pred, y))
        

        
        
if __name__ == '__main__':
    unittest.main()

