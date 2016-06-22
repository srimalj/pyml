import unittest
import numpy as np
import grid2d

class grid2dTest (unittest.TestCase):

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

    # def testBestNPredictions(self):
    #     bestK = grid2d.bestNPredictions([0.1, 0.2, 0.3, 0.4, 0.5], range(0,6), 3)
    #     print(bestK)
        #self.assertEqual([0,1,2], [0, 1, 2])
        
if __name__ == '__main__':
    unittest.main()
