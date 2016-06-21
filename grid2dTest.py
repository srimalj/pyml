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

if __name__ == '__main__':
    unittest.main()
