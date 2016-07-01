import unittest
import numpy as np
import common

class commonTest (unittest.TestCase):

    def testLargestk(self):
        
        # a = np.array([1,2,3,4,5,6,7,8,9,10])
        # result = common.largestK(a, 1)
        # self.assertEqual(result[0], 9)

        # result = common.largestK(a, 3)
        # self.assertEqual(result[0], 9)
        # self.assertEqual(result[1], 8)
        # self.assertEqual(result[2], 7)

        
        a = np.array([2, 1])
        result = common.largestK(a, 1)
        self.assertEqual(result[0], 0)
        self.assertEqual(len(result), 1)
        

        
        result = common.largestK(a, 2)
        self.assertEqual(result[0], 0)
        self.assertEqual(result[1], 1)
        
        result = common.largestK(a, 4)
        self.assertEqual(result[0], 0)
        self.assertEqual(result[1], 1)

        a = np.array([2])

        result = common.largestK(a, 1)
        self.assertEqual(result[0], 0)
        self.assertEqual(len(result), 1)
        
        result = common.largestK(a, 10)
        self.assertEqual(result[0], 0)
                
if __name__ == '__main__':
    unittest.main()
