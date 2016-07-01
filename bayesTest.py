import unittest
import numpy as np
import bayes

class bayesTest (unittest.TestCase):

    def testPriorDistribution(self):
        
        a = np.array([1,1,2,2])
        r = bayes.priorDistribution(a)
        self.assertEqual(r[0], 0.5)
        self.assertEqual(r[1], 0.5)


        a = np.array([1,1,2,2,3,3,3,3])
        r = bayes.priorDistribution(a)
        self.assertEqual(r[0], 0.25)
        self.assertEqual(r[1], 0.25)
        self.assertEqual(r[2], 0.5)

if __name__ == '__main__':
    unittest.main()

