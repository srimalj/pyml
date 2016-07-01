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

class BayesClassifierKDETest (unittest.TestCase):

    def testPosteriors(self):
        bc = bayes.BayesClassifierKDE()
        m1 = np.array([100, 0])
        m2 = np.array([0, 100])
        X1 = np.random.multivariate_normal(m1, np.eye(2, 2), 100)
        X2 = np.random.multivariate_normal(m2, np.eye(2, 2), 100)
        X = np.vstack((X1, X2))
        y = np.hstack((np.ones(100), 2 * np.ones(100)))
        bc.fit(X,y)
       
        p1 =  bc.posteriors(m1)
        self.assertGreater(p1[0], p1[1])
        p2 =  bc.posteriors(m2)
        self.assertGreater(p2[1], p2[0])
        
    def testPredictk(self):
        bc = bayes.BayesClassifierKDE()
        m1 = np.array([100, 0])
        m2 = np.array([0, 100])
        X1 = np.random.multivariate_normal(m1, np.eye(2, 2), 100)
        X2 = np.random.multivariate_normal(m2, np.eye(2, 2), 100)
        X = np.vstack((X1, X2))
        y = np.hstack((np.ones(100), 2 * np.ones(100)))
        bc.fit(X,y)
        print bc.predictk(np.vstack((m1, m2)), 2)
        
        
        
if __name__ == '__main__':
    unittest.main()

