import unittest
import numpy as np
from elm import GenELMClassifier
from random_layer import RBFRandomLayer, MLPRandomLayer
from sklearn.linear_model import LogisticRegression
from ensemble import ELMEnsemble

class ensembleTest (unittest.TestCase):

    def testELMEnsemble(self):
        nrows = 5;
        ncols = 8;
        Xtrain = np.random.random([nrows, ncols])
        Xeval = np.random.random([nrows, ncols])
        ytrain = np.array([0, 1, 2, 1, 0])
                
        classifier = ELMEnsemble(n_learners=2, n_hidden_layer_nodes= 10,\
                                 activation_func='tanh',\
                                 nfolds=1)
        
        classifier.fit(Xtrain, ytrain)
        train_probs = classifier.predict_proba(Xtrain)
        print ytrain
        print train_probs
        for row in range(0,Xtrain.shape[0]):
            self.assertEqual(ytrain[row], np.argmax(train_probs[row,:]))
        #print classifier.predict_proba(Xeval)

        # print "Training:"
        # print "decision_function:"
        # print classifier.decision_function(X)
        # print classifier.predict_proba(X)
        
        # print "Testing:"
        # X = np.random.random([nrows, ncols])
        # classes = classifier.predict(X)
        # probs = classifier.predict_proba(X)
        # for row in range(0,nrows):
        #     self.assertEqual(classes[row], np.argmax(probs[row,:]))

        # print "decision_function:"
        # print classifier.decision_function(X)
        # print "probabilities:"
        # print probs
        # print "classes:"
        # print classes
        
if __name__ == '__main__':
    unittest.main()
