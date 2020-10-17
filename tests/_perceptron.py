import unittest
import numpy as np
import pandas as pd
import sys 
sys.path.append('..')
from perceptron import Perceptron as Perceptronnn
from sklearn.datasets import load_iris
from sklearn.utils import check_random_state
from sklearn.linear_model import Perceptron as Percy


class TestPerceptron(unittest.TestCase):

    def test_sign_of_classification(self):
        
        myperceptron =  Perceptronnn(np.zeros((2,1)), np.zeros((2)), 1)

        x_vector = [[1,1,1], [1,2,3], [-1,-1,-1]]
        w_vector = [1,1,1]
        y_vector = [1, -1, -1]
        expected_result = [True,False,True]
        given_result = []
        for i in range(3):
            given_result.append( myperceptron.sign_of_classification(i,w_vector, x_vector, y_vector) > 0)
        self.assertListEqual(given_result, expected_result)

    def test_iris_data_set(self):
        iris_data = pd.read_csv("Iris.csv") 
        iris_data = iris_data[:100] 
        iris_data = iris_data.sample(frac=1)
        x = iris_data[:100].to_numpy()
        x = x[:,:5]
        y = iris_data[:100].to_numpy()
        y = y[:,5]
        y = np.array([1 if val == 'Iris-setosa' else -1 for val in y] )

        train_x = x[:80]
        train_y = y[:80]

        test_x = x[81:]
        test_y = y[81:]

        myperceptron =  Perceptronnn(train_x, train_y, 1)
        myperceptron.train()
        self.assertListEqual(myperceptron.predict(test_x).tolist(),test_y.tolist())

if __name__ == '__main__':
    unittest.main()