import unittest
import numpy as np
import pandas as pd
import sys 
sys.path.append('..')
from adaline import Adaline


class TestAdaline(unittest.TestCase):
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

        myadaline =  Adaline(train_x, train_y, 1)
        myadaline.train()


        predicted_y = myadaline.predict(test_x)

        print("The accuracy score is ",end='')
        print( (predicted_y == test_y).sum()/19.0,end='')
        print("%")

        self.assertListEqual(myadaline.predict(test_x).tolist(),test_y.tolist())



if __name__ == '__main__':
    unittest.main()