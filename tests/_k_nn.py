import unittest
import numpy as np 
import pandas as pd 
import sys 
sys.path.append('..')
from k_nn import K_NN

class TestKNN(unittest.TestCase):

    def test_closest_k_points(self):
        
        my_model = K_NN([[1,2],[1,4],[1,5],[2,3],[4,5],[-5,6],[1,1]], [-1,-1,1,1,1,1,-1], 3)
        random_point = [1,1]

        closest_to_random_point = my_model.closest_k_points(random_point)
        self.assertCountEqual(closest_to_random_point,[[1,2],[2,3],[1,1]],  )

    def test_predict(self):
        my_model = K_NN([[1,2],[1,4],[1,5],[2,3],[4,5],[-5,6],[1,1]], [-1,-1,1,1,1,1,-1], 3)
        random_point = [1,1]
        self.assertEqual(my_model.predict(random_point), -1)
        self.assertEqual(my_model.predict([-5,6]), 1)
        self.assertEqual(my_model.predict([-5,5]), 1)
        self.assertEqual(my_model.predict([0,0]), -1)

    def test_predict_3D(self):
        my_model = K_NN([[1,2,-1],[1,4,0],[1,5,25],[2,3,33],[4,5,57],[-5,6,-23],[1,1,0]], [1,1,1,-1,1,1,1], 3)
        random_point = [1,1,1]
        self.assertEqual(my_model.predict(random_point), 1)
        self.assertEqual(my_model.predict([-5,6,-22]), 1)

if __name__ == '__main__':
    unittest.main()