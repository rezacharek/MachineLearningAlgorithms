import unittest
import numpy as np
import pandas as pd
import sys 
sys.path.append('..')
from kernel import Kernel

class TestKernel(unittest.TestCase):

    def test_kernalise(self):
        x = np.array([[1,2,3]])
        y = np.array([1,1,1])
        my_kernel = Kernel(x,y)

        self.assertListEqual(my_kernel.kernalised(), [[1,1,2,3,6]])
        x = np.array([[1,2,3,4,5]])
        y = np.array([1,1,1,1])
        my_kernel = Kernel(x,y)
        self.assertListEqual(my_kernel.kernalised(), [[1, 1, 2, 3, 6, 4, 8, 12, 24, 5, 10, 15, 30, 20, 40, 60, 120]])
if __name__ == '__main__':
    unittest.main()