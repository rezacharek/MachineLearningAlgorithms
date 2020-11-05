import unittest
import numpy as np
import pandas as pd 
import sys 
sys.path.append('..')
from point import Point 

class TestPoint(unittest.TestCase):

    def test_sub(self):
        first_point = Point([1,2,3,4,5])
        second_point = Point([1,2,3,4,5])
        self.assertListEqual(first_point-second_point, [0,0,0,0,0])
    
    def test_sub_two(self):
        first_point = Point([1,34,6,7,0,19,0])
        second_point = Point([1,3,5,8,6,2,3])

        self.assertListEqual(first_point-second_point, [0,31,1,1,6,17,3])

    def test_lt(self):

        self.assertTrue(Point([1,2]) < Point([3,4]))
        self.assertTrue(Point([1,2,1,1,1]) < Point([1,2,3,4,5]))

if __name__ == '__main__':
    unittest.main()