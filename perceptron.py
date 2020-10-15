import numpy as np
import pandas as pd 

class Perceptron:

    def __init__(self, train_set_x, train_set_y, learning_rate = 1):
        self.train_set_x = train_set_x
        self.train_set_y = train_set_y
        

        if (learning_rate <= 0):
            