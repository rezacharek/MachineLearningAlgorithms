import numpy as np
from errors import WrongConstant

class Adaline:
    """Initialisation of the Adaline mode

    Attributes:
        train_set_x -- the training set of the X input
        train_set_y --  the training set of the Y input
        learning_rate -- the learning rate of the model, must be greater than 0
    """

    def __init__(self, train_set_x, train_set_y, learning_rate):
        self.train_set_x = train_set_x
        self.train_set_y = train_set_y
        
        if (learning_rate <= 0):
            raise WrongConstant(learning_rate)
        else:
            self.learning_rate = learning_rate
    
    """Trains the model"""
    def train(self):
        (number_of_rows, number_of_cols) = self.train_set_x.shape
        self.w = np.zeros((1, number_of_cols + 1))
        self.train_set_x = np.c_[self.train_set_x,np.ones(number_of_rows)]

        for i in range(len(self.train_set_x)):

                if np.dot(self.w, self.train_set_x[i]) < 0:
                    self.w = np.add(self.w,self.learning_rate*(self.train_set_y[i] + 1)*self.train_set_x[i]) 
                else:
                    self.w = np.add(self.w,self.learning_rate*(self.train_set_y[i] - 1)*self.train_set_x[i]) 
        
        return self.w
    
    """Predicts the classification based on the input

    Attributes:
        test_set_x -- the x input of the new data
    """
    def predict(self, test_set_x):
        (number_of_rows, number_of_cols) = test_set_x.shape
        test_set_x = np.c_[test_set_x,np.ones(number_of_rows)]
        self.predicted_y = np.arange(number_of_rows)

        for i in range(number_of_rows):
            if np.dot(self.w, test_set_x[i]) > 0:
                self.predicted_y[i] = 1
            else:
                self.predicted_y[i] = -1
        
        return np.array(self.predicted_y)



