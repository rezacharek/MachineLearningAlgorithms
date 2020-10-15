import numpy as np
from errors import WrongConstant

class Perceptron:
    """Initialisation of the perceptron model

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
    
    """Returns the sign of the dot product between the vector w and the vector x_i, multiplied by the value of y_i
    
    Attributes:
        index -- integer
        w -- the vector that is normal to the hyperplane
        x -- the x vector of the model
        y -- the y vector of the model
    """

    def sign_of_classification(self, index, w, *argv):
        if not argv:
            return self.train_set_y[index] * np.dot(w, self.train_set_x[index])
        else:
            return argv[1][index] * np.dot(w, argv[0][index])

    """Training the model"""

    def train(self):
        (number_of_rows, _) = self.train_set_x.shape()
        self.w = np.array((number_of_rows, 1))

        while(True):
            count_wrong_classification = 0

            for i in range(len(self.train_set_x)):
                if self.sign_of_classification(i,self.w):
                    self.w += self.learning_rate*self.train_set_y[i]*self.train_set_x[i] 
                    count_wrong_classification += 1
            
            if count_wrong_classification == 0: break
        return self

    """Predicts the classification based on the input

    Attributes:
        test_set_x -- the x input of the new data
    """
    
    def predict(self, test_set_x):

        (number_of_rows, _) = test_set_x.shape()
        self.predicted_y = np.array((number_of_rows,1))

        for i in range(len(test_set_x)):
            if np.dot(self.w, test_set_x[i]) > 0:
                self.predicted_y[i] = 1
            else:
                self.predicted_y[i] = -1
        
        return self.predicted_y




                
