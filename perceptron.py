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
        (number_of_rows, number_of_cols) = self.train_set_x.shape
        self.w = np.zeros((1, number_of_cols + 1))
        iter = 0
        self.historical_w = []
        while(iter < 10000):
            count_wrong_classification = 0

            for i in range(len(self.train_set_x)):
                x_vector = self.train_set_x[i]
                x_vector = np.concatenate(([1], x_vector))
                if self.train_set_y[i] * np.dot(self.w, x_vector) <= 0:
                    self.w = np.add(self.w,self.learning_rate*self.train_set_y[i]*x_vector)
                    self.historical_w.append(self.w[0])
                    count_wrong_classification += 1
            
            if count_wrong_classification == 0: break
            iter += 1
        return (self.w[0] ,self.historical_w)

    """Predicts the classification based on the input

    Attributes:
        test_set_x -- the x input of the new data
    """
    
    def predict(self, test_set_x):

        (number_of_rows, number_of_columns) = test_set_x.shape
        self.predicted_y = np.arange(number_of_rows)
        for i in range(number_of_rows):
            x_vector =  test_set_x[i]
            x_vector = np.concatenate(([1],x_vector))
            if np.dot(self.w, x_vector) > 0:
                self.predicted_y[i] = 1
            else:
                self.predicted_y[i] = -1
        
        return np.array(self.predicted_y)

    """Computes the prediction error

    Attributes:
        axis_x -- the x input of the data
        axis_y -- the correct prediction
    """

    def prediction_error(self, axis_x, axis_y):
        prediction_error = 0.0

        predicted_y = self.predict(axis_x)

        for i in range(len(predicted_y)):
            if (predicted_y[i] != axis_y[i]): prediction_error += 1.0
        
        return prediction_error/len(predicted_y)


                
