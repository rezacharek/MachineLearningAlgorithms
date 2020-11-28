import numpy as np
# from errors import WrongConstant

class Kernel:
    """Initialisation of the Kernel class

    Attributes:
        set_x -- the X input that will be kernalised
        set_y -- the Y input that represents the classes
    """

    def __init__(self, set_x, set_y):
        self.set_x = set_x
        self.set_y = set_y

    """Returns the kernalised x and y vectors """
    
    def kernalised(self):
        self.kernalised_x = self.kernalise(self.set_x)
        return self.kernalised_x 


    def kernalise(self, set_x):
        n_rows, n_cols = set_x.shape
        kernalised_x = [None]*n_rows

        for row_idx in range(n_rows):
            kernalised_x[row_idx] = [1]
            for col_idx in range(n_cols):

                if len(kernalised_x[row_idx]) == 1:
                    start = 0
                else:
                    start = 1

                for i in range(start, len(kernalised_x[row_idx])):
                    current_multiplication = kernalised_x[row_idx][i]
                    current_multiplication *= set_x[row_idx][col_idx]
                    kernalised_x[row_idx].append(current_multiplication)
        
        return kernalised_x



