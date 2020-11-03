from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
import sys
sys.path.append('..')
from perceptron import Perceptron


#1 and 8

indexes_of_one = []
indexes_of_eight = []

for i in range(len(Y_train)):
    if (Y_train[i] == 1 and len(indexes_of_one) < 6):
        indexes_of_one.append(i)
    if (Y_train[i] == 8 and len(indexes_of_eight) < 6):
        indexes_of_eight.append(i)
    
    if len(indexes_of_one) >= 5 and len(indexes_of_eight) >= 5: break
    

def plot_(all_digits, index_digits_of_interest, w):
    all_digits_in_one_matrix = np.zeros((56, 28 * len(index_digits_of_interest)))

    digits = [None]*len(index_digits_of_interest)

    for row in range(28):
        for col in range(28):
            for i in range(len(index_digits_of_interest)):
                pixels = np.array(all_digits[index_digits_of_interest[i]], dtype='uint8')
                digits[i] = pixels.ravel()
                if True: 
                    all_digits_in_one_matrix[row][col + 28*i] = pixels[row][col]
                else:
                    all_digits_in_one_matrix[row+ 28][col + 28*i] = pixels[row][col]
    

    # plt.imshow(all_digits_in_one_matrix, cmap='gray')
    # plt.show()
    return np.array(digits)




digits = plot_(X_train, indexes_of_one + indexes_of_eight, 0)

y = []
for i in range(len(indexes_of_one + indexes_of_eight)):
    if i < len(indexes_of_one):
        y.append(1)
    else:
        y.append(-1)
my_perceptron =  Perceptron(digits, y, 1)
w, historical_w = my_perceptron.train()

def plot(all_digits, index_digits_of_interest, w):
    all_digits_in_one_matrix = np.zeros((56, 28 * len(index_digits_of_interest)))

    digits = [None]*len(index_digits_of_interest)

    for row in range(28):
        for col in range(28):
            for i in range(len(index_digits_of_interest)):
                pixels = np.array(all_digits[index_digits_of_interest[i]], dtype='uint8')
                to_test =  pixels.ravel()
                to_test = np.concatenate(([1],to_test))
                if  np.dot(to_test,w)<0: 
                    all_digits_in_one_matrix[row][col + 28*i] = pixels[row][col]
                else:
                    all_digits_in_one_matrix[row+ 28][col + 28*i] = pixels[row][col]
    

    plt.imshow(all_digits_in_one_matrix, cmap='gray')
    plt.draw()
    plt.pause(1)
    plt.clf()

    # return np.array(digits)
for w_ in historical_w:
    plot(X_train, indexes_of_one + indexes_of_eight, w_)