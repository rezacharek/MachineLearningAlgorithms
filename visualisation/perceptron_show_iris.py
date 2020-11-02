import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from perceptron import Perceptron
from matplotlib.legend_handler import HandlerLine2D


def main():
    iris_data = pd.read_csv("Iris.csv") 
    iris_data = iris_data[:100] 
    x = iris_data.to_numpy()
    x = x[:,[2,4]]
    y = iris_data.to_numpy()
    y = y[:,5]
    y = np.array([1 if val == 'Iris-setosa' else -1 for val in y])

    print(x)

    points1 = plt.scatter(x[:50,0], x[:50,1], color = "orange", label = "Iris-Setosa")
    points2 = plt.scatter(x[50:,0], x[50:,1], color = "yellow", label = "Iris-Versicolor")
    # Create a legend for the first line.
    plt.xlabel("Sepal Length in cm")
    plt.ylabel("Petal Length in cm")

    first_legend = plt.legend(handles=[points1], loc='lower right')

    # Add the legend manually to the current Axes.
    ax = plt.gca().add_artist(first_legend)

    # Create another legend for the second line.
    plt.legend(handles=[points2], loc='upper right')
    
    my_perceptron =  Perceptron(x, y, 1)
    w = my_perceptron.train()
    w = w[0]

    a = -w[1] / w[2]
    xx = np.linspace(1.5, 5.5)
    yy = -w[0]/w[2]  + a * xx 
    plt.plot(xx,yy)
    ax = plt.gca()

    plt.show()


    # myperceptron.train()
    # self.assertListEqual(myperceptron.predict(test_x).tolist(),test_y.tolist())
main()