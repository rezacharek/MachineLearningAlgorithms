import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('..')
from perceptron import Perceptron

def main():

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])

    x_axis = []
    y_axis = []

    def onclick(event):
        if event.button == 1:
            print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
                (event.button, event.x, event.y, event.xdata, event.ydata))
            plt.plot(event.xdata, event.ydata, 'ro', color="green")
            fig.canvas.draw()
            x_axis.append( [event.xdata, event.ydata] )
            y_axis.append(-1)
        if event.button == 3:
            print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
                (event.button, event.x, event.y, event.xdata, event.ydata))
            plt.plot(event.xdata, event.ydata, 'ro', color="red")
            fig.canvas.draw()
            x_axis.append([event.xdata, event.ydata ] )
            y_axis.append(1)
        
            
           
    def press(event):
        print('press', event.key)
        sys.stdout.flush()
        if event.key == 'x':
            my_perceptron = Perceptron(np.array(x_axis), np.array(y_axis), 1)
            w = my_perceptron.train()
            w = w[0]
            print(w)

            a = -w[1] / w[2]
            xx = np.linspace(0, 10)
            yy = -w[0]/w[2]  + a * xx 
            plt.plot(xx,yy)
            ax = plt.gca()

            if (my_perceptron.prediction_error(np.array(x_axis), np.array(y_axis)) != 0):
                print("here")
                ax.set_facecolor('xkcd:crimson')
            else:
                print("here")
                ax.set_facecolor('xkcd:goldenrod')
            

            
            ax.set_facecolor((1.0, 0.47, 0.42))
            fig.canvas.draw()


    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_press_event', press)
    
    
    plt.show()
    


main()