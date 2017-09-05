"""""
Filename:    perceptron.py

Description: This file contains a function to perform and visualize a n-D perceptron. Finds the 
             weight vector W such that: 0 = w_0 + w_1 * x_1 + w_2 * x_2 * ... * w_i * x_i
"""""

from datetime import datetime

import numpy as np

from benchmark import Benchmark
from point import Point
from visualize import Visualize


def perceptron(c, n, gamma, i, scale=1, plot=Visualize.plot_parallel, epoch_plot=True,
               pause=0.01):
    """ Implementation of the perceptron procedure! Generates a random data-set specified by the 
    parameters and visualizes the process.
    
    :param c: Training rate. Step size to correct the weights if there exists a misclassification.
    :param n: Size of the training data to generate.
    :param gamma: Minimum separation between the two "classes" of data in training data.
    :param i: Dimensionality of the points themselves (2D, 3D, etc...).
    :param scale: Scale of the data and the weight vectors. Defaults to 1.
    :param plot: The plot function to use. Defaults to parallel weight plots.
    :param epoch_plot: Display the plot per epoch, or per data point.
    :param pause: Seconds to pause between epochs or points (dependent on epoch_plot).
    :return: The training data set, the decision boundary weights, and the final weights.
    """
    # Generate the training data.
    d, w_star = Benchmark.generate(n, gamma, i, scale=scale)

    # We initialize a weight vector of size [1*(i + 1)]. Weights are random between 0 and 1.
    w = np.random.sample(i + 1)

    # Begin the perceptron!
    is_misclassified = True
    while is_misclassified:
        # Remember our previous weight vector.
        w_epoch = list(w)

        for j in range(0, len(d)):
            # Determine the output from the perceptron. Determine if this is correctly classified.
            y_j = Point.theta(d[j], w)

            # Adjust our weight vector. No change occurs if our output is correct (i.e. == 0).
            w[0] -= c * (d[j].ell + y_j)
            w[1:] -= c * (d[j].ell + y_j) * d[j].x

            # Display per data-point if defined. Use the given function.
            not epoch_plot and plot(d, w, w_star, d[j], scale, False, True, pause)

        # Display per epoch if defined. Use the given function.
        epoch_plot and plot(d, w, w_star, None, scale, False, True, pause)

        # If our weight vector has not changed, then our data is classified.
        is_misclassified = not all(map(lambda a, b: a == b, w_epoch, w))

    # Flash the solution 5 times. Leave for 5 seconds.
    plot(d, w, w_star, None, scale, False, True, 5, 5)
    return d, w_star, w
