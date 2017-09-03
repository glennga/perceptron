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


def perceptron(c, n, gamma, i, scale=1, epoch_plot=True, pause=0.01):
    """ Implementation of the perceptron procedure! Generates a random data-set specified by the 
    parameters and visualizes the process.
    
    :param c: Training rate. Step size to correct the weights if there exists a misclassification.
    :param n: Size of the training data to generate.
    :param gamma: Minimum separation between the two "classes" of data in training data.
    :param i: Dimensionality of the points themselves (2D, 3D, etc...).
    :param scale: Scale of the data and the weight vectors. Defaults to 1.
    :param epoch_plot: Display the plot per epoch, or per data point.
    :param pause: Seconds to pause between epochs or points (dependent on epoch_plot).
    :return: The training data set, the decision boundary weights, and the final weights.
    """
    # Seed our RNG with time. Ensure that the seed is different from that used in Benchmark.
    np.random.seed(datetime.now().toordinal() + 1)

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

            # Display per data-point if defined.
            not epoch_plot and Visualize.plot(d, w, w_star, scale=scale, focus=d[j],
                                                       dynamic=True, pause=pause)

        # Display per epoch if defined
        epoch_plot and Visualize.plot(d, w, w_star, scale=scale, dynamic=True, pause=pause)

        # If our weight vector has not changed, then our data is classified.
        is_misclassified = not all(map(lambda a, b: a == b, w_epoch, w))

    return d, w_star, w


# Construct a dataset with 50 3D points and a margin of 0.2 using a scale of 1. Visualize the
# peceptron and plot every epoch. The step size per correction is 0.5.
perceptron(0.001, 30, 0.0001, 3, 1, True, 0.1)
