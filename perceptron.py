"""""
Filename:    perceptron.py

Description: This file is a script implementing the perceptron procedure. 
"""""

from point import Point
from visualize import Visualize
from benchmark import Benchmark

import numpy as np
import matplotlib.pyplot as plt


# We define a training rate c, which must exist between 0 and 1.
c = 0.5

# Generate the training data with N = 100, gamma = 0.1, and degree 2.
d, w_star = Benchmark.generate(100, 0.001, 2)

# We initialize a weight vector of size [1x3]. Weights are random between 0 and 1.
w = np.random.sample(3)

# Begin the 2D perceptron!
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

    # Display the current state.
    Visualize.plot_2d(d, w, w_star, dynamic=True)

    # If our weight vector has not changed, then our data is classified.
    is_misclassified = not all(map(lambda a, b: a == b, w_epoch, w))

# Compare w_star and w.
print (w_star)
print (w)