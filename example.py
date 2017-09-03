"""""
Filename:    example.py

Description: This file contains example usages of the perceptron function. 
"""""

from perceptron import perceptron
from visualize import Visualize

# Construct a dataset with 30 2D points and a margin of 0.01 using the default scale of 1.
# Visualize the dataset and weights with 2D plot, and plot every epoch. The perceptron step size
# is 0.001.
perceptron(c=0.001, n=30, gamma=0.01, i=2, plot=Visualize.plot_2d)

# Construct a dataset with 1000 10D points and a margin of 0.001 using the default scale of 1.
# Visualize the weights and plot every epoch. The perceptron step size is 0.001.
perceptron(c=0.001, n=1000, gamma=0.001, i=10, plot=Visualize.plot_parallel)

# Construct a dataset with 30 4D points and a margin of 0.01 using the default scale of 1.
# Visualize the weights (scatter matrix) and plot every point. The perceptron step size is 0.001.
perceptron(c=0.001, n=30, gamma=0.01, i=4, plot=Visualize.plot_matrix)
