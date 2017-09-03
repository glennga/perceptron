""""
Filename:    benchmark.py

Description: This file contains an n-D training data generation class. This generates a set of 
             points matching specified criteria.
"""""

from matplotlib import pyplot as plt
from datetime import datetime
from point import Point
import numpy as np


class Benchmark(object):
    """ Methods:    generate = Generate a list of random points that meet the given criteria. """

    @staticmethod
    def generate(n, gamma, i, scale=1):
        """ Generate a random list of points that meet the given criteria. 
        
        :param n: Number of points to generate. These are randomly plotted.
        :param gamma: Minimum separation between the two "classes" of data in D.
        :param i: Dimensionality of the points themselves (2D, 3D, etc...).
        :param scale: The scale of the points and the classifying weight vector. Defaults to 1.
        :return: Random list of points that meet the given criteria, as well as w_star used to 
                 classify the points.
        """
        # Seed our RNG with time.
        np.random.seed(datetime.now().toordinal())

        # Generate our decision boundary (w_star). Scale appropriately. We set our bias to **0**.
        w_star = np.append([0], ((2 * np.random.rand(i)) - 1) * scale)

        # Generate our random points. Gamma condition must be met for each point added.
        d = []
        while len(d) < n:
            p = Point(((2 * np.random.rand(i)) - 1) * scale)
            p.find_d(w_star) > gamma and d.append(p)

        # We classify each point given w_star. Perform for each point in D.
        list(map(lambda a: Point.classify(a, w_star), d))

        return d, w_star