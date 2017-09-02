""""
Filename:    point.py

Description: This file contains an n-D point class, which hold the coordinates of the point and 
             the classification type.
"""""

import numpy as np


class Point(object):
    """ Attributes: x = Coordinates of the point instance.
                    ell = Classification (label) of the point. Defaults to 0.
    
        Methods:    find_d = Find the distance to the given hyperplane described by given weights.
                    theta = Given point and weight vector, classify the point (nondestructive).
                    classify = Given point and weight vector, classify the point (destructive).
                   
    """

    def __init__(self, x, ell=0):
        """ Constructor. 'x' is assumed to be a row vector containing the point's coordinates as 
        [x_0, x_1, x_2, ...]. 'ell' is the classification of the point, which defaults to 0 (not 
         valid when used in perceptron.py, as we only label points as -1, or 1).
        
        :param x: Row vector containing the coordinates of the point.
        :param ell: Classification of the point. Must be -1 or 1.
        """
        self.ell = 0 if not (ell == -1 or ell == -1) else ell
        self.x = x

    def find_d(self, w):
        """ Given a weight vector, determine the distance from the current point instance to the 
        hyperplane described by the 'w'. 
        
        Solution found here: https://math.stackexchange.com/a/1210685
        
        :param w: Weight vector. |w + 1| must equal |self.x|.
        :return: None if |w + 1| != |self.x|. Otherwise, the distance to the hyperplane of w.
        """
        return np.abs(np.dot(self.x, w[1:]) + w[0]) / (len(w) - 1)

    @staticmethod
    def theta(p, w):
        """ Given a point 'p' and a weight vector 'w' (with bias w_0), classify the point as 1 if it 
        falls to the right of the hyperplane and -1 otherwise. This only returns a classification.

        :param p Point to classify. |w+1| must equal |p.x|.
        :param w: Weight vector to classify the point 'p' with. |w+1| must equal |p.x|.
        :return: None if |w+1| != |p.x|. Otherwise, the classification of the point (-1 or 1).
        """
        if not len(p.x) + 1 == len(w):
            return None

        return 1 if np.dot(w[1:], p.x) + w[0] > 0 else -1

    @staticmethod
    def classify(p, w):
        """ Given a point 'p' and a weight vector 'w' (with bias w_0), classify the point as 1 if it 
        falls to the right of the hyperplane and -1 otherwise. This will destroy the previous 
        classification!

        :param p Point to classify. |w+1| must equal |p.x|.
        :param w: Weight vector to classify the point 'p' with. |w+1| must equal |p.x|.
        :return: None if |w+1| != |p.x|. Otherwise, the classification of the point (-1 or 1).
        """
        if not len(p.x) + 1 == len(w):
            return None

        p.ell = Point.theta(p, w)
        return p.ell
