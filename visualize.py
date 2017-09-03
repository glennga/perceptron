""""
Filename:    visualize.py

Description: This file contains a set of methods (wrapped in a Visualize class) to display the 
             perceptron in real-time or as a static image.
"""""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas.plotting import parallel_coordinates

from point import Point


class Visualize(object):
    """ Methods: plot_2d = Produce a plot of the given line and points.
                 plot_parallel_weights = Produce a parallel plot of the two weight vectors.
                 plot_matrix = Produce a matrix of plots given the line and points. 
    """

    @staticmethod
    def plot_2d(d, w, w_star=None, focus=None, scale=1, show=False, dynamic=False, pause=0.01,
                flash_w=0):
        """ Produce a plot given the data-set "d" and the weight vector "w". If defined, 
        draw the decision boundary line from "w_star" and adjust the axis according to 
        "scale". If desired, the call to "plt.show() and plt.ion()" can be omitted.
        
        :param d: Data-set containing list of Point instances.
        :param w: Weight vector to derive line from.
        :param w_star: Weight vector to derive decision boundary from.
        :param focus: Point to focus. Represents the current point selected.
        :param scale: The axis to display.
        :param show: If false, do not call "plt.show()".
        :param dynamic: If false, do not call "plt.ion()".
        :param pause: Number of seconds to pause plot for.
        :param flash_w: Used to indiciate that we have found a solution. Flashes w line.
        :return: -1 if the given data-set is not two-dimensional. 0 otherwise.
        """
        if len(d[0].x) != 2:
            return -1

        # Enable interactive mode if desired. Clear the previous plot.
        dynamic and plt.cla() and plt.ion()

        # Plot our data-set. Colored red and blue.
        [plt.scatter(p.x[0], p.x[1], c=("r" if p.ell == -1 else "b")) for p in d]
        plt.axis([-scale, scale, -scale, scale])

        # Plot the line derived from the weights.
        pl_x = np.arange(-2 * scale, 2 * scale)
        pl_y = pl_x * -w[1] / w[2] - w[0] / w[2]
        plt.plot(pl_x, pl_y, c="m")

        # Circle the focus point if defined.
        if focus is not None:
            plt.scatter(focus.x[0], focus.x[1], s=80, facecolors='none', edgecolors='m')

        # If defined, plot the decision boundary.
        if w_star is not None:
            db_x = np.arange(-2 * scale, 2 * scale)
            db_y = db_x * -w_star[1] / w_star[2] - w_star[0] / w_star[2]
            plt.plot(db_x, db_y, "--", c="k")

            # Shade in area classified by w (not w_star).
            plt.fill_between(pl_x, -np.ones(len(pl_x)) * scale, pl_y, facecolor="m", alpha=0.1)

        # We flash the line red, blue, and magenta if specified.
        for f in range(flash_w):
            plot_and_pause = lambda c: plt.plot(pl_x, pl_y, c=c) and plt.pause(0.1)
            [plot_and_pause(a) for a in ['r', 'b', 'm']]

        # Dynamic mode must be toggled off to call show.
        show and not dynamic and plt.show()

        # Dynamic mode must be on, show must be off to pause.
        not show and dynamic and plt.pause(pause)

        return 0

    @staticmethod
    def plot_parallel_weights(w, w_star, show=False, dynamic=False, pause=0.01, flash_w=0):
        """ Produce a parallel coordinate given the weight vectors "w" and "w_star". If desired, 
        the call to "plt.show() and plt.ion()" can be omitted. 

        :param w: Weight vector to derive line from.
        :param w_star: Weight vector to derive decision boundary from.
        :param show: If false, do not call "plt.show()".
        :param dynamic: If false, do not call "plt.ion()".
        :param pause: Number of seconds to pause plot for.
        :param flash_w: Used to indiciate that we have found a solution. Flashes w line.
        :return: None.
        """
        # Enable interactive mode if desired. Clear the previous plot.
        dynamic and plt.subplot(111) and plt.cla() and plt.ion()

        def plot_vary_w(w_c):
            """ Produce a parallel coordinate plot. Vary the color of the 'w' line.
            
            :param w_c: Color of 'w' line.
            :return: None.
            """
            # Label and title the plot.
            plt.xlabel('Dimension (x_0 represents bias)')
            plt.title('Decision Boundary vs. Current Weight Vector')

            # Prepare weight-vectors as a data-frame. Find the X coefficients of the equation:
            # x_m = (x_0 + x_1 * w_1 + x_2 * w_2 + ...) / w_m
            weight_data = [['w'], ['w_star']]
            for a in range(0, len(w) - 1):
                weight_data[0].append(w[a] / w[len(w) - 1])
                weight_data[1].append(w_star[a] / w_star[len(w_star) - 1])

            parallel_coordinates(pd.DataFrame(weight_data), 0, color=[w_c, 'k'])

        # Plot as regular...
        plot_vary_w('b')

        # We flash the line blue and green if specified.
        for f in range(flash_w):
            plt.cla()
            plot_vary_w('g')
            plt.pause(0.25)
            plt.cla()
            plot_vary_w('b')
            plt.pause(0.25)

        # Dynamic mode must be toggled off to call show.
        show and not dynamic and plt.show()

        # Dynamic mode must be on, show must be off to pause.
        not show and dynamic and plt.pause(pause)

    @staticmethod
    def plot_parallel(d, w, w_star=None, focus=None, scale=1, show=False, dynamic=False,
                      pause=0.01, flash_w=0):
        """ Wrapper for the 'plot_parallel_weights' function. 'd', 'focus', and 'scale' do nothing.

        :param d: Data-set containing list of Point instances. (NOT USED).
        :param w: Weight vector to derive line from.
        :param w_star: Weight vector to derive decision boundary from.
        :param focus: Point to focus. Represents the current point selected. (NOT USED).
        :param scale: The axis to display. (NOT USED).
        :param show: If false, do not call "plt.show()".
        :param dynamic: If false, do not call "plt.ion()".
        :param pause: Number of seconds to pause plot for.
        :param flash_w: Used to indiciate that we have found a solution. Flashes w line.
        :return: None.
        """
        Visualize.plot_parallel_weights(w, w_star, show, dynamic, pause, flash_w)

    @staticmethod
    def plot_matrix(d, w, w_star=None, focus=None, scale=1, show=False, dynamic=False, pause=0.01,
                    flash_w=0):
        """ Produce a matrix of plots given the data-set "d" and the weight vector "w". For each 
        diagonal (where the dimensions are equal), a histogram of the data itself is displayed. If
        defined, draw the decision boundary line from "w_star" and adjust the axis according to
        "scale". If desired, the call to "plt.show() and plt.ion()" can be omitted.
        
        Note: I initially thought this would be cool to visualize, but it doesn't really make 
        sense to plot the points because each plot shows data that is not linearly separable, 
        even though they are when combined with other dimensions. Because of this, I omitted the 
        plotting of the points themselves (which makes it boring). 

        :param d: Data-set containing list of Point instances.
        :param w: Weight vector to derive line from.
        :param w_star: Weight vector to derive decision boundary from.
        :param focus: Star to focus. Represents the current star selected.
        :param scale: The axis to display.
        :param show: If false, do not call "plt.show()".
        :param dynamic: If false, do not call "plt.ion()".
        :param pause: Number of seconds to pause plot for.
        :param flash_w: Used to indiciate that we have found a solution. Flashes w line.
        :return: None.
        """
        degree = len(d[0].x)

        # Enable interactive mode if desired. Clear the previous plot.
        dynamic and plt.subplot(111) and plt.cla() and plt.ion()

        # For every dimension, we plot against another.
        for m in range(degree):
            for n in range(degree):
                plt.subplot(degree, degree, degree * m + n + 1)

                # If this an edge plot, label appropriately.
                m == 0 and plt.title("x" + str(n), fontsize=10)
                n == 0 and plt.ylabel("x" + str(m), fontsize=10)

                # We are comparing against the same dimension. Display a histogram instead.
                if m == n:
                    plt.hist([a.x[m] for a in d])

                else:
                    # Otherwise, we set dimension 1 and 2 of this plot as m and n here.
                    # d_mn = [Point([a.x[m], a.x[n]], a.ell) for a in d]
                    d_mn = [Point([0, 0], -1)]

                    # Keep our bias. Represent the weights with respect to m and n.
                    w_mn = [w[0], w[m + 1], w[n + 1]]
                    w_star_mn = [w_star[0], w_star[m + 1], w_star[n + 1]]

                    # Given a focus, define one with respect to m and n.
                    focus_mn = None if focus is None else Point([focus.x[m], focus.x[n]])

                    Visualize.plot_2d(d_mn, w_mn, w_star_mn, focus_mn, scale, False, False, 0.0,
                                      flash_w)

        # This is probably going to be tight!
        plt.tight_layout()

        # Dynamic mode must be toggled off to call show.
        show and not dynamic and plt.show()

        # Dynamic mode must be on, show must be off to pause.
        not show and dynamic and plt.pause(pause)
