""""
Filename:    visualize.py

Description: This file contains a set of methods (wrapped in a Visualize class) to display the 
             perceptron in real-time or as a static image.
"""""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from pandas.plotting import parallel_coordinates


class Visualize(object):
    """ Methods: plot_2d = Produce a plot of the given line and points.
                 plot_parallel = Produce a parallel plot of the two weight vectors.
                 plot_3d = Produce a matrix of plots given the line and points. 
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
            plt.xlabel('Coefficient for X term')
            plt.title('Decision Boundary (black) and Current Weight Vector (blue)')

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
    def plot_3d(d, w, w_star=None, focus=None, scale=1, show=False, dynamic=False, pause=0.01,
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
        :param flash_w: Used to indiciate that we have found a solution. Flashes w plane.
        :return: -1 if the given data-set is not three-dimensional. 0 otherwise.
        """
        if len(d[0].x) != 3:
            return -1

        # Enable interactive mode if desired. Clear the previous plot.
        dynamic and plt.cla() and plt.ion()

        # Our projection is now 3D.
        ax = plt.gca(projection='3d')

        # Plot our data-set. Colored red and blue.
        [ax.scatter(p.x[0], p.x[1], p.x[2], c=("r" if p.ell == -1 else "b")) for p in d]

        # Plot the plane derived from the weights.
        xx, yy = np.meshgrid(np.arange(-scale, scale, scale / 30),
                             np.arange(-scale, scale, scale / 30))
        w_zz = -w[1] / w[3] * xx + -w[2] / w[3] * yy - w[0] / w[3]
        w_plane = ax.plot_surface(xx, yy, w_zz, cmap=cm.seismic, alpha=0.6)

        # Circle the focus point if defined.
        if focus is not None:
            ax.scatter(focus.x[0], focus.x[1], focus.x[2], s=80, facecolors='none', edgecolors='m')

        # If defined, plot the decision boundary.
        if w_star is not None:
            w_star_zz = -w_star[1] / w_star[3] * xx + -w_star[2] / w_star[3] * yy - w_star[0] / \
                                                                                    w_star[3]
            ax.plot_surface(xx, yy, w_star_zz, cmap=cm.binary, alpha=0.6)

        # We flash the plane red, blue, and magenta if specified.
        for f in range(flash_w):
            plot_and_pause = lambda c: w_plane.set_cmap(c) or plt.pause(0.1)
            [plot_and_pause(a) for a in ["bwr", "seismic"]]

        # Dynamic mode must be toggled off to call show.
        show and not dynamic and plt.show()

        # Dynamic mode must be on, show must be off to pause.
        not show and dynamic and plt.pause(pause)

        return 0
