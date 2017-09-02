""""
Filename:    visualize.py

Description: This file contains complete visualizations for only the 2D perceptron. 
"""""

from benchmark import Benchmark
from point import Point

from matplotlib import pyplot as plt
import numpy as np

class Visualize(object):
    """ Methods: plot_2d = Produce a plot of the given line and points.
                 plot
    """

    @staticmethod
    def plot_2d(d, w, w_star=None, scale=1, show=False, dynamic=False, pause=0.01):
        """ Produce a static static plot given the data-set "d" and the weight vector "w". If 
        defined, draw the decision boundary line from "w_star" and adjust the axis according to 
        "scale". If desired, the call to "plt.show() and plt.ion()" can be omitted.
        
        :param d: Data-set containing list of Point instances.
        :param w: Weight vector to derive line from.
        :param w_star: Weight vector to derive decision boundary from.
        :param scale: The axis to display.
        :param show: If false, do not call "plt.show()".
        :param dynamic: If false, do not call "plt.ion()".
        :param pause: Number of seconds to pause plot for.
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
        pl_x = np.arange(-2  * scale, 2 * scale)
        pl_y = pl_x * -w[1]/w[2] - w[0]/w[2]
        plt.plot(pl_x, pl_y, c="m")

        # If defined, plot the decision boundary.
        if w_star is not None:
            db_x = np.arange(-2  * scale, 2 * scale)
            db_y = db_x * -w_star[1]/w_star[2] - w_star[0]/w_star[2]
            plt.plot(db_x, db_y, "--", c="k")

            # Shade in the area classified by w (not w_star).
            plt.fill_between(pl_x, -np.ones(len(pl_x)) * scale, pl_y, facecolor="b", alpha=0.1)
            plt.fill_between(pl_x, np.ones(len(pl_x)) * scale,  pl_y, facecolor="r", alpha=0.1)

        # Dynamic mode must be toggled off to call show.
        show and not dynamic and plt.show()

        # Dynamic mode must be on, show must be off to pause.
        not show and dynamic and plt.pause(pause)

        return 0

