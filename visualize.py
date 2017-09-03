""""
Filename:    visualize.py

Description: This file contains complete visualizations for only the 2D perceptron. 
"""""

from pandas.plotting import parallel_coordinates
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

from point import Point


class Visualize(object):
    """ Methods: plot_2d = Produce a plot of the given line and points.
                 plot = Produce a matrix of plots given the line and points. 
    """

    @staticmethod
    def plot_2d(d, w, w_star=None, focus=None, scale=1, show=False, dynamic=False, pause=0.01):
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

            # Shade in the area classified by w (not w_star).
            plt.fill_between(pl_x, -np.ones(len(pl_x)) * scale, pl_y, facecolor="m", alpha=0.1)
            # plt.fill_between(pl_x, np.ones(len(pl_x)) * scale, pl_y, facecolor="r", alpha=0.1)

        # Dynamic mode must be toggled off to call show.
        show and not dynamic and plt.show()

        # Dynamic mode must be on, show must be off to pause.
        not show and dynamic and plt.pause(pause)

        return 0

    @staticmethod
    def plot_parallel(d, w, w_star=None, focus=None, scale=1, show=False, dynamic=False,
                      pause=0.01):
        """ Produce a parallel coordinate given the data-set "d" and the weight vector "w". 
        
        # If
        # defined, draw the decision boundary line from "w_star" and adjust the axis according to
        # "scale". If desired, the call to "plt.show() and plt.ion()" can be omitted.

        :param d: Data-set containing list of Point instances.
        :param w: Weight vector to derive line from.
        :param w_star: Weight vector to derive decision boundary from.
        :param focus: Star to focus. Represents the current star selected.
        :param scale: The axis to display.
        :param show: If false, do not call "plt.show()".
        :param dynamic: If false, do not call "plt.ion()".
        :param pause: Number of seconds to pause plot for.
        :return: None.
        """
        degree = len(d[0].x)

        # Enable interactive mode if desired. Clear the previous plot.
        dynamic and plt.subplot(111) and plt.cla() and plt.ion()

        parallel_coordinates(pd.DataFrame([np.append(a.ell, a.x) for a in d]), 0)

        # # For every dimension, we plot against another.
        # for m in range(degree):
        #     for n in range(degree):
        #         plt.subplot(degree, degree, degree * m + n + 1)
        #
        #         # We are comparing against the same dimension. Display a histogram instead.
        #         if m == n:
        #             plt.hist([a.x[m] for a in d])
        #
        #         else:
        #             # Otherwise, we set dimension 1 and 2 of this plot as m and n here.
        #             d_mn = [Point([a.x[m], a.x[n]], a.ell) for a in d]
        #
        #             # Keep our bias. Represent the weights with respect to m and n.
        #             w_mn = [w[0], w[m + 1], w[n + 1]]
        #             w_star_mn = [w_star[0], w_star[m + 1], w_star[n + 1]]
        #
        #             # Given a focus, define one with respect to m and n.
        #             focus_mn = None if focus is None else Point([focus.x[m], focus.x[n]])
        #
        #             Visualize.plot_2d(d_mn, w_mn, w_star_mn, focus_mn, scale, False, False, 0.0)
        #
        # # This is probably going to be tight!
        # plt.tight_layout()

        # Dynamic mode must be toggled off to call show.
        show and not dynamic and plt.show()

        # Dynamic mode must be on, show must be off to pause.
        not show and dynamic and plt.pause(pause)

    @staticmethod
    def plot(d, w, w_star=None, focus=None, scale=1, show=False, dynamic=False, pause=0.01):
        """ Produce a matrix of plots given the data-set "d" and the weight vector "w". If
        defined, draw the decision boundary line from "w_star" and adjust the axis according to
        "scale". If desired, the call to "plt.show() and plt.ion()" can be omitted.

        :param d: Data-set containing list of Point instances.
        :param w: Weight vector to derive line from.
        :param w_star: Weight vector to derive decision boundary from.
        :param focus: Star to focus. Represents the current star selected.
        :param scale: The axis to display.
        :param show: If false, do not call "plt.show()".
        :param dynamic: If false, do not call "plt.ion()".
        :param pause: Number of seconds to pause plot for.
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
                    d_mn = [Point([a.x[m], a.x[n]], a.ell) for a in d]

                    # Keep our bias. Represent the weights with respect to m and n.
                    w_mn = [w[0], w[m + 1], w[n + 1]]
                    w_star_mn = [w_star[0], w_star[m + 1], w_star[n + 1]]

                    # Given a focus, define one with respect to m and n.
                    focus_mn = None if focus is None else Point([focus.x[m], focus.x[n]])

                    Visualize.plot_2d(d_mn, w_mn, w_star_mn, focus_mn, scale, False, False, 0.0)

        # This is probably going to be tight!
        plt.tight_layout()

        # Dynamic mode must be toggled off to call show.
        show and not dynamic and plt.show()

        # Dynamic mode must be on, show must be off to pause.
        not show and dynamic and plt.pause(pause)
