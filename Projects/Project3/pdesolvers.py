import numpy as np
import sys
import matplotlib.pyplot as plt


class PDEsolvers:
    def __init__(self, bound_cond1=0, bound_cond2=0):
        self.bound1 = bound_cond1
        self.bound2 = bound_cond2

    def initial_condition(self, x):
        self.x = x
        self.init = np.sin(np.pi * x)
        self.init[0] = self.bound1
        self.init[-1] = self.bound2

    def ForwardEuler(self, N, T, dt, dx, plot=False):

        self.u = np.zeros((T, N))
        self.u[0, :] = self.init

        ddx = dx**2
        D = dt / ddx

        fig = plt.figure(111)
        ax = fig.add_subplot(111)

        for i in range(0, T - 1):

            sys.stdout.write("\rCompletion: %d %%" % (100 * (i + 1) / T))
            sys.stdout.flush()

            self.u[i + 1, 1:-1] = self.u[i, 1:-1] + \
                D * (self.u[i, 2:] - (2 * self.u[i, 1:-1]) + self.u[i, :-2])

            self.u[i + 1, 0] = self.bound1
            self.u[i + 1, -1] = self.bound2

            ax.clear()
            ax.set_ylim(-1.5, 1.5)
            ax.plot(self.x, self.u[i + 1, :], 'r')
            ax.set_title("step %s" % (i))
            plt.pause(0.0001)

        print()
        return self.u

    # def analytic()
