#!/usr/bin/env python
import math
import sys
import numpy as np
import scipy
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt


def plot_smooth(x1, x2, x1g, x2g):
    plt.figure(1)
    plt.subplot(221)
    plt.plot(x1)
    plt.subplot(222)
    plt.plot(x2, 'k', label='original x2')
    plt.subplot(223)
    plt.plot(x1g, 'k', label='x1 smooth')
    plt.subplot(224)
    plt.plot(x2g, 'k', label='x2 smooth')
    plt.show()


def main(argv):
    # get signals to numpy array
    x1 = gaussian_filter1d(np.load(argv[0])['x1'], sigma=1)
    x2 = gaussian_filter1d(np.load(argv[1])['x2'], sigma=1)

    # match signals length
    if len(x1) > len(x2):
        x2 = x2[:len(x1)]
    else:
        x1 = x1[:len(x2)]

    # loop
    dx = 0
    dx_update = 1
    # TODO: for now we assume dx is positive
    # while dx_update > 0.01:
    #     # build and solve linear equation
    #     n = len(x1) - math.ceil(dx)
    #     # shift x2
    #     dx_frac = dx - math.floor(dx)
    #     x2s = x2[math.floor(dx):]  # move right arik used ndimage.shift
    #     x2s = np.array([(x2s[i] + (x2s[i + 1] - x2s[i]) * dx_frac) for i in range(n - 1)])
    #
    #     a = np.array([(x1[i + 1] - x1[i - 1]) for i in range(1, n - 2)]).reshape((n - 3, 1))  # /2 removed
    #     b = np.array([(x2s[i] - x1[i]) for i in range(1, n - 2)]).reshape((n - 3, 1))
    #     dx_update = np.linalg.solve(a.T @ a, a.T @ b)[0][0]  # solve manualy
    #     dx += dx_update
    #     print(dx, '\t', dx_update)

    while abs(dx_update) > 0.01:
        # build and solve linear equation
        n = len(x2) - math.ceil(dx)
        # shift x1
        dx_frac = dx - math.floor(dx)
        x1s = x1[math.floor(dx):]  # move right arik used ndimage.shift
        x1s = np.array([(x1s[i] + (x1s[i + 1] - x1s[i]) * (1-dx_frac)) for i in range(n - 1)])

        a = np.array([(x2[i + 1] - x2[i - 1]) for i in range(1, n - 2)]).reshape((n - 3, 1))  # /2 removed
        b = np.array([(x1s[i] - x2[i]) for i in range(1, n - 2)]).reshape((n - 3, 1))
        dx_update = np.linalg.solve(a.T @ a, a.T @ b)[0][0]
        dx += dx_update
        print(dx, '\t', dx_update)


if __name__ == '__main__':
    main(sys.argv[1:])
