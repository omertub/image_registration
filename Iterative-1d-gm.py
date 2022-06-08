#!/usr/bin/env python
import math
import sys
import numpy as np
import scipy
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt


def plot_smooth(x1g, x2g):
    plt.figure(1)
    plt.subplot(211)
    plt.plot(x1g, 'k', label='x1 smooth')
    plt.subplot(212)
    plt.plot(x2g, 'k', label='x2 smooth')
    plt.show()


def main(argv):
    # get signals to numpy array
    x1 = gaussian_filter1d(np.load(argv[0])['x1'], sigma=1)
    x2 = gaussian_filter1d(np.load(argv[1])['x2'], sigma=1)

    # plot_smooth(x1, x2)

    # match signals length
    if len(x1) > len(x2):
        x2 = x2[:len(x1)]
    else:
        x1 = x1[:len(x2)]

    # loop
    dx = 0
    dx_update = 1
    # TODO: for now we assume dx is positive
    while np.abs(dx_update) > 0.0001:
        if dx_update < 0:
            x1, x2 = x2, x1
            dx_update = np.abs(dx_update)
            dx = np.abs(dx)
        # build and solve linear equation
        n = min(len(x2), len(x1)) - math.ceil(dx)
        # shift x1
        dx_frac = dx - math.floor(dx)
        x1s = x1[math.floor(dx):]
        x1s = np.array([(x1s[i] + (x1s[i + 1] - x1s[i]) * dx_frac) for i in range(n - 1)])
    
        a = np.array([(x1s[i + 1] - x1s[i - 1]) for i in range(1, n - 2)]).reshape((n - 3, 1))  # /2 removed
        b = np.array([(x2[i] - x1s[i]) for i in range(1, n - 2)]).reshape((n - 3, 1))
        dx_update = np.linalg.solve(a.T @ a, a.T @ b)[0][0]  # solve manualy
        dx += dx_update
        print(dx, '\t', dx_update)



if __name__ == '__main__':
    main(sys.argv[1:])
