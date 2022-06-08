#!/usr/bin/env python
import math
import sys
import numpy as np
from scipy.ndimage import gaussian_filter


def gaussian_filter2d(param, sigma):
    pass


def main(argv):
    # get signals to numpy array
    x1 = gaussian_filter(np.load(argv[0])['x1'], sigma=1)
    x2 = gaussian_filter(np.load(argv[1])['x2'], sigma=1)

    # match images shapes
    if x1.shape[0] > x2.shape[0]:
        x1 = x1[:x2.shape[0], :x2.shape[0]]
    else:
        x2 = x2[:x1.shape[0], :x1.shape[0]]

    # update iteratively
    dx, dy = 0, 0
    threshold = 0.0001
    dx_update, dy_update = threshold, threshold
    while np.abs(dx_update) >= 0.0001 or np.abs(dy_update) >= 0.0001:

        nx = min(x1.shape[0], x2.shape[0]) - math.ceil(np.abs(dx))  # overlap area size axis x
        ny = min(x1.shape[1], x2.shape[1]) - math.ceil(np.abs(dy))  # overlap area size axis y

        if dx < 0:
            # shift x2 right
            dx_abs = np.abs(dx)
            dx_frac = dx_abs - math.floor(dx_abs)
            x2s = x2[math.floor(dx_abs):]                                                      # integer part of the shift
            x2s = np.array([(x2s[i] + (x2s[i + 1] - x2s[i]) * dx_frac) for i in range(n - 1)]) # fraction part of the shift
            x1c = x1[0:-math.floor(dx_abs) - 1]
        else:
            # shift x2 left
            dx_frac = dx - math.floor(dx)                                                      # should be negative
            x2s = x2[:-math.floor(dx)] if math.floor(dx) > 0 else x2                           # integer part of the shift
            x2s = np.array([(x2s[i] - (x2s[i] - x2s[i - 1]) * dx_frac) for i in range(1, n)])  # fraction part of the shift
            x1c = x1[math.floor(dx) + 1:]

        # build and solve linear equation

        a_x, a_y = np.array([]), np.array([])
        for y in range(ny):
            a_x.append(np.array([(x1c[i + 1, y] - x1c[i - 1, y]) for i in range(1, n - 2)]).reshape((n - 3, 1)))

        a = np.array([(x1c[i + 1] - x1c[i - 1]) for i in range(1, n - 2)]).reshape((n - 3, 1))
        b = np.array([(x2s[i] - x1c[i]) for i in range(1, n - 2)]).reshape((n - 3, 1))
        dx_update = np.linalg.solve(a.T @ a, a.T @ b)[0][0]
        dx += dx_update
        # print(dx, dx_update)

    print("dx = %.2f" % dx, ", dy = %.2f" % dy)


if __name__ == '__main__':
    main(sys.argv[1:])
