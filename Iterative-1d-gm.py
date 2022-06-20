#!/usr/bin/env python
import math
import sys
import numpy as np
from scipy.ndimage import gaussian_filter1d


def main(argv):
    # get signals to numpy array
    # NOTE: assuming names are 'x1', 'x2'
    x1 = gaussian_filter1d(np.load(argv[0])['x1'], sigma=1)
    x2 = gaussian_filter1d(np.load(argv[1])['x2'], sigma=1)

    # match signals length (just for generated example, generally need to be same length)
    if len(x1) > len(x2):
        x1 = x1[:len(x2)]
    else:
        x2 = x2[:len(x1)]

    # start iterative algorithm
    dx = 0
    threshold = 0.0001
    dx_update = threshold
    while np.abs(dx_update) >= 0.0001:
        n = len(x1) - math.ceil(np.abs(dx))  # overlap area size

        if dx < 0:
            # shift x2 right
            dx_abs = np.abs(dx)
            dx_frac = dx_abs - math.floor(dx_abs)
            x2s = x2[math.floor(dx_abs):]  # integer shift
            x2s = np.array([(x2s[i] + (x2s[i + 1] - x2s[i]) * dx_frac) for i in range(n - 1)])  # fraction shift
            x1c = x1[0:-math.floor(dx_abs) - 1]  # cut unnecessary part of x1 (keep just the overlap area)
        else:  # dx >= 0
            # shift x2 left
            dx_frac = dx - math.floor(dx)
            x2s = x2[:-math.floor(dx)] if math.floor(dx) > 0 else x2  # integer shift
            x2s = np.array([(x2s[i] - (x2s[i] - x2s[i - 1]) * dx_frac) for i in range(1, n)])  # fraction shift
            x1c = x1[math.floor(dx) + 1:]  # cut unnecessary part of x1 (keep just the overlap area)

        # build and solve linear equation
        a = np.array([(x1c[i + 1] - x1c[i - 1]) for i in range(1, n - 2)]).reshape((n - 3, 1))  # derivative
        b = np.array([(x2s[i] - x1c[i]) for i in range(1, n - 2)]).reshape((n - 3, 1))  # difference
        dx_update = np.linalg.solve(a.T @ a, a.T @ b)[0][0]  # least square
        dx += dx_update

    print('dx = %.2f' % dx)


if __name__ == '__main__':
    main(sys.argv[1:])
