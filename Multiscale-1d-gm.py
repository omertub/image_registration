#!/usr/bin/env python
import math
import sys
import numpy as np
from scipy.ndimage import gaussian_filter1d


# this function is a copy of Iterative-1d-gm.py
# BUT: get as parameter initial dx
def iterative1d(x1, x2, dx):
    # match signals length
    if len(x1) > len(x2):
        x1 = x1[:len(x2)]
    else:
        x2 = x2[:len(x1)]

    # update iteratively
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

    return dx


def main(argv):
    # get signals to numpy array
    # NOTE: assuming names are 'x1', 'x2'
    x1 = gaussian_filter1d(np.load(argv[0])['x1'], sigma=1)
    x2 = gaussian_filter1d(np.load(argv[1])['x2'], sigma=1)

    # decimate to 1/4 resolution and get initial dx4
    dx = iterative1d(gaussian_filter1d(x1, sigma=1)[::4], gaussian_filter1d(x2, sigma=1)[::4], 0)
    # decimate to 1/2 resolution, start from 2*dx4 and get dx2
    dx = iterative1d(gaussian_filter1d(x1, sigma=1)[::2], gaussian_filter1d(x2, sigma=1)[::2], dx * 2)
    # original resolution, start from 2*dx2 and get dx
    dx = iterative1d(x1, x2, dx * 2)

    print('dx = %.2f' % dx)


if __name__ == '__main__':
    main(sys.argv[1:])
