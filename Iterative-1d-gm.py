#!/usr/bin/env python
import math
import sys
import numpy as np
from scipy.ndimage import gaussian_filter1d


def main(argv):
    # get signals to numpy array
    x1 = gaussian_filter1d(np.load(argv[0])['x1'], sigma=1)
    x2 = gaussian_filter1d(np.load(argv[1])['x2'], sigma=1)

    # match signals length
    if len(x1) > len(x2):
        x2 = x2[:len(x1)]
    else:
        x1 = x1[:len(x2)]

    # update iteratively
    dx = 0
    threshold = 0.0001
    dx_update = threshold
    while np.abs(dx_update) >= 0.0001:
        # in case of negative shift - swap the signals
        # we always move x1 to the right
        if dx_update < 0:
            x1, x2 = x2, x1
            dx = np.abs(dx)
        n = min(len(x2), len(x1)) - math.ceil(dx)  # overlap area size

        # shift x1 with taylor
        dx_frac = dx - math.floor(dx)
        x1s = x1[math.floor(dx):]  # integer part of the shift
        x1s = np.array([(x1s[i] + (x1s[i + 1] - x1s[i]) * dx_frac) for i in range(n - 1)])  # fraction part of the shift

        # build and solve linear equation
        a = np.array([(x1s[i + 1] - x1s[i - 1]) for i in range(1, n - 2)]).reshape((n - 3, 1))
        b = np.array([(x2[i] - x1s[i]) for i in range(1, n - 2)]).reshape((n - 3, 1))
        dx_update = np.linalg.solve(a.T @ a, a.T @ b)[0][0]
        dx += dx_update

    print("dx = %.2f" % dx)


if __name__ == '__main__':
    main(sys.argv[1:])
