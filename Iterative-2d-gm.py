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

    # check for grayscale
    if len(x1.shape) > 2:
        x1 = x1[:, :, 0]
    if len(x2.shape) > 2:
        x2 = x2[:, :, 0]

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

        # TODO: no need for min
        nx = min(x1.shape[0], x2.shape[0]) - math.ceil(np.abs(dx))  # overlap area size axis x
        ny = min(x1.shape[1], x2.shape[1]) - math.ceil(np.abs(dy))  # overlap area size axis y

        # Bilinear interpolation
        # TODO: for now dx,dy > 0
        dx_frac = dx - math.floor(dx)
        dy_frac = dy - math.floor(dy)
        end = lambda x, n: -n if n > 0 else len(x)
        x2si = x2[:end(x2, math.floor(dx)), :end(x2, math.floor(dy))]  # integer part of the shift
        x2s = np.zeros((x2si.shape[0], x2si.shape[1]))
        for i in range(1, nx):
            for j in range(1, ny):
                x2s[i, j] = x2si[i, j] * (1 - dx_frac) * (1 - dy_frac) + x2si[i, j - 1] * dx_frac * (1 - dy_frac) + \
                            x2si[i - 1, j] * (1 - dx_frac) * dy_frac + x2si[i - 1, j - 1] * dx_frac * dy_frac
        x2s = x2s[1:, 1:]
        x1c = x1[math.floor(dx) + 1:, math.ceil(dy) + 1:]

        # build and solve linear equation
        # TODO: maybe they are not in same size, problem?
        nx = min(x1c.shape[0], x2s.shape[0])  # overlap area size axis x
        ny = min(x1c.shape[1], x2s.shape[1])  # overlap area size axis y

        a_x, a_y, b = np.array([]), np.array([]), np.array([])
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                a_x = np.append(a_x, int(x1c[i + 1, j]) - int(x1c[i - 1, j]))
                a_y = np.append(a_y, int(x1c[i, j + 1]) - int(x1c[i, j - 1]))
                b = np.append(b, int(x2s[i, j]) - int(x1c[i, j]))
        a_x = a_x.reshape((-1, 1))
        a_y = a_y.reshape((-1, 1))
        a = np.concatenate((a_x, a_y), 1)
        b = b.reshape((-1, 1))

        print(dx, dy)

        sol = np.linalg.solve(a.T @ a, a.T @ b)
        dx_update, dy_update = sol[0, 0], sol[1, 0]
        dx += dx_update
        dy += dy_update
        # print(dx, dx_update)

    print("dx = %.2f" % dx, ", dy = %.2f" % dy)


if __name__ == '__main__':
    main(sys.argv[1:])
