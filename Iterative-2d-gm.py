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

        # Bilinear interpolation
        dx_frac = dx - math.floor(dx)
        dy_frac = dy - math.floor(dy)
        x2s = x2[math.floor(dx):,:math.floor(dy)]                                           # integer part of the shift
        for i in range(1, nx - 1):
           for j in range(1, ny - 1):
                x2s = np.array([(x2s[i,j] + (x2s[i + 1] - x2s[i]) * dx_frac) for i in range(nx - 1)]) # fraction part of the shift
                x1c = x1[0:-math.floor(dx_abs) - 1]


        # build and solve linear equation

        # TODO: maybe they are not in same size
        nx = min(x1c.shape[0], x2s.shape[0])  # overlap area size axis x
        ny = min(x1c.shape[1], x2s.shape[1])  # overlap area size axis y

        a_x, a_y, b = np.array([]), np.array([]), np.array([])
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                a_x.append(x1c[i + 1, j] - x1c[i - 1, j])
                a_y.append(x1c[i, j + 1] - x1c[i, j - 1])
                b.append((x2s[i,j] - x1c[i,j]))

        a_x = a_x.reshape((nx-3,1,3))
        a_y = a_y.reshape((ny-3,1,3))
        a = np.concatenate((a_x,a_y), 1)
        
        sol = np.linalg.solve(a.T @ a, a.T @ b)[0]
        dx_update, dy_update = sol[0], sol[1]
        dx += dx_update
        dy += dy_update
        # print(dx, dx_update)

    print("dx = %.2f" % dx, ", dy = %.2f" % dy)


if __name__ == '__main__':
    main(sys.argv[1:])
