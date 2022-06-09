#!/usr/bin/env python
import math
import sys
import numpy as np
from scipy.ndimage import gaussian_filter

def match_shapes(x1, x2):
    # x
    if x1.shape[0] > x2.shape[0]:
        x1 = x1[:x2.shape[0], :]
    else:
        x2 = x2[:x1.shape[0], :]

    # y
    if x1.shape[1] > x2.shape[1]:
        x1 = x1[:, :x2.shape[1]]
    else:
        x2 = x2[:, :x1.shape[1]]

    return x1, x2


def interpolation_2d(x1, x2, dx, dy):
    nx = min(x1.shape[0], x2.shape[0]) - math.ceil(np.abs(dx))  # overlap area size axis x
    ny = min(x1.shape[1], x2.shape[1]) - math.ceil(np.abs(dy))  # overlap area size axis y

    dx_abs = np.abs(dx)
    dy_abs = np.abs(dy)

    # shift x1/x2 left by integer
    x1si = x1.copy()
    x2si = x2.copy()

    ## x direction - if dx>0 we need to shift x2, otherwise we shift x1
    if dx > 0:
        x2si = x2si[:-math.floor(dx), :] if math.floor(dx) > 0 else x2si
        dir_x = -1
    else:
        x1si = x1si[math.floor(dx_abs):, :]
        dir_x = 1

    ## y direction - if dy>0 we need to shift up
    if dy > 0:
        x2si = x2si[:, :-math.floor(dy)]  if math.floor(dy) > 0 else x2si
        dir_y = -1
    else:
        x1si = x1si[:, math.floor(dy_abs):]
        dir_y = 1

    # shift x2 by fraction
    x2s = x2si.copy()
    alpha = dx_abs - math.floor(dx_abs) # fraction of dx
    beta = dy_abs - math.floor(dy_abs)  # fraction of dy
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            x2s[i, j] = x2si[i, j] * (1 - alpha) * (1 - beta) + x2si[i, j + dir_y] * alpha * (1 - beta) + \
                        x2si[i + dir_x, j] * (1 - alpha) * beta + x2si[i + dir_x, j + dir_y] * alpha * beta
    x2s = x2s[1:-1, 1:-1]

    return match_shapes(x1si, x2si)

def main(argv):
    # get signals to numpy array
    x1 = np.load(argv[0])['x1']
    x2 = np.load(argv[1])['x2']

    # check for grayscale
    if len(x1.shape) == 3:
        x1 = x1[:, :, 0]
    if len(x2.shape) == 3:
        x2 = x2[:, :, 0]

    # match images shapes
    if x1.shape[0] > x2.shape[0]:
        x1 = x1[:x2.shape[0], :x2.shape[0]]
    else:
        x2 = x2[:x1.shape[0], :x1.shape[0]]

    # apply gaussian
    x1 = gaussian_filter(x1, sigma=5)
    x2 = gaussian_filter(x2, sigma=5)

    # update iteratively
    dx, dy = 0, 0
    threshold = 0.000001
    dx_update, dy_update = threshold, threshold
    while np.abs(dx_update) >=threshold or np.abs(dy_update) >=threshold:

        x1c, x2s = interpolation_2d(x1, x2, dx, dy)
        # print(x1c.shape, x2s.shape)

        # build and solve linear equation
        nx = min(x1.shape[0], x2.shape[0]) - math.ceil(np.abs(dx))  # overlap area size axis x
        ny = min(x1.shape[1], x2.shape[1]) - math.ceil(np.abs(dy))  # overlap area size axis y

        a, b = np.ndarray((1,2)), np.ndarray((1,1))
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                der_x = int(x1c[i + 1, j]) - int(x1c[i - 1, j])
                der_y = int(x1c[i, j + 1]) - int(x1c[i, j - 1])
                if i==1 and j==1:
                    a[0,0], a[0,1] = der_x, der_y
                    b[0] = int(x2s[i, j]) - int(x1c[i, j])
                else:
                    a = np.append(a, [[der_x, der_y]], axis = 0)
                    b = np.append(b, int(x2s[i, j]) - int(x1c[i, j]))
        b = b.reshape((-1, 1))

        sol = np.linalg.solve(a.T @ a, a.T @ b)
        dx_update, dy_update = sol[0, 0], sol[1, 0]
        dx += dx_update
        dy += dy_update

        print(dx, dy)


    print("dx = %.2f" % dx, ", dy = %.2f" % dy)


if __name__ == '__main__':
    main(sys.argv[1:])
