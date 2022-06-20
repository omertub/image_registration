#!/usr/bin/env python
import math
import sys
import numpy as np
from scipy.ndimage import gaussian_filter


# this function is from Iterative-2d-gm.py
# this function shift the images and cut them to the overlap area
def interpolation_2d(x1, x2, dx, dy):
    dx_abs = np.abs(dx)
    dy_abs = np.abs(dy)

    # integer shift (x1 or x2)
    x1si = x1.copy()
    x2si = x2.copy()
    # x direction
    if dx < 0:  # shift x2 left
        x2si = x2si[math.floor(dx_abs):, :]  # shift left
        x1si = x1si[:-math.floor(dx_abs), :] if math.floor(dx_abs) > 0 else x1si  # cut to the overlap area
        dir_x = 1  # for the interpolation
    else:  # shift x1 left
        x1si = x1si[math.floor(dx):, :]  # shift left
        x2si = x2si[:-math.floor(dx_abs), :] if math.floor(dx_abs) > 0 else x2si  # cut to the overlap area
        dir_x = -1  # for the interpolation
    # y direction
    if dy < 0:  # shift x2 up
        x2si = x2si[:, math.floor(dy_abs):]  # shift up
        x1si = x1si[:, :-math.floor(dy_abs)] if math.floor(dy_abs) > 0 else x1si  # cut to the overlap area
        dir_y = 1  # for the interpolation
    else:  # shift x1 up
        x1si = x1si[:, math.floor(dy):]  # shift up
        x2si = x2si[:, :-math.floor(dy_abs)] if math.floor(dy_abs) > 0 else x2si  # cut to the overlap area
        dir_y = -1  # for the interpolation

    # shift x2 by fraction
    x2s = x2si.copy()
    alpha = dx_abs - math.floor(dx_abs)  # fraction of dx
    beta = dy_abs - math.floor(dy_abs)  # fraction of dy
    # interpolation equation from class
    for i in range(1, x1si.shape[0] - 1):
        for j in range(1, x1si.shape[1] - 1):
            x2s[i, j] = x2si[i, j] * (1 - alpha) * (1 - beta) + x2si[i + dir_x, j] * alpha * (1 - beta) + \
                        x2si[i, j + dir_y] * (1 - alpha) * beta + x2si[i + dir_x, j + dir_y] * alpha * beta
    x2s = x2s[1:-1, 1:-1]  # cut out the frame (the equation doesn't valid there)
    x1si = x1si[1:-1, 1:-1]  # cut to the same shape

    return x1si, x2s


# this function is a copy of Iterative-1d-gm.py::main()
# BUT: get as parameter initial dx, dy
def iterative2d(x1, x2, dx, dy):
    # TODO: dx, dy = 0, 0
    threshold = 0.0002
    dx_update, dy_update = threshold, threshold
    while np.abs(dx_update) + np.abs(dy_update) >= threshold:

        # shift images and keep only overlap area
        x1o, x2o = interpolation_2d(x1, x2, dx, dy)  # 'o' for 'overlap'

        # build linear equation
        a, b = np.zeros((1, 2)), np.zeros((1, 1))  # initiate first dummy row to append to it
        for i in range(1, x1o.shape[0] - 1):
            for j in range(1, x1o.shape[1] - 1):
                # calculations
                der_x = int(x1o[i + 1, j]) - int(x1o[i - 1, j])
                der_y = int(x1o[i, j + 1]) - int(x1o[i, j - 1])
                b_val = int(x2o[i, j]) - int(x1o[i, j])
                # add row (=equation)
                a = np.append(a, [[der_x, der_y]], axis=0)
                b = np.append(b, [[b_val]], axis=0)
        a, b = a[1:, :], b[1:, :]  # remove dummy first row

        # solve linear equation and update dx, dy
        sol = np.linalg.solve(a.T @ a, a.T @ b)  # least square
        dx_update, dy_update = sol[0, 0], sol[1, 0]
        dx += dx_update
        dy += dy_update

    return dx, dy


def main(argv):
    # get signals to numpy array
    # NOTE: assuming names are 'x1', 'x2'
    x1 = np.load(argv[0])['x1']
    x2 = np.load(argv[1])['x2']

    # make sure its grayscale
    if len(x1.shape) == 3:
        x1 = x1[:, :, 0]
    if len(x2.shape) == 3:
        x2 = x2[:, :, 0]

    # match images shapes
    # NOTE: just for generated example, generally need to be same shapes
    if x1.shape[0] > x2.shape[0]:
        x1 = x1[:x2.shape[0], :]
    elif x1.shape[0] < x2.shape[0]:
        x2 = x2[:x1.shape[0], :]

    if x1.shape[1] > x2.shape[1]:
        x1 = x1[:, :x2.shape[1]]
    elif x1.shape[1] < x2.shape[1]:
        x2 = x2[:, :x1.shape[1]]

    # smooth with gaussian
    x1 = gaussian_filter(x1, sigma=1)
    x2 = gaussian_filter(x2, sigma=1)

    # decimate to 1/4 resolution and get initial (dx4, dy4)
    dx, dy = iterative2d(gaussian_filter(x1, sigma=0.5)[::4, ::4], gaussian_filter(x2, sigma=0.5)[::4, ::4], 0, 0)
    # decimate to 1/2 resolution, start from (2*dx4, 2*dy4) and get (dx2, dy2)
    dx, dy = iterative2d(gaussian_filter(x1, sigma=0.5)[::2, ::2], gaussian_filter(x2, sigma=0.5)[::2, ::2], dx * 2, dy * 2)
    # original resolution, start from (2*dx2, 2*dy2) and get (dx, dy)
    dx, dy = iterative2d(x1, x2, dx * 2, dy * 2)

    print("dx = %.2f" % dx, ", dy = %.2f" % dy)


if __name__ == '__main__':
    main(sys.argv[1:])
