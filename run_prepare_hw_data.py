import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import gaussian_filter

# create 1D signal
x = np.random.uniform(0, 1, 10000)
x = np.convolve(x, np.ones(10) / 10, mode='same')

Downsample = 2
Delta = 1
x1 = x[0::Downsample]
x2 = x[Delta::Downsample]
dx = Delta / Downsample

plt.subplot(131)
plt.plot(x)
plt.subplot(132)
plt.plot(x1)
plt.subplot(133)
plt.plot(x2)
plt.show()

print('dx = ' + repr(dx))

# np.savez_compressed('data1dx1.npz', x=x, Downsample=Downsample, Delta=Delta, x1=x1)
np.savez_compressed('data1dx1.npz', x1=x1)
np.savez_compressed('data1dx2.npz', x2=x2)

# create 2D signal
im = cv2.imread('peppers.bmp')
cv2.imshow('image', im)
cv2.waitKey(0)

SmoothdImg = gaussian_filter(im, sigma=3)
cv2.imshow('smoothed', SmoothdImg)
cv2.waitKey(0)

Downsample = 2
DeltaX = 3
DeltaY = 3
im1 = SmoothdImg[0::Downsample, 0::Downsample]
im2 = SmoothdImg[DeltaY::Downsample, DeltaX::Downsample]
dx = DeltaX / Downsample
dy = DeltaY / Downsample

cv2.imwrite('im1.bmp', im1)
cv2.imwrite('im2.bmp', im2)

np.savez_compressed('data2d.npz', dx=dx, dy=dy, Downsample=Downsample,
                    im1=im1, im2=im2)
