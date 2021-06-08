import cv2
import numpy as np

from transforms import blend_zeros, change_coords

img = cv2.imread('grid.png')

x = np.linspace(-1, 1, img.shape[1])
y = np.linspace(-1, 1, img.shape[0])
xx, yy = np.meshgrid(x, y)
z = xx + yy * 1j

zt = np.exp(z)

new_img = change_coords(img, zt)
new_img = blend_zeros(new_img)

cv2.imwrite('grid_new.png', new_img)
