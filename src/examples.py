import cv2
import numpy as np

from transforms import blend_zeros, change_coords, get_coord_grid


img = cv2.imread('../images/grid.png')
z = get_coord_grid(img)

zt = z * (np.cos(np.pi / 4) + 1j * np.sin(np.pi / 4))
# zt = np.exp(z)

new_img = change_coords(img, zt)

cv2.imshow('image', new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
