import cv2
import matplotlib.pyplot as plt
import numpy as np
import pdb

def conformal_map(w):
    coords = np.power(w, (2 * np.pi * 1j + np.log(4)) / (2 * np.pi * 1j))
    coords = coords / np.max(np.abs(coords))
    return coords

def sample_pixels(original_img, new_coords):
    mid_x = original_img.shape[0] / 2
    mid_y = original_img.shape[1] / 2

    new_img = np.zeros_like(original_img)
    new_x = np.rint(mid_x + new_coords.real * new_coords.shape[0] / 2).astype(int)
    new_y = np.rint(mid_y + new_coords.imag * new_coords.shape[1] / 2).astype(int)

    for i in range(new_img.shape[0]):
        for j in range(new_img.shape[1]):
            new_img[i, j] = original_img[new_x[i, j], new_y[i, j]]

    return new_img

img = cv2.imread("tess2.png", 0)
# img = cv2.imread("gallery.jpg", 0)

xstep = np.linspace(-1, 1, img.shape[0])
ystep = np.linspace(-1, 1, img.shape[1])
complex_grid = xstep[:, None] + 1j * ystep

transformed_coords = conformal_map(complex_grid)
new_img = sample_pixels(img, transformed_coords)

fig, axs = plt.subplots(1, 2)
axs[0].imshow(img, cmap="gray")
axs[1].imshow(new_img, cmap="gray")
plt.tight_layout()
plt.show()

# plt.imshow(np.abs(transformed_coords))
# plt.show()