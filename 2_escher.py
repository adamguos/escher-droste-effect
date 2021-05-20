import cv2
import matplotlib.pyplot as plt
import numpy as np
import pdb

def conformal_map(z, a):
    # conformal = np.zeros_like(z)
    # for i in range(z.shape[0]):
    #     for j in range(z.shape[1]):
    #         if z[i, j].real >= 0:
    #             conformal[i, j] = np.log(z[i, j] + a)
    #         else:
    #             conformal[i, j] = 2 * np.log(a) - np.log(-z[i, j] + a)
    # return conformal
    conformal = np.log(z + a)
    # return conformal / np.abs(conformal).max()
    return conformal

def sample_pixels(original_img, new_coords):
    mid_x = original_img.shape[0] / 2
    mid_y = original_img.shape[1] / 2

    new_img = np.zeros_like(original_img)
    # new_x = np.rint(mid_x + new_coords.real * new_coords.shape[0] / 2).astype(int)
    # new_y = np.rint(mid_y + new_coords.imag * new_coords.shape[1] / 2).astype(int)
    new_x = np.mod(np.rint(new_coords.real * new_coords.shape[0]).astype(int), new_coords.shape[0])
    new_y = np.mod(np.rint(new_coords.imag * new_coords.shape[1]).astype(int), new_coords.shape[1])

    for i in range(new_img.shape[0]):
        for j in range(new_img.shape[1]):
            new_img[i, j] = original_img[new_x[i, j], new_y[i, j]]

    return new_img

img = cv2.imread("frame.png", 0)
# img = cv2.imread("gallery.jpg", 0)

xstep = np.linspace(-1, 1, img.shape[0])
ystep = np.linspace(-1, 1, img.shape[1])
# xstep = np.arange(0, img.shape[0])
# ystep = np.arange(0, img.shape[1])
complex_grid = xstep[:, None] + 1j * ystep
# conformal = conformal_map(complex_grid, 1)
# 
# fig, axs = plt.subplots(1, 2)
# axs[0].imshow(np.abs(complex_grid), cmap="gray")
# axs[1].imshow(np.abs(conformal), cmap="gray")
# plt.show()

transformed_coords = conformal_map(complex_grid, 1)
new_img = sample_pixels(img, transformed_coords)

fig, axs = plt.subplots(1, 2)
axs[0].imshow(img, cmap="gray")
axs[1].imshow(new_img, cmap="gray")
plt.tight_layout()
plt.show()

# plt.imshow(np.abs(transformed_coords))
# plt.show()
