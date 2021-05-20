import cv2
import matplotlib.pyplot as plt
import numpy as np
import pdb


def conformal(orig_img):
    mindim = min(orig_img.shape[0], orig_img.shape[1])
    xstep = np.linspace(np.exp(-16), np.log((mindim - 1) / 2), mindim)
    ystep = np.linspace(2 * np.pi, -2 * np.pi, mindim)
    coords = xstep[None, :] + 1j * ystep[:, None]

    new_coords = np.exp(coords)
    # new_x = np.mod(np.rint(new_coords.real * orig_img.shape[0] * 2).astype(int), orig_img.shape[0])
    # new_y = np.mod(np.rint(new_coords.imag * orig_img.shape[1] * 2).astype(int), orig_img.shape[1])
    new_x = np.rint(new_coords.real).astype(int)
    new_y = np.rint(new_coords.imag).astype(int)
    new_x -= new_x.min()
    new_y -= new_y.min()

    if orig_img.ndim == 3:
        con = np.zeros((coords.shape[0], coords.shape[1], orig_img.shape[2]))
    else:
        con = np.zeros((coords.shape[0], coords.shape[1]))

    for i in range(con.shape[0]):
        for j in range(con.shape[1]):
            con[i, j] = orig_img[new_x[i, j], new_y[i, j]]

    return con


img = cv2.imread("frame.png")
con = conformal(img)
cv2.imshow("image", con.astype("uint8"))
cv2.waitKey()
