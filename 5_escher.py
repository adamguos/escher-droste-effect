import cv2
import matplotlib.pyplot as plt
import numpy as np
import pdb
import sys


def escher(img_path):
    img = cv2.imread(img_path)

    x = np.linspace(-1, 1, img.shape[1])
    y = np.linspace(-1, 1, img.shape[0])
    xx, yy = np.meshgrid(x, y)
    z = xx + yy * 1j

    r1 = 0.5
    r2 = 1

    zt = transform(z, r1, r2)
    new_img = change_coords(img, zt)

    filename = f"{'.'.join(img_path.split('.')[:-1])}_escher.{img_path.split('.')[-1]}"
    cv2.imwrite(filename, new_img)


def transform(z, r1, r2):
    alpha = np.arctan(np.log(r2/r1) / (2 * np.pi))
    f = np.cos(alpha)
    beta = f * np.exp(1j * alpha)

    annulus = np.where((np.abs(z) < r2) & (np.abs(z) > r1), np.log(z/r1), 0)
    # annulus = z
    # annulus = np.log(annulus/r1)
    # annulus *= beta
    # annulus = np.exp(annulus)

    x_disp = np.log(r2/r1)
    # nulus = np.concatenate((annulus, annulus + x_disp))
    
    # plt.gca().set_aspect("equal")
    # plt.scatter(annulus.real.flatten(), annulus.imag.flatten())
    # plt.show()

    return annulus


def change_coords(img, new_coords):
    x = new_coords.real
    y = new_coords.imag
    scale = np.max([np.abs(x) / img.shape[1], np.abs(y) / img.shape[0]]) * 2

    x_new = x / scale + img.shape[1] / 2
    y_new = y / scale + img.shape[0] / 2
    x_new = x_new.astype(int)
    y_new = y_new.astype(int)
    x_new = np.clip(x_new, 0, img.shape[1] - 1)
    y_new = np.clip(y_new, 0, img.shape[0] - 1)

    new_img = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            new_img[y_new[i, j], x_new[i, j]] = img[i, j]

    pdb.set_trace()
    return new_img


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError("Pass a single path to an image")

    escher(sys.argv[1])
