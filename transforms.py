import cv2
import matplotlib.pyplot as plt
import numpy as np
import pdb
import scipy.signal
import sys


def escher(img_path):
    img = cv2.imread(img_path)

    x = np.linspace(-1, 1, img.shape[1])
    y = np.linspace(-1, 1, img.shape[0])
    xx, yy = np.meshgrid(x, y)
    z = xx + yy * 1j

    r1 = 0.2
    r2 = 0.8

    zt = transform(z, r1, r2)
    new_img = change_coords(img, zt)
    new_img = blend_zeros(new_img)

    filename = f"{'.'.join(img_path.split('.')[:-1])}_escher.{img_path.split('.')[-1]}"
    cv2.imwrite(filename, new_img)


def transform(z, r1, r2):
    alpha = np.arctan(np.log(r2 / r1) / (2 * np.pi))
    f = np.cos(alpha)
    beta = f * np.exp(1j * alpha)

    annulus = np.where((np.abs(z) < r2) & (np.abs(z) > r1), np.log(z / r1), 0)
    annulus *= beta

    # h_disp = np.log(r2 / r1) - 1j * np.sin(alpha) * np.log(r2 / r1)
    h_disp = np.log(r2 / r1) - (np.tan(alpha) * np.log(r2 / r1) *
                                np.sin(alpha)) + (1j * np.sin(alpha) * np.log(r2 / r1))
    # v_disp = -2j * np.pi
    annulus = np.concatenate((
        annulus - h_disp,
        annulus,
        annulus + h_disp,
        annulus + 2 * h_disp,
        annulus + 3 * h_disp,
    ))
    # annulus = np.concatenate((annulus, annulus + v_disp))

    annulus = np.exp(annulus)

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

    for i in range(x_new.shape[0]):
        for j in range(x_new.shape[1]):
            new_img[y_new[i, j], x_new[i, j]] = img[i % img.shape[0], j]

    return new_img


def blend_zeros(arr):
    new_arr = np.copy(arr).astype(int)
    zero = np.nonzero(np.all(arr == [0, 0, 0], axis=-1))

    r = (np.mod(zero[0] + 1, arr.shape[0]), zero[1])
    l = (np.mod(zero[0] - 1, arr.shape[0]), zero[1])
    u = (zero[0], zero[1] - 1)
    d = (zero[0], np.mod(zero[1] + 1, arr.shape[1]))
    ur = (np.mod(zero[0] + 1, arr.shape[0]), zero[1] - 1)
    ul = (zero[0] - 1, zero[1] - 1)
    dr = (np.mod(zero[0] + 1, arr.shape[0]), np.mod(zero[1] + 1, arr.shape[1]))
    dl = (zero[0] - 1, np.mod(zero[1] + 1, arr.shape[1]))

    new_arr[zero] += arr[r]
    new_arr[zero] += arr[l]
    new_arr[zero] += arr[u]
    new_arr[zero] += arr[d]
    new_arr[zero] += arr[ur]
    new_arr[zero] += arr[ul]
    new_arr[zero] += arr[dr]
    new_arr[zero] += arr[dl]
    new_arr[zero] = new_arr[zero] / 8

    return new_arr.astype("uint8")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError("Pass a single path to an image")

    escher(sys.argv[1])
