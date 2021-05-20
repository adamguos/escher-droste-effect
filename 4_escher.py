import cv2
import matplotlib.pyplot as plt
import numpy as np
import pdb
import scipy.signal


def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def gen_circle(radius, inner_radius=0):
    rng = np.random.default_rng()
    # length = np.sqrt(np.linspace(inner_radius**2, radius**2, 200))
    # angle = np.pi * np.linspace(0, 2, 1000)
    length = rng.uniform(inner_radius**2, radius**2, 500) ** (1/2)
    angle = np.pi * rng.uniform(0, 2, 5000)

    x = length[None, :] * np.cos(angle)[:, None]
    y = length[None, :] * np.sin(angle)[:, None]

    return x.flatten() + y.flatten() * 1j


def sample(old_coords, new_coords, img):
    scale = min(img.shape) / 2
    center = img.shape[0]/2 + img.shape[1]/2 * 1j

    new_coords /= max([new_coords.real.max(), np.abs(new_coords.real.min()),
                      new_coords.imag.max(), np.abs(new_coords.imag.min())])

    old_x = np.rint((old_coords * scale + center).real).astype(int)
    old_y = np.rint((old_coords * scale + center).imag).astype(int)

    new_x = np.rint((new_coords * scale + center).real).astype(int)
    new_y = np.rint((new_coords * scale + center).imag).astype(int)
    x_min, y_min = new_x.min(), new_y.min()

    new_img = np.zeros((new_x.max() - x_min + 1,
                       new_y.max() - y_min + 1))

    for i in range(len(new_x)):
        new_img[new_x[i] - x_min, new_y[i] - y_min] = img[old_x[i], old_y[i]]
    
    # i = 0
    # while np.sum(new_img == 0) > 0:
        # print(f"avg {i}, zeros {np.sum(new_img == 0)}")
        # i += 1
        # new_img = avg_zeros(new_img)
    
    return new_img


def avg_zeros(arr):
    # Define kernel for convolution                                         
    kernel = np.array([[0,1,0],
                       [1,0,1],
                       [0,1,0]]) 
    
    # Perform 2D convolution with input data and kernel 
    conv_out = scipy.signal.convolve2d(arr, kernel, boundary='wrap', mode='same')/kernel.sum()
    
    # Initialize output array as a copy of input array
    arr_out = arr.copy()
    
    # Setup a mask of zero elements in input array and 
    # replace those in output array with the convolution output
    mask = arr==0
    arr_out[mask] = conv_out[mask]

    return arr_out


def escher(r1, r2):
    # http://www.josleys.com/article_show.php?id=82
    alpha = np.arctan(np.log(r2/r1) / (2 * np.pi))
    f = np.cos(alpha)
    beta = f * np.exp(1j * alpha)

    circle = gen_circle(r2, r1)
    transformed = np.log(circle / r1)
    transformed *= beta

    h_disp = (2*np.pi - np.sin(alpha) * np.log(r2/r1))
    v_disp = np.log(r2/r1)
    disp = v_disp + np.sin(alpha) * np.log(r2/r1) * 1j
    # transformed = np.concatenate((transformed, transformed + h_disp, transformed + 2*h_disp), axis=0)
    # transformed = np.concatenate((transformed, transformed + v_disp, transformed + 2*v_disp), axis=0)
    transformed = np.concatenate((transformed, transformed + disp))
    # plt.scatter(transformed.real, transformed.imag)
    # plt.show()
    transformed = np.exp(transformed)

    circle = np.tile(circle, 2)

    return circle, transformed


def escher_inv(r1, r2):
    x_step = np.linspace(-1, 1, 500)
    y_step = np.linspace(-1, 1, 500)
    grid = (x_step[None, :] + y_step[:, None] * 1j).flatten()

    alpha = np.arctan(np.log(r2/r1) / (2 * np.pi))
    f = np.cos(alpha)
    beta = f * np.exp(1j * alpha)

    inverse = np.log(grid)

    plt.scatter(inverse.real, inverse.imag, s=0.01)
    plt.gca().set_aspect("equal")
    plt.show()


img = cv2.imread("frame.png", 0)
# circle, esch = escher(0.0632716, 0.2777777)
circle, esch = escher(0.1, 0.4)
esch = sample(circle, esch, img)
cv2.imwrite("escher.png", esch.astype("uint8"))

# escher_inv(0.2, 0.4)