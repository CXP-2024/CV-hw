import numpy as np
import tqdm
import matplotlib.pyplot as plt


def my_conv2d(kernel, img, conv_type="same"):
    """
    Implement 2D image convolution
    :param kernel: float/int array, shape: (x, x)
    :param img: float/int array, shape: (height, width)
    :param conv_type: str, convolution padding choices, should in ['full', 'same', 'valid']
    :return conv results, numpy array
    """
    # Get dimensions
    k_h, k_w = kernel.shape
    i_h, i_w = img.shape

    # Rotate kernel by 180 degrees for convolution
    kernel = np.rot90(kernel, 2)

    # Calculate output dimensions based on conv_type
    half_k_h = k_h // 2
    half_k_w = k_w // 2
    if conv_type == "full":
        pad_h = k_h - 1
        pad_w = k_w - 1
        out_h = i_h + pad_h - 1
        out_w = i_w + pad_w - 1
    elif conv_type == "same":
        pad_h = (k_h - 1) // 2
        pad_w = (k_w - 1) // 2
        out_h = i_h
        out_w = i_w
    elif conv_type == "valid":
        pad_h = pad_w = 0
        out_h = i_h - k_h + 1
        out_w = i_w - k_w + 1
    else:
        raise ValueError("conv_type must be 'full', 'same', or 'valid'")

    # Pad the image
    padded_img = np.pad(
        img, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant", constant_values=0
    )

    # Perform convolution
    result = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            result[i, j] = np.sum(padded_img[i : i + k_h, j : j + k_w] * kernel)

    return result


def clip_show(img):
    max_val = np.percentile(img, 99)
    min_val = np.min(img)
    img = np.clip(img, min_val, max_val)
    img = (img - min_val) / (max_val - min_val) * 255
    return img.astype(np.uint8)


def plot_2_fig(
    origin_img,
    conv_img,
    title1,
    title2,
    cmap="gray",
    target_width=2000,
    target_height=1000,
    dpi=100,
):
    plt.figure(figsize=(target_width / dpi, target_height / dpi), dpi=dpi)

    # Plot original image
    plt.subplot(1, 2, 1)
    plt.imshow(origin_img, cmap=cmap)
    plt.title(title1)

    # Plot convolution result
    plt.subplot(1, 2, 2)
    plt.imshow(conv_img, cmap=cmap)
    plt.title(title2)

    plt.tight_layout()
    plt.show()


def generate_gaussian_kernel(size, sigma=1):
    """This function generates a 2D Gaussian kernel. And it output a 2D numpy array whose values are the Gaussian values between 0 and 1."""
    # generate an axis
    x = np.linspace(-size // 2, size // 2, size)
    gauss_x = np.exp(-(x**2) / (2 * sigma**2))

    # generate a 2D gaussian kernel and normalize
    gauss_2d = np.outer(gauss_x, gauss_x)
    gauss_2d /= gauss_2d.sum()
    return gauss_2d


def get_derivative_magnitude(img, size, sigma, strategy=1):
    if strategy == 1:
        # Strategy 1: Use the Sobel operator to calculate the gradient
        # change to gray scale
        img = np.mean(img, axis=2)
        derivative_x = my_conv2d(
            np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32) / 4.0, img
				)
        derivative_y = my_conv2d(
            np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32) / 4.0, img
				)
        derivative_magnitude = np.sqrt(derivative_x ** 2 + derivative_y ** 2)
        return derivative_magnitude
    
		# else strategy 2
    gaussian_kernel = generate_gaussian_kernel(size, sigma)
    derivative_x_kernel = (
        np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]], dtype=np.float32) / 2.0
    )
    derivative_y_kernel = (
        np.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]], dtype=np.float32) / 2.0
    )
    derivative_x_gaussian_kernel = my_conv2d(
        derivative_x_kernel, gaussian_kernel, conv_type="same"
    )
    derivative_y_gaussian_kernel = my_conv2d(
        derivative_y_kernel, gaussian_kernel, conv_type="same"
    )

    derivative_x = []
    derivative_y = []
    for i in tqdm.tqdm(range(img.shape[2])):
        derivative_x.append(
            my_conv2d(derivative_x_gaussian_kernel, img[:, :, i], conv_type="same")
        )
        derivative_y.append(
            my_conv2d(derivative_y_gaussian_kernel, img[:, :, i], conv_type="same")
        )
        derivative_x[i] = clip_show(derivative_x[i])
        derivative_y[i] = clip_show(derivative_y[i])

    derivative_magnitude = np.zeros(img.shape[:2])
    for i in range(img.shape[2]):
        derivative_magnitude += np.sqrt(
            derivative_x[i] ** 2 + derivative_y[i] ** 2
        )  # sum of all channels

    return derivative_magnitude


def find_min(arr):
    min_index = 0
    for i in range(len(arr)):
        if arr[i] < arr[min_index]:
            min_index = i
    return min_index, arr[min_index]


def find_y_seam_box(img, size=5, sigma=1.0):
    h, w, c = img.shape  # Fix: correctly unpack as height, width, channels
    derivative_magnitude = get_derivative_magnitude(img, size, sigma)
    energy_map = derivative_magnitude / np.max(derivative_magnitude)  # change to 0-1

    img_padded = np.pad(
        energy_map, ((0, 0), (1, 1)), mode="constant", constant_values=10
    )
    # create a point matrix (h* w) to store its link point to the previous row, which means a[x, y] = z means the point (i, j) is the previous point of (i - 1, j + z - 1)
    point_matrix = np.zeros((h, w), int)

    for i in range(1, h):
        for j in range(w):
            min_point, min_val = find_min(img_padded[i - 1, j : j + 3])  # 0, 1, 2
            point_matrix[i, j] = j + min_point - 1
            energy_map[i, j] += min_val

    # find the seam with the smallest energy
    finial_y_index = find_min(energy_map[-1, :])[0]
    seam_box = []
    for i in range(h - 1, -1, -1):  # from bottom to top
        seam_box.append((i, finial_y_index))
        finial_y_index = point_matrix[i, finial_y_index]
    seam_box.reverse()

    return seam_box


def remove_y_seam(img, seam_box):
    h, w, c = img.shape
    reduced_img = np.zeros((h, w - 1, c))
    for i in range(h):
        seam = seam_box[i]
        reduced_img[i, :, :] = np.delete(img[i, :, :], seam[1], axis=0)
    return reduced_img

def find_x_seam_box(img, size = 5, sigma = 1.0):
    img = np.transpose(img, (1, 0, 2))
    seam_box = find_y_seam_box(img, size, sigma)
    seam_box = [(y, x) for x, y in seam_box]
    return seam_box

def remove_x_seam(img, seam_box):
    h, w, c = img.shape
    reduced_img = np.zeros((h - 1, w, c))
    for i in range(w):
        seam = seam_box[i]
        reduced_img[:, i, :] = np.delete(img[:, i, :], seam[0], axis=0)
    return reduced_img
