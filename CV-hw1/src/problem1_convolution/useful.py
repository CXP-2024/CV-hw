import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import tqdm

def rotate180(kernel):
    return np.rot90(kernel, 2)

def pad(img, pad_h, pad_w):
    return np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), "constant", constant_values=0)

def compute_pix(img, kernel):
    return np.sum(np.multiply(img, kernel))

def erase_extra(result_expand, half_k_h, half_k_w):
    return result_expand[half_k_h:-half_k_h, half_k_w:-half_k_w]

def my_conv2d(kernel, img, conv_type="full"):
    """
    Implement 2D image convolution
    :param kernel: float/int array, shape: (x, x)
    :param img: float/int array, shape: (height, width)
    :param conv_type: str, convolution padding choices, should in ['full', 'same', 'valid']
    :return conv results, numpy array
    """
    k_h, k_w = kernel.shape
    i_h, i_w = img.shape

    kernel = rotate180(kernel)  # Rotate kernel by 180 degrees for convolution

    # Calculate output dimensions based on conv_type
    half_k_h = k_h // 2
    half_k_w = k_w // 2
    if conv_type == "full":
        pad_h = k_h - 1
        pad_w = k_w - 1
    elif conv_type == "same":
        pad_h = (k_h - 1) // 2
        pad_w = (k_w - 1) // 2
    elif conv_type == "valid":
        pad_h = pad_w = 0
    else:
        raise ValueError("conv_type must be 'full', 'same', or 'valid'")

    out_h = i_h + 2 * pad_h
    out_w = i_w + 2 * pad_w
    # Pad the image, put it in the center with 0 outside
    padded_img = pad(img, pad_h, pad_w)

    # Perform convolution
    result_expand = np.zeros((out_h, out_w))
    for i in tqdm.tqdm(range(half_k_h, out_h - half_k_h)):
        for j in range(half_k_w, out_w - half_k_w):
            result_expand[i, j] = compute_pix(
                padded_img[
                    i - half_k_h : i + half_k_h + 1, j - half_k_w : j + half_k_w + 1
                ],
                kernel,
            )

    result = erase_extra(result_expand, half_k_h, half_k_w)
    pass
    # =========================================================================================================
    return result

def my_conv3d(kernel, img, conv_type="full"):
    """
    Implement 3D image convolution
    :param kernel: float/int array, shape: (h, w, c)
    :param img: float/int array, shape: (height, width, channel)
    :param conv_type: str, convolution padding choices, should in ['full', 'same', 'valid']
    :return conv results, numpy array
    """
    assert len(kernel.shape) == 3 and len(img.shape) == 3, "The dimensions of kernel and img should be 3."
    i_h, i_w, i_c = img.shape

    result = np.zeros((i_h, i_w, i_c))
    for ch in range(i_c):
        result[:, :, ch] = my_conv2d(kernel[:, :, ch], img[:, :, ch], conv_type=conv_type)
        
    return result

def generate_gaussian_kernel(size, sigma = 1):
    '''This function generates a 2D Gaussian kernel. And it output a 2D numpy array whose values are the Gaussian values between 0 and 1.'''
    # generate an axis
    x = np.linspace(-size // 2, size // 2, size)
    gauss_x = np.exp(-x ** 2 / (2 * sigma ** 2))

    # generate a 2D gaussian kernel and normalize
    gauss_2d = np.outer(gauss_x, gauss_x)
    gauss_2d /= gauss_2d.sum()
    return gauss_2d


def plot_2_fig(
    origin_img,
    conv_img,
    title1,
    title2,
    cmap="gray",
    target_width=1200,
    target_height=700,
    dpi=100,
):
    plt.figure(figsize=( target_width / dpi,  target_height / dpi), dpi=dpi)

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


def non_max_suppression(gradient_magnitude, gradient_angle):
    """
    Perform non-maximum suppression on the gradient magnitude image

    Args:
            gradient_magnitude: numpy array of gradient magnitudes
            gradient_angle: numpy array of gradient angles in degrees [0, 180]

    Returns:
            numpy array of same size as input with non-maximum suppression applied
    """
    M, N = gradient_magnitude.shape
    suppressed = np.zeros((M, N), dtype=gradient_magnitude.dtype)

    # Map angles to 4 directions: 0, 45, 90, 135 degrees
    # 0-22.5 & 157.5-180 -> 0 degrees (horizontal)
    # 22.5-67.5 -> 45 degrees (diagonal)
    # 67.5-112.5 -> 90 degrees (vertical)
    # 112.5-157.5 -> 135 degrees (diagonal)

    # Initialize to 0 degrees
    quantized_angle = np.zeros((M, N), dtype=np.uint8)

    # Quantize the angles
    make_0 = (gradient_angle <= 22.5) | (gradient_angle > 157.5)
    quantized_angle[make_0] = 0
    mask_45 = (gradient_angle > 22.5) & (gradient_angle <= 67.5)
    quantized_angle[mask_45] = 45
    mask_90 = (gradient_angle > 67.5) & (gradient_angle <= 112.5)
    quantized_angle[mask_90] = 90
    mask_135 = (gradient_angle > 112.5) & (gradient_angle <= 157.5)
    quantized_angle[mask_135] = 135

    # Pad the gradient magnitude to handle border pixels
    padded_magnitude = np.pad(
        gradient_magnitude, ((1, 1), (1, 1)), mode="constant", constant_values=0
    )

    # For each pixel, check if it's a local maximum along the gradient direction
    for i in range(M):
        for j in range(N):
            # Get the angle at this pixel
            a = quantized_angle[i, j]

            # Compute indices in the padded image
            pi, pj = i + 1, j + 1

            # Check neighbors based on gradient direction
            if a == 0:  # Horizontal gradient
                if (
                    padded_magnitude[pi, pj] >= padded_magnitude[pi, pj - 1]
                    and padded_magnitude[pi, pj] >= padded_magnitude[pi, pj + 1]
                ):
                    suppressed[i, j] = gradient_magnitude[i, j]
            elif a == 45:  # Diagonal gradient (top-right to bottom-left)
                if (
                    padded_magnitude[pi, pj] >= padded_magnitude[pi - 1, pj + 1]
                    and padded_magnitude[pi, pj] >= padded_magnitude[pi + 1, pj - 1]
                ):
                    suppressed[i, j] = gradient_magnitude[i, j]
            elif a == 90:  # Vertical gradient
                if (
                    padded_magnitude[pi, pj] >= padded_magnitude[pi - 1, pj]
                    and padded_magnitude[pi, pj] >= padded_magnitude[pi + 1, pj]
                ):
                    suppressed[i, j] = gradient_magnitude[i, j]
            else:  # a == 135, Diagonal gradient (top-left to bottom-right)
                if (
                    padded_magnitude[pi, pj] >= padded_magnitude[pi - 1, pj - 1]
                    and padded_magnitude[pi, pj] >= padded_magnitude[pi + 1, pj + 1]
                ):
                    suppressed[i, j] = gradient_magnitude[i, j]

    return suppressed


def threshold(gradient_magnitude, low_threshold, high_threshold):
    """
    Apply double thresholding to the gradient magnitude image

    Args:
            gradient_magnitude: numpy array of gradient magnitudes
            low_threshold: low threshold value
            high_threshold: high threshold value

    Returns:
            numpy array of same size as input with thresholding applied
    """
    M, N = gradient_magnitude.shape
    result = np.zeros((M, N), dtype=np.uint8)

    # Define weak and strong pixel values
    weak = 60
    medium = 150
    strong = 255

    # Define the 8-connectivity offsets
    offsets = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]

    # Initialize the result with strong pixels
    for i in range(M):
        for j in range(N):
            if gradient_magnitude[i, j] >= high_threshold:
                result[i, j] = strong

    # Find weak pixels
    weak_i, weak_j = [], []
    for i in range(M):
        for j in range(N):
            if (
                gradient_magnitude[i, j] >= low_threshold
                and gradient_magnitude[i, j] < high_threshold
            ):
                result[i, j] = weak
                weak_i.append(i)
                weak_j.append(j)

    # define a map which map a point to a bool value
    searched = np.ones((M, N), dtype=bool)
    for i, j in zip(weak_i, weak_j):
        searched[i, j] = False

    # Check medium pixels, we should convert all those edges which are connected to strong edges to medium edges
    for i, j in zip(weak_i, weak_j):
        if searched[i, j]:
            continue
        searched[i, j] = True
        weak_group = [(i, j)]
        pending_to_change = [(i, j)]
        change_to_medium = False  # only when the group is connected to strong edges, we should convert them to medium edges
        while weak_group:
            i, j = weak_group.pop()
            for offset_i, offset_j in offsets:
                new_i, new_j = i + offset_i, j + offset_j
                if 0 <= new_i < M and 0 <= new_j < N:
                    if result[new_i, new_j] == weak and not searched[new_i, new_j]:
                        weak_group.append((new_i, new_j))
                        pending_to_change.append((new_i, new_j))
                        searched[new_i, new_j] = True
                    elif (
                        result[new_i, new_j] == strong or result[new_i, new_j] == medium
                    ):
                        change_to_medium = True
                        break
            if change_to_medium:
                break
        if change_to_medium:
            for i, j in pending_to_change:
                result[i, j] = medium

    return result


def clip_show(img):
    max_val = np.percentile(img, 99)
    min_val = np.min(img)
    img = np.clip(img, min_val, max_val)
    img = (img - min_val) / (max_val - min_val) * 255
    return img


def complete_canny_edge_detection(
    img, low_threshold=70, high_threshold=125, sigma=10.0, size=5
):
    """Complete Canny edge detection pipeline"""

    gaussian_kernel = generate_gaussian_kernel(size, sigma)
    derivative_x_kernel = (
        np.array([[0, 0, 0], [1, 0, -1], [0, 0, 0]], dtype=np.float32) / 2.0
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

    # 1. Compute gradients
    derivative_x_gaussian_result = my_conv2d(
        derivative_x_gaussian_kernel, img, conv_type="same"
    )
    derivative_y_gaussian_result = my_conv2d(
        derivative_y_gaussian_kernel, img, conv_type="same"
    )

    # 2. Calculate gradient magnitude and direction
    gradient_magnitude = np.sqrt(
        derivative_x_gaussian_result**2 + derivative_y_gaussian_result**2
    )
    gradient_magnitude = clip_show(gradient_magnitude)

    gradient_angle = np.arctan2(
        derivative_y_gaussian_result, derivative_x_gaussian_result
    )
    gradient_angle = np.degrees(gradient_angle) % 180

    # 3. Non-maximum suppression
    suppressed = non_max_suppression(gradient_magnitude, gradient_angle)

    # 4. Thresholding
    thresholded = threshold(suppressed, low_threshold, high_threshold)

    return thresholded
