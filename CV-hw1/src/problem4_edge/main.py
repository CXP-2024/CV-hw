import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import cv2
from useful import my_conv2d, non_max_suppression, threshold, complete_canny_edge_detection, clip_show

def do_canny_edge_detection(img):
    """
    Implement canny edge detection for 2D image.
    :param img: float/int array, given image, shape: (height, width)
    :return detection results, a 2D image numpy array.
    """
    # =========================================================================================================
    # TODO: Please fill this part with your code, you can add other functions or files.
    # But DO NOT change this interface
    derivative_x_gaussian_kernel = np.array(
        [[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32
    )
    derivative_y_gaussian_kernel = np.array(
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32
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
    low_threshold = 50
    high_threshold = 100

    thresholded = threshold(suppressed, low_threshold, high_threshold)

    return thresholded

# =========================================================================================================


if __name__ == '__main__':
    path = 'src/problem4_edge/Hepburn.jpeg'
    img = mpimg.imread(path)

    # =========================================================================================================
    # for testing this demo:
    # using sobel kernel for edge detection
    ncc_map = do_canny_edge_detection(img)

    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(ncc_map, cmap="gray")
    plt.title("Thresholded Image")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
# =========================================================================================================


# using gaussian kernel for edge detection
# Apply the complete function and test for different sigma values and sizes
low_threshold = 50
high_threshold = 100
sigmas = [1.0, 10.0, 70.0]
size = [5, 23, 51]
finial_edges = []
for sigma in sigmas:
    for s in size:
        finial_edges.append(
            complete_canny_edge_detection(img, low_threshold, high_threshold, sigma, s)
        )

# Compare results
# show the size differences in the same row
# show the sigma differences in the same column
plt.figure(figsize=(30, 30))
for i, edge in enumerate(finial_edges):
    plt.subplot(3, 3, i + 1)
    plt.imshow(edge, cmap="gray")
    plt.title(f"sigma={sigmas[i//3]}, size={size[i%3]}")
plt.tight_layout()
plt.show()
