import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
from useful import my_conv2d, my_conv3d, generate_gaussian_kernel, plot_2_fig, complete_canny_edge_detection

# functions are stored in useful.py
# it's better to put the output colorful images in full screen mode to see the details (since the image is large)
# =========================================================================================================

if __name__ == "__main__":
    kernel = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ],
        dtype=np.float32,
    )

    img = np.array(
        [
            [3, 5, 7, 6, 5, 5],
            [5, 5, 0, 1, 6, 6],
            [8, 7, 0, 0, 5, 3],
            [0, 8, 2, 8, 9, 6],
            [1, 2, 6, 7, 1, 6],
            [7, 7, 5, 3, 7, 7],
        ],
        dtype=np.float32,
    )

    left_move_kernel = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 0],
        ],
        dtype=np.float32,
    )

    # =========================================================================================================

# for testing this demo:
conv_type = "full"
print("Convolution Type:", conv_type)


my_result = my_conv2d(kernel, img, conv_type=conv_type)
print("Kernel:\n", kernel, "\nImages:\n", img, "\nMy Result:\n", my_result)
plot_2_fig(img, my_result, "Original Image", "Convolution Result (full)")

my_result = my_conv2d(kernel, img, conv_type="same")
print("Kernel:\n", kernel, "\nImages:\n", img, "\nMy Result:\n", my_result)
my_result = my_conv2d(kernel, img, conv_type="valid")
print("Kernel:\n", kernel, "\nImages:\n", img, "\nMy Result:\n", my_result)

left_result = my_conv2d(left_move_kernel, img, conv_type="same")
print("Left Move Result:\n", left_result)
plot_2_fig(img, left_result, "Original Image", "Left Move Result (same)")

# =========================================================================================================


# first test for colorful image
# change to [0,1] range for gaussian kernel convolution
path = "src/problem1_convolution/mjy.jpg"
img = mpimg.imread(path).astype(np.float32) / 255.0  

gaussian_kernel = generate_gaussian_kernel(21, 40.0) # 21* 21 kernel with sigma = 40.0
gaussian_color_kernel = np.stack([gaussian_kernel] * 3, axis=-1)  # changed to (h,w,3)
print("kernel total: ", gaussian_color_kernel.sum())  # 应为3.0
print("Gauss: \n", gaussian_color_kernel[..., 0])

blur_img = my_conv3d(gaussian_color_kernel, img, conv_type="same")

# Plot original image
plt.figure(figsize=(50, 30))
plt.imshow(img)
plt.title("Original Image")

# Plot convolution result
plt.figure(figsize=(50, 30))
plt.imshow(blur_img)
plt.title("Blurred Image")
plt.show()
# =========================================================================================================


# test cannny edge detection
# change to gray image
# Plot original image
plt.figure(figsize=(50, 30))
plt.imshow(img, cmap="gray")
plt.title("Original Image")

img = mpimg.imread(path)[:, :, 0]  # gray image
edge_result = complete_canny_edge_detection(img, sigma = 10.0, size = 5)


# Plot convolution result
plt.figure(figsize=(50, 30))
plt.imshow(edge_result, cmap="gray")
plt.title("Edge Detection Result")
plt.show()
# =========================================================================================================
