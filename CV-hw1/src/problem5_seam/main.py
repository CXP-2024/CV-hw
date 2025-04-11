import cv2
from useful import find_y_seam_box, remove_y_seam, find_x_seam_box, remove_x_seam
import tqdm as tqdm
import numpy as np

#######################################################
##########       This is very slow !!!! 		 ##########
##########       This is very slow !!!! 		 ##########
##########       This is very slow !!!! 		 ##########
##########       This is very slow !!!! 		 ##########
##########       This is very slow !!!! 		 ##########
##########      See seam_carving.ipynb	!	   ##########
#######################################################

class SeamCarver(object):
    # =========================================================================================================
    # TODO: Please fill this class with your code, you can add other functions or files.
    # =========================================================================================================
    def __init__(self):
        pass

    def reduce_height(self, img, num_pixels):
        """
        Implement reduce height part for 3D image in paper "Seam Carving for Content-Aware Image Resizing".
        :param img: float/int array, given image, shape: (height, width, channel)
        :param num_pixels: int, the number of reduced height.
        :return reduced results, a 3D image numpy array (height, width, channel).
        """
        # =========================================================================================================
        # TODO: Please fill this part with your code, you can add other functions or files.
        # But DO NOT change this interface
        reduced_img = img.copy()
        for i in tqdm.tqdm(range(num_pixels)):
            seam_box = find_x_seam_box(reduced_img, size = 3, sigma = 1.0)
            reduced_img = remove_x_seam(reduced_img, seam_box)
        pass
        # =========================================================================================================
        return reduced_img

    def reduce_width(self, img, num_pixels):
        """
        Implement reduce width part for 3D image in paper "Seam Carving for Content-Aware Image Resizing".
        :param img: float/int array, given image, shape: (height, width, channel)
        :param num_pixels: int, the number of reduced width.
        :return reduced results, a 3D image numpy array (height, width, channel).
        """
        # =========================================================================================================
        # TODO: Please fill this part with your code, you can add other functions or files.
        # But DO NOT change this interface
        reduced_img = img.copy()
        for i in tqdm.tqdm(range(num_pixels)):
            seam_box = find_y_seam_box(reduced_img, size = 3, sigma = 1.0)
            reduced_img = remove_y_seam(reduced_img, seam_box)
        pass
        # =========================================================================================================
        return reduced_img


if __name__ == "__main__":
    path = "src/problem5_seam/motor.jpg"
    img = cv2.imread(path)

    # change BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    seam_carver = SeamCarver()
    num_pixels = 100
    # =========================================================================================================
    # for testing this demo:
    reduced_height_img = seam_carver.reduce_height(img, num_pixels)
    reduced_width_img = seam_carver.reduce_width(img, num_pixels)
    # =========================================================================================================


# show
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.imshow(img.astype(np.uint8))
plt.title(f"Original Image {img.shape[1]} * {img.shape[0]}")

plt.subplot(1, 2, 2)
plt.imshow(reduced_height_img.astype(np.uint8))
plt.title(
    f"Height Reduced by {num_pixels} pixels {reduced_height_img.shape[1]} * {reduced_height_img.shape[0]}"
)

plt.show()

plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.imshow(img.astype(np.uint8))
plt.title(f"Original Image {img.shape[1]} * {img.shape[0]}")

plt.subplot(1, 2, 2)
plt.imshow(reduced_width_img.astype(np.uint8))
plt.title(
		f"Width Reduced by {num_pixels} pixels {reduced_width_img.shape[1]} * {reduced_width_img.shape[0]}"
)

plt.show()