import matplotlib.image as mpimg
import numpy as np
import random
import matplotlib.pyplot as plt

# in donoise_notebook.ipynb, we have same code which maybe prettier

def add_salt_pepper_noise(img, prob):
    result = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if random.random() < prob:
                result[i][j] = 0 if random.random() < 0.5 else 255
    return result


def add_impulse_noise(img, prob):
    result = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if random.random() < prob:
                result[i][j] = 255  # Only add white noise for impulse
    return result


def add_gaussian_noise(img, mean, std):
    # randomly select a sample from a normal distribution for each pixel
    noise = np.random.normal(mean, std, img.shape).astype(np.uint8)
    result = img.copy() + noise
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


def median_filter(img):
    def median(window):
        array = np.sort(window.flatten())
        # flatten means the 3x3 window become 1x9 array, then sort it to get median
        return array[len(array) // 2]

    # we use default window size 3x3
    result = img.copy()
    pad_img = np.pad(img, ((1, 1), (1, 1)), "constant", constant_values=0)

    for i in range(0, img.shape[0] - 2):
        for j in range(0, img.shape[1] - 2):
            window = pad_img[i : i + 3, j : j + 3]
            result[i][j] = median(window)

    return result


def mean_filter(img):
    def mean(window):
        return np.mean(window)

    result = img.copy()
    pad_img = np.pad(img, ((1, 1), (1, 1)), "constant", constant_values=0)

    for i in range(0, img.shape[0] - 2):
        for j in range(0, img.shape[1] - 2):
            window = pad_img[i : i + 3, j : j + 3]
            result[i][j] = mean(window)

    return result


def psnr(img1, img2):
    # Implement PSNR calculation
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    max_pixel = 255.0
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr_value


if __name__ == "__main__":
    path = "src/problem2_denoise/Hepburn.jpeg"
    img = mpimg.imread(path)
    print(img)
    # =========================================================================================================
    # TODO: 1. add noise; 2. denoise with mean/median fiter; 3. calculate PSNR.
    # You are free to design your own code&format.

# Display original image
plt.figure(figsize=(4, 4))
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")
plt.show()

# Add salt and pepper noise
salt_pepper_img = add_salt_pepper_noise(img, 0.1)

# Display noisy image and the original image
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(salt_pepper_img, cmap="gray")
plt.title("Salt and Pepper Noise")
plt.axis("off")

# Apply filters
median_denoised_img = median_filter(salt_pepper_img)
mean_denoised_img = mean_filter(salt_pepper_img)

# Display denoised images
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(salt_pepper_img, cmap="gray")
plt.title("Noisy Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(median_denoised_img, cmap="gray")
plt.title(f"Median Filter (PSNR: {psnr(img, median_denoised_img):.2f}dB)")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(mean_denoised_img, cmap="gray")
plt.title(f"Mean Filter (PSNR: {psnr(img, mean_denoised_img):.2f}dB)")
plt.axis("off")

plt.tight_layout()
plt.show()


# Add impulse noise
impulse_img = add_impulse_noise(img, 0.1)

# Display noisy image and the original image
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(impulse_img, cmap="gray")
plt.title("Impulse Noise")
plt.axis("off")

# Apply filters
median_denoised_img = median_filter(impulse_img)
mean_denoised_img = mean_filter(impulse_img)

# Display denoised images
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(impulse_img, cmap="gray")
plt.title("Noisy Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(median_denoised_img, cmap="gray")
plt.title(f"Median Filter (PSNR: {psnr(img, median_denoised_img):.2f}dB)")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(mean_denoised_img, cmap="gray")
plt.title(f"Mean Filter (PSNR: {psnr(img, mean_denoised_img):.2f}dB)")
plt.axis("off")

plt.tight_layout()
plt.show()


# Add gaussian noise
gaussian_img = add_gaussian_noise(img, 0, 25)

# Display noisy image and the original image
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(gaussian_img, cmap="gray")
plt.title("Gaussian Noise")
plt.axis("off")

# Apply filters
median_denoised_img = median_filter(gaussian_img)
mean_denoised_img = mean_filter(gaussian_img)

# Display denoised images
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(gaussian_img, cmap="gray")
plt.title("Noisy Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(median_denoised_img, cmap="gray")
plt.title(f"Median Filter (PSNR: {psnr(img, median_denoised_img):.2f}dB)")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(mean_denoised_img, cmap="gray")
plt.title(f"Mean Filter (PSNR: {psnr(img, mean_denoised_img):.2f}dB)")
plt.axis("off")

plt.tight_layout()
plt.show()
# =========================================================================================================
