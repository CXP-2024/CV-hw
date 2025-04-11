import cv2
import numpy as np
import matplotlib.pyplot as plt


def diff_ssd(source_patch, template_img):
    if source_patch.shape != template_img.shape:
        raise ValueError("Shapes must be identical!")
    return np.sum((source_patch - template_img) ** 2)


def diff_ncc(source_patch, template_img):
    """Calculate Normalized Cross Correlation between source patch and template."""
    if source_patch.shape != template_img.shape:
        raise ValueError("Shapes must be identical!")

    source_patch = source_patch.astype(np.float64)
    template_img = template_img.astype(np.float64)

    source_patch -= np.mean(source_patch)
    template_img -= np.mean(template_img)

    numerator = np.sum(source_patch * template_img)
    denom_src = np.sqrt(np.sum(source_patch**2))
    denom_tpl = np.sqrt(np.sum(template_img**2))
    denominator = denom_src * denom_tpl

    if denominator == 0:
        if np.all(source_patch == 0) and np.all(template_img == 0):
            return 1.0  # totally the same
        else:
            return 0.0  # I dont't know what to do here

    return numerator / denominator


def NCC_map(source_img, template_img):
    """generate NCC feature map and show the best match"""

    s_h, s_w, s_c = source_img.shape
    t_h, t_w, t_c = template_img.shape

    # Initial
    ncc_map = np.zeros((s_h - t_h + 1, s_w - t_w + 1))

    max_ncc = -1.0
    best_position = (0, 0)

    for i in range(s_h - t_h + 1):
        for j in range(s_w - t_w + 1):

            source_patch = source_img[i : i + t_h, j : j + t_w]
            ncc = diff_ncc(source_patch, template_img)
            ncc_map[i, j] = ncc

            if ncc > max_ncc:
                max_ncc = ncc

                best_position = (i, j)

    # plot
    plt.figure(figsize=(12, 6))

    # show
    plt.subplot(1, 2, 1)
    ncc_display = (ncc_map + 1) * 127.5  # map to [0, 255]
    plt.imshow(ncc_display, cmap="gray")
    plt.colorbar()
    plt.title("NCC Feature Map")
    plt.axis("off")

    # pick
    plt.subplot(1, 2, 2)
    marked_img = source_img.copy()
    top_left = (best_position[1], best_position[0])  # (x,y)
    bottom_right = (top_left[0]+t_w, top_left[1]+t_h)
    cv2.rectangle(marked_img, top_left, bottom_right, (0, 0, 0), 10)
    plt.imshow(marked_img)
    plt.title(f"Best Match (NCC={max_ncc:.3f})")
    plt.axis("off")
    plt.show()

    print(f"Best position: ({best_position[0]}, {best_position[1]})")

    return ncc_display


if __name__ == "__main__":
    source_path = "src/problem3_template/waldo.jpg"
    template_path = "src/problem3_template/template.jpg"

    # H, W, C (BGR)
    source_img = cv2.imread(source_path)
    template_img = cv2.imread(template_path)
    print("source_img shape: ", source_img.shape)
    print("template_img shape: ", template_img.shape)
    # print(source_img)
    # print("template:", template_img)

    # change BGR to RGB
    source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
    template_img = cv2.cvtColor(template_img, cv2.COLOR_BGR2RGB)

    # =========================================================================================================
    # for testing this demo:
    ncc_map = NCC_map(source_img, template_img)


plt.figure(figsize=(15, 5))

# Plot original image
plt.subplot(1, 3, 1)
plt.imshow(source_img)
plt.title("Original Image")
plt.axis("off")

# Plot convolution result
plt.subplot(1, 3, 2)
plt.imshow(ncc_map, cmap="gray")
plt.title(f"Convolution Result (same)")
plt.axis("off")


# Plot the template image
plt.subplot(1, 3 , 3)
plt.imshow(template_img)
plt.title("Template Image")
plt.axis("off")

plt.tight_layout()
plt.show()

# =========================================================================================================
