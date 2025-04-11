import numpy as np

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

pad_h = 1
pad_w = 2
a1 = img[0:3, 0:2]
a2 = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), "constant", constant_values=0)
a3 = img[1:-1, 1:-1]
a4 = a3.flatten()
a5 = np.sort(a4)
a6 = np.random.normal(10, 7, (3, 3))
print(a1)
print(a2)
print(a3)
print(a4)
print(a5)
print(a6)

x = [1, 2, 3, 4, 5]
y = [3, 4, 5, 6, 7]
is_searched = [False] * 5
searched = list(zip(x, y, is_searched))  # Convert to list immediately
print(searched)
# No need to print list(searched) again since it's already a list
# Access by index, not by tuple key
# For example, to get the item at index 2:
# Find an item where the second element is 5 and the third element is False
item = next((item for item in searched if item[0] == 5 and item[1] == 7))
print(item[2])

# Or to access a specific index directly:
print(searched[2])  # Prints the third item in the list
